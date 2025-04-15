#!/usr/bin/env python3

import numpy as np
import math
import matplotlib.pyplot as plt
from rclpy.node import Node

import rclpy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from controller import PID
from controller import PID  

from datetime import datetime
import os

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from sklearn.linear_model import RANSACRegressor



def save_point_cloud(x, y, save_dir="/home/sonieth/ros/ros2/ros2_ws/src/ldrobot-lidar-ros2/ldlidar_node/pics"):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    filename = f"pic_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    
    # Create plot
    fig, ax = plt.subplots()
    ax.plot(x, y, '.', markersize=2, color='blue')
    ax.set_aspect('equal')
    
    # Add labels and title (optional)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Lidar Point Cloud')
    
    # Save and clean up
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved point cloud to: {save_path}")


class WallFollower(Node):
    # Import ROS parameters from the "params.yaml" file.
    # Access these variables in class functions with self:
    # i.e. self.CONSTANT
    SCAN_TOPIC = "/ldlidar_node/scan"
    DRIVE_TOPIC = "cmd_vel"
    
    VELOCITY = 0.6
    

    def __init__(self):
        super().__init__('wall_follower')

        self.SIDE = +1 # +1 right || -1 is left

        self.subscription = self.create_subscription( LaserScan, self.SCAN_TOPIC,self.LaserCb, 1)
        

        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 1)
        self.drive_pub = self.create_publisher(Twist, self.DRIVE_TOPIC, 10)

        # Variables to keep track of drive commands being sent to robot.
        self.seq_id = 0

        # Class variables for following.
        # self.side_angle_window_fwd_ = math.pi*0.1
        # self.side_angle_window_bwd_ = math.pi - math.pi*0.3

        self.point_buffer_x_ = np.array([])
        self.point_buffer_y_ = np.array([])
        self.num_readings_in_buffer_ = 0
        
        self.filtered_points_pub = self.create_publisher(Marker, '/filtered_points_marker', 10)

        self.scan_buffer = []  # Buffer to hold last 6 scans' (x, y) points
        self.num_scans_to_avg = 8
        
        #TODO: add thresh distance to wall where we should start rotation behaviour

        self.DESIRED_DISTANCE = 0.4 #TODO: add thresh distance to wall left/right side, to keep it
        self.min_reject_dist = 0.0 #0.25
        self.max_reject_dist = 0.9

        self.steer_cmd = 0
        self.vel_cmd = self.VELOCITY

        self.PID = PID()
    
    def GetLocalWallCoords(self, ranges, angles):
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        distances = np.sqrt(x**2 + y**2)
        mask =(self.min_reject_dist <= distances) & (distances <= self.max_reject_dist) & ( x  * self.SIDE > 0)
        # filtered with mask
        return x[mask], y[mask]

    def PublishWallLine(self, m, c):
        marker = Marker()
        marker.header.frame_id = "ldlidar_base"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        x_vals = np.array([-1.0, 1.0])
        y_vals = m * x_vals + c

        for xi, yi in zip(x_vals, y_vals):
            p = Point()
            p.x = xi
            p.y = yi
            p.z = 0.0
            marker.points.append(p)

        self.marker_pub.publish(marker)

    def publish_filtered_scan(self, original_msg, avg_x, avg_y):
        # Create Marker message for RViz visualization
        marker = Marker()
        marker.header = original_msg.header
        marker.header.frame_id = "ldlidar_base"  # Match your LiDAR frame
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()  # Short lifetime
        
        # Purple color (RGB: 1.0, 0.0, 1.0)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0  # Full opacity
        
        # Point size (adjust as needed)
        marker.scale.x = 0.05  # 5 cm points
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        # Add all averaged points
        for x, y in zip(avg_x, avg_y):
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0  # Assuming 2D LiDAR
            marker.points.append(p)

        # Publish the marker
        self.filtered_points_pub.publish(marker)


    def average_scans(self, scan_buffer):
        all_x = np.concatenate([scan[0] for scan in scan_buffer])
        all_y = np.concatenate([scan[1] for scan in scan_buffer])
        
        # Average duplicates by rounding coordinates (adjust tolerance as needed)
        # This groups nearby points to avoid oversampling the same location
        decimals = 2  # Adjust based on your LiDAR's resolution
        unique_points, indices = np.unique(
            np.round(np.column_stack((all_x, all_y)), decimals),
            axis=0,
            return_inverse=True
        )
        
        # Compute mean for each unique point
        avg_x = []
        avg_y = []
        for i in range(len(unique_points)):
            mask = (indices == i)
            avg_x.append(np.mean(all_x[mask]))
            avg_y.append(np.mean(all_y[mask]))
        
        return np.array(avg_x), np.array(avg_y)


    def fit_line_ransac(self, x, y):
        model = RANSACRegressor().fit(x.reshape(-1, 1), y)
        inlier_mask = model.inlier_mask_
        if np.sum(inlier_mask) < 2:
            return None, None
        m = model.estimator_.coef_[0]
        c = model.estimator_.intercept_
        return m, c

    def LaserCb(self, msg):
        #   This is a line equation, with respect to the car at (0,0), with the x axis being the heading.
        #   Get vector theta for the line, and theta_0 as the y intersection.
        # * Find the distance from the line to the origin with ( theta_T dot [[0],[0]] + theta_0 ) / (norm theta)
        # TLDR, We have a vector theta for the line we have found, and a distance to that wall.

        angle_step = msg.angle_increment
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        ranges = np.array(msg.ranges)

        angles = np.linspace(angle_min, angle_max, len(ranges))

        x_thresh, y_thresh = self.GetLocalWallCoords(ranges, angles)

        self.scan_buffer.append((x_thresh, y_thresh))

        # If we have enough data, then filter it and find line of best fit.
        if len(self.scan_buffer) >= self.num_scans_to_avg:
            
            # Save a pictutes of detected points as png
            # save_point_cloud(self.point_buffer_x_, self.point_buffer_y_)

            avg_x, avg_y = self.average_scans(self.scan_buffer)
            # save_point_cloud(x_thresh, y_thresh)
            self.publish_filtered_scan(msg, avg_x, avg_y)
            self.point_buffer_x_ = avg_x
            self.point_buffer_y_ = avg_y


            # Find line of best fit.
            A = np.vstack([self.point_buffer_x_, np.ones(len(self.point_buffer_x_))]).T
            m, c = np.linalg.lstsq(A, self.point_buffer_y_, rcond=0.001)[0]

            # m, c = self.fit_line_ransac(avg_x, avg_y)

            # Find angle from heading to wall.

            # Vector of wall. Call wall direction vector theta.
            # th = np.array([[m],[1]])
            # th /= np.linalg.norm(th)
            # Scalar to define the (hyper) plane
            # th_0 = c

            # Distance to wall is (th.T dot x_0 + th_0)/(norm(th))
            # dist_to_wall = abs(c/np.linalg.norm(th))
            # Angle between heading and wall.
            # angle_to_wall = math.atan2(m, 1)
            
            th = np.array([m, -1])  # Correct coefficients for y = mx + c
            th_norm = np.linalg.norm(th)
            dist_to_wall = abs(c) / th_norm  # Correct distance formula
            angle_to_wall = np.arctan(m)  # Simplify angle calculation

            #Wanna plot line y = mx + c with markers in rviz2
            self.PublishWallLine(m, c)

            # Clear scan buffers.
            self.point_buffer_x_=np.array([])
            self.point_buffer_y_=np.array([])
            self.scan_buffer = []
            
            # Feeding the current angle ERROR(with target 0), and the distance ERROR to wall. Desired error to be 0.
            # steer, error = self.PID.GetControl(0.0 - angle_to_wall, self.DESIRED_DISTANCE - dist_to_wall, self.SIDE)
            steer, error = self.PID.GetControl(np.pi/2 - angle_to_wall, self.DESIRED_DISTANCE - dist_to_wall, self.SIDE)

            self.get_logger().info(f"m: {m:.2f}, c: {c:.2f}")
            self.get_logger().info(f"Angle: {np.rad2deg(angle_to_wall):.2f}, Distance: {dist_to_wall:.2f}")
            self.get_logger().info(f"Error: {error:.2f}, Steer: {steer:.2f}")

            # # Create and Publish control after controller.
            # drive_msg = Twist()

            # drive_msg.header.seq = self.seq_id
            # self.seq_id += 1

            # Populate the command itself.
            # drive_msg.linear.x = self.VELOCITY
            # drive_msg.angular.z = steer

            # drive_msg.drive.steering_angle = steer
            # drive_msg.drive.steering_angle_velocity = 0.1
            # drive_msg.drive.speed = self.VELOCITY
            # drive_msg.drive.acceleration = 1
            # drive_msg.drive.acceleration = 0.5

            # self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    wall_follower = WallFollower()
    wall_follower.get_logger().info("WallFollower node started!")
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()