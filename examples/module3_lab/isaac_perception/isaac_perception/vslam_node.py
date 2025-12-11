#!/usr/bin/env python3

"""
VSLAM (Visual Simultaneous Localization and Mapping) node using Isaac ROS.
This node implements visual SLAM functionality for humanoid robots using Isaac Sim
and Isaac ROS components.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from tf2_ros import TransformBroadcaster

import numpy as np
import cv2
from cv_bridge import CvBridge
import message_filters
from tf2_ros import Buffer, TransformListener
import tf2_ros
import tf_transformations

# Isaac ROS imports (these would be available when running in Isaac Sim environment)
try:
    from isaac_ros_visual_slam_msgs.msg import TfArray
    from isaac_ros_visual_slam_msgs.msg import TrackResult
    print("Isaac ROS Visual SLAM messages imported successfully")
except ImportError:
    print("Isaac ROS Visual SLAM messages not available. Using mock implementations.")

    # Mock Isaac ROS message types for documentation purposes
    class TfArray:
        pass

    class TrackResult:
        pass


class IsaacVSLAMNode(Node):
    """
    VSLAM node for humanoid robot localization and mapping using Isaac ROS.
    """

    def __init__(self):
        super().__init__('isaac_vslam_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # TF broadcaster for publishing transforms
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS profile for sensor data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create subscribers for stereo camera data
        self.left_image_sub = message_filters.Subscriber(
            self, Image, '/camera/left/image_rect_color', qos_profile=qos_profile)
        self.right_image_sub = message_filters.Subscriber(
            self, Image, '/camera/right/image_rect_color', qos_profile=qos_profile)
        self.left_cam_info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/left/camera_info', qos_profile=qos_profile)
        self.right_cam_info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/right/camera_info', qos_profile=qos_profile)

        # Synchronize stereo image and camera info messages
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.left_image_sub, self.right_image_sub,
             self.left_cam_info_sub, self.right_cam_info_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.stereo_callback)

        # Alternative: subscribe to RGB-D data if available
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_rect_color', self.rgb_callback, qos_profile)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, qos_profile)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/visual_odometry', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_pose', 10)
        self.map_pub = self.create_publisher(MarkerArray, '/visual_map', 10)

        # Internal state
        self.prev_frame = None
        self.prev_pose = np.eye(4)  # 4x4 identity matrix
        self.current_pose = np.eye(4)
        self.map_points = []  # For storing map points
        self.frame_count = 0

        # Feature detector and matcher
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Camera parameters (will be updated from camera info)
        self.camera_matrix = None
        self.dist_coeffs = None

        # Processing flags
        self.processing_enabled = True
        self.first_frame = True

        self.get_logger().info('Isaac VSLAM node initialized')

    def stereo_callback(self, left_msg, right_msg, left_cam_info, right_cam_info):
        """
        Process synchronized stereo camera data.
        """
        if not self.processing_enabled:
            return

        try:
            # Convert ROS images to OpenCV
            left_cv = self.cv_bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            right_cv = self.cv_bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

            # Update camera parameters from camera info
            if self.camera_matrix is None:
                self.camera_matrix = np.array(left_cam_info.k).reshape(3, 3)
                self.dist_coeffs = np.array(left_cam_info.d)

            # Process stereo data for VSLAM
            pose_change = self.process_stereo_vslam(left_cv, right_cv)

            if pose_change is not None:
                self.update_pose(pose_change, left_msg.header.stamp)

        except Exception as e:
            self.get_logger().error(f'Error processing stereo data: {e}')

    def rgb_callback(self, rgb_msg):
        """
        Process RGB camera data as an alternative to stereo.
        """
        if not self.processing_enabled:
            return

        try:
            # Convert ROS image to OpenCV
            rgb_cv = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

            # Process RGB data for visual odometry
            pose_change = self.process_visual_odometry(rgb_cv)

            if pose_change is not None:
                self.update_pose(pose_change, rgb_msg.header.stamp)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB data: {e}')

    def depth_callback(self, depth_msg):
        """
        Process depth camera data for 3D reconstruction.
        """
        if not self.processing_enabled:
            return

        try:
            # Convert ROS depth image to OpenCV
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

            # Process depth data for mapping
            self.process_depth_mapping(depth_cv)

        except Exception as e:
            self.get_logger().error(f'Error processing depth data: {e}')

    def process_stereo_vslam(self, left_img, right_img):
        """
        Process stereo images for VSLAM.
        """
        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=15,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute disparity
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        # Convert disparity to depth
        if self.camera_matrix is not None:
            # Calculate depth from disparity (simplified)
            baseline = 0.1  # Baseline in meters (typical for stereo cameras)
            focal_length = self.camera_matrix[0, 0]
            depth_map = (baseline * focal_length) / (disparity + 1e-6)

            # Process depth map for mapping
            self.update_map_from_depth(depth_map, left_img)

        # For pose estimation, use feature matching between frames
        if self.prev_frame is not None:
            pose_change = self.estimate_motion_stereo(self.prev_frame, left_img)
            self.prev_frame = left_img.copy()
            return pose_change
        else:
            self.prev_frame = left_img.copy()
            return None

    def process_visual_odometry(self, current_img):
        """
        Process visual odometry using feature tracking.
        """
        if self.prev_frame is None:
            self.prev_frame = current_img.copy()
            return None

        # Detect features in current frame
        curr_kp, curr_desc = self.feature_detector.detectAndCompute(current_img, None)
        prev_kp, prev_desc = self.feature_detector.detectAndCompute(self.prev_frame, None)

        if curr_desc is None or prev_desc is None:
            return None

        # Match features
        matches = self.bf_matcher.knnMatch(prev_desc, curr_desc, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # Need minimum number of matches
        if len(good_matches) < 10:
            self.prev_frame = current_img.copy()
            return None

        # Extract matched points
        prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate motion using essential matrix (if camera matrix is available)
        if self.camera_matrix is not None:
            E, mask = cv2.findEssentialMat(
                curr_pts, prev_pts,
                cameraMatrix=self.camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )

            if E is not None:
                # Decompose essential matrix to get rotation and translation
                _, R, t, mask_pose = cv2.recoverPose(E, curr_pts, prev_pts, self.camera_matrix)

                # Create transformation matrix
                pose_change = np.eye(4)
                pose_change[:3, :3] = R
                pose_change[:3, 3] = t.flatten()

                # Update previous frame
                self.prev_frame = current_img.copy()
                return pose_change

        # Fallback: just update the frame
        self.prev_frame = current_img.copy()
        return None

    def estimate_motion_stereo(self, prev_img, curr_img):
        """
        Estimate motion between two stereo frames.
        """
        # Detect features in both frames
        prev_kp, prev_desc = self.feature_detector.detectAndCompute(prev_img, None)
        curr_kp, curr_desc = self.feature_detector.detectAndCompute(curr_img, None)

        if curr_desc is None or prev_desc is None:
            return None

        # Match features
        matches = self.bf_matcher.knnMatch(prev_desc, curr_desc, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # Need minimum number of matches
        if len(good_matches) < 10:
            return None

        # Extract matched points
        prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate motion using essential matrix (if camera matrix is available)
        if self.camera_matrix is not None:
            E, mask = cv2.findEssentialMat(
                curr_pts, prev_pts,
                cameraMatrix=self.camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )

            if E is not None:
                # Decompose essential matrix to get rotation and translation
                _, R, t, mask_pose = cv2.recoverPose(E, curr_pts, prev_pts, self.camera_matrix)

                # Create transformation matrix
                pose_change = np.eye(4)
                pose_change[:3, :3] = R
                pose_change[:3, 3] = t.flatten()

                return pose_change

        return None

    def process_depth_mapping(self, depth_img):
        """
        Process depth image for 3D mapping.
        """
        # Convert depth image to point cloud
        if self.camera_matrix is not None:
            height, width = depth_img.shape
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]

            # Generate 3D points from depth image
            points_3d = []
            for v in range(0, height, 10):  # Subsample for performance
                for u in range(0, width, 10):
                    z = depth_img[v, u]
                    if z > 0 and z < 10:  # Valid depth range
                        x = (u - cx) * z / fx
                        y = (v - cy) * z / fy
                        points_3d.append([x, y, z])

            # Add points to map if we have a valid transformation
            if len(points_3d) > 0:
                self.update_local_map(points_3d)

    def update_map_from_depth(self, depth_map, rgb_img):
        """
        Update the map using depth information.
        """
        # This is a simplified approach - in a real implementation,
        # you would use more sophisticated mapping techniques
        if self.camera_matrix is not None:
            height, width = depth_map.shape
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]

            # Sample points from the depth map
            for v in range(0, height, 20):  # Further subsample for performance
                for u in range(0, width, 20):
                    z = depth_map[v, u]
                    if z > 0 and z < 10:  # Valid depth range
                        # Convert to 3D point in camera frame
                        x_cam = (u - cx) * z / fx
                        y_cam = (v - cy) * z / fy
                        z_cam = z

                        # Transform to world frame using current pose
                        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
                        point_world = self.current_pose @ point_cam

                        # Add to map points
                        self.map_points.append(point_world[:3])

    def update_local_map(self, points_3d):
        """
        Update local map with 3D points.
        """
        # In a real implementation, this would use more sophisticated mapping
        # like occupancy grids or point cloud maps
        # For now, we'll just keep track of recent points
        if len(self.map_points) > 1000:  # Limit map size
            self.map_points = self.map_points[-500:]  # Keep last 500 points

        self.map_points.extend(points_3d)

    def update_pose(self, pose_change, stamp):
        """
        Update the current pose based on the estimated change.
        """
        # Apply pose change to current pose
        self.current_pose = self.current_pose @ pose_change

        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = self.current_pose[0, 3]
        odom_msg.pose.pose.position.y = self.current_pose[1, 3]
        odom_msg.pose.pose.position.z = self.current_pose[2, 3]

        # Set orientation from rotation matrix
        rotation_matrix = self.current_pose[:3, :3]
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]

        # Set velocity (approximated)
        dt = 0.1  # Assumed time delta
        linear_vel = np.linalg.norm(pose_change[:3, 3]) / dt
        odom_msg.twist.twist.linear.x = linear_vel

        # Publish odometry
        self.odom_pub.publish(odom_msg)

        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'map'
        pose_msg.pose = odom_msg.pose.pose
        self.pose_pub.publish(pose_msg)

        # Publish TF
        self.publish_transform(stamp)

        # Log position
        self.get_logger().info(
            f'Position: ({self.current_pose[0, 3]:.2f}, {self.current_pose[1, 3]:.2f}, {self.current_pose[2, 3]:.2f})'
        )

    def publish_transform(self, stamp):
        """
        Publish the transform from map to base_link.
        """
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'

        # Set translation
        t.transform.translation.x = self.current_pose[0, 3]
        t.transform.translation.y = self.current_pose[1, 3]
        t.transform.translation.z = self.current_pose[2, 3]

        # Set rotation
        rotation_matrix = self.current_pose[:3, :3]
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        # Send the transform
        self.tf_broadcaster.sendTransform(t)

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        """
        Convert a 3x3 rotation matrix to quaternion.
        """
        # Using tf_transformations library
        quat = tf_transformations.quaternion_from_matrix(
            np.vstack([np.hstack([rotation_matrix, np.zeros((3, 1))]),
                      [0, 0, 0, 1]])
        )
        return quat

    def reset_slam(self):
        """
        Reset the SLAM system.
        """
        self.prev_frame = None
        self.prev_pose = np.eye(4)
        self.current_pose = np.eye(4)
        self.map_points = []
        self.frame_count = 0
        self.first_frame = True
        self.get_logger().info('VSLAM system reset')

    def enable_processing(self, enable=True):
        """
        Enable or disable VSLAM processing.
        """
        self.processing_enabled = enable
        state = "enabled" if enable else "disabled"
        self.get_logger().info(f'VSLAM processing {state}')


def main(args=None):
    """
    Main function to run the Isaac VSLAM node.
    """
    rclpy.init(args=args)

    vslam_node = IsaacVSLAMNode()

    try:
        rclpy.spin(vslam_node)
    except KeyboardInterrupt:
        pass
    finally:
        vslam_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()