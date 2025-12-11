#!/usr/bin/env python3

"""
Isaac-based Perception Pipeline Lab Exercise
===========================================

This lab demonstrates the integration of SLAM and object detection using Isaac Sim and Isaac ROS.
Students will implement a complete perception pipeline for humanoid robots that combines:
- Visual SLAM for mapping and localization
- Object detection for scene understanding
- Integration with navigation systems
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import MarkerArray
from tf2_ros import TransformBroadcaster

import numpy as np
import cv2
from cv_bridge import CvBridge
import message_filters
import tf2_ros
import tf_transformations

# Isaac ROS imports
try:
    from isaac_ros_visual_slam_msgs.msg import TfArray
    from isaac_ros_visual_slam_msgs.msg import TrackResult
    from isaac_ros_object_detection_msgs.msg import Detection2DArray
    print("Isaac ROS perception messages imported successfully")
except ImportError as e:
    print(f"Isaac ROS perception messages not available: {e}")
    # Mock imports for documentation purposes
    class TfArray:
        pass

    class TrackResult:
        pass

    class Detection2DArray:
        pass


class IsaacPerceptionPipeline(Node):
    """
    Complete perception pipeline integrating SLAM and object detection.
    """

    def __init__(self):
        super().__init__('isaac_perception_pipeline')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # TF broadcaster and buffer
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS profile for sensor data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to stereo camera data for SLAM
        self.left_image_sub = message_filters.Subscriber(
            self, Image, '/camera/left/image_rect_color', qos_profile=qos_profile)
        self.right_image_sub = message_filters.Subscriber(
            self, Image, '/camera/right/image_rect_color', qos_profile=qos_profile)
        self.left_cam_info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/left/camera_info', qos_profile=qos_profile)
        self.right_cam_info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/right/camera_info', qos_profile=qos_profile)

        # Synchronize stereo data
        self.stereo_sync = message_filters.ApproximateTimeSynchronizer(
            [self.left_image_sub, self.right_image_sub,
             self.left_cam_info_sub, self.right_cam_info_sub],
            queue_size=10,
            slop=0.1
        )
        self.stereo_sync.registerCallback(self.stereo_callback)

        # Subscribe to RGB camera for object detection
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_rect_color', self.rgb_callback, qos_profile)

        # Subscribe to object detection results
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/isaac_ros/object_detection', self.detection_callback, 10)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/visual_odom', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/semantic_map', 10)
        self.detection_viz_pub = self.create_publisher(MarkerArray, '/detection_markers', 10)
        self.combined_map_pub = self.create_publisher(OccupancyGrid, '/combined_map', 10)

        # Internal state
        self.current_pose = np.eye(4)
        self.map_points = []
        self.camera_matrix = None
        self.detections = []
        self.processing_enabled = True

        # Feature detection for SLAM
        self.feature_detector = cv2.ORB_create(nfeatures=2000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.prev_frame = None

        # Object detection parameters
        self.detection_threshold = 0.5
        self.tracked_objects = {}  # Track objects across frames

        self.get_logger().info('Isaac Perception Pipeline initialized')

    def stereo_callback(self, left_msg, right_msg, left_cam_info, right_cam_info):
        """
        Process synchronized stereo data for SLAM.
        """
        if not self.processing_enabled:
            return

        try:
            # Convert ROS images to OpenCV
            left_cv = self.cv_bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            right_cv = self.cv_bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

            # Update camera parameters
            if self.camera_matrix is None:
                self.camera_matrix = np.array(left_cam_info.k).reshape(3, 3)

            # Process VSLAM
            pose_change = self.process_vslam(left_cv, right_cv)

            if pose_change is not None:
                self.update_pose(pose_change, left_msg.header.stamp)
                self.publish_visualization(left_cv, left_msg.header.stamp)

        except Exception as e:
            self.get_logger().error(f'Error in stereo callback: {e}')

    def rgb_callback(self, rgb_msg):
        """
        Process RGB camera data for object detection integration.
        """
        if not self.processing_enabled:
            return

        try:
            # Convert ROS image to OpenCV
            rgb_cv = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

            # Process detections if available
            if self.detections:
                self.process_detection_integration(rgb_cv, rgb_msg.header.stamp)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB data: {e}')

    def detection_callback(self, detection_msg):
        """
        Process object detection results.
        """
        self.detections = detection_msg.detections
        self.get_logger().info(f'Received {len(self.detections)} detections')

    def process_vslam(self, left_img, right_img):
        """
        Process stereo images for VSLAM.
        """
        if self.prev_frame is None:
            self.prev_frame = left_img.copy()
            return None

        # Extract features from current frame
        curr_kp, curr_desc = self.feature_detector.detectAndCompute(left_img, None)
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
            self.prev_frame = left_img.copy()
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
                self.prev_frame = left_img.copy()
                return pose_change

        # Update previous frame
        self.prev_frame = left_img.copy()
        return None

    def process_detection_integration(self, rgb_img, stamp):
        """
        Integrate object detections with SLAM map.
        """
        if not self.detections or self.camera_matrix is None:
            return

        # Process each detection
        for detection in self.detections:
            if detection.results and detection.results[0].score > self.detection_threshold:
                # Get bounding box
                bbox = detection.bbox
                center_x = bbox.center.position.x
                center_y = bbox.center.position.y

                # Estimate depth using stereo (simplified)
                # In a real implementation, you would use depth information
                estimated_depth = 1.0  # Placeholder

                # Convert image coordinates to 3D world coordinates
                if self.current_pose is not None:
                    # Calculate 3D position relative to camera
                    fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
                    cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

                    # Convert to 3D point in camera frame
                    x_cam = (center_x - cx) * estimated_depth / fx
                    y_cam = (center_y - cy) * estimated_depth / fy
                    z_cam = estimated_depth

                    # Transform to world frame
                    point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
                    point_world = self.current_pose @ point_cam

                    # Add to semantic map
                    self.add_semantic_point(
                        point_world[:3],
                        detection.results[0].class_id,
                        detection.results[0].score
                    )

        # Publish updated semantic map
        self.publish_semantic_map(stamp)

    def add_semantic_point(self, point_3d, class_id, confidence):
        """
        Add a semantic point to the map.
        """
        # In a real implementation, this would add to a semantic occupancy grid
        # For now, we'll just store in a list
        semantic_point = {
            'position': point_3d,
            'class_id': class_id,
            'confidence': confidence,
            'timestamp': self.get_clock().now().nanoseconds
        }

        # Add to map points
        self.map_points.append(semantic_point)

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

        # Publish odometry
        self.odom_pub.publish(odom_msg)

        # Publish TF
        self.publish_transform(stamp)

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

    def publish_semantic_map(self, stamp):
        """
        Publish the semantic map with detected objects.
        """
        # In a real implementation, this would create a proper semantic occupancy grid
        # For now, we'll create a simple visualization
        pass

    def publish_visualization(self, image, stamp):
        """
        Publish visualization data for debugging.
        """
        # Draw detections on the image
        if self.detections:
            viz_img = image.copy()
            for detection in self.detections:
                if detection.results and detection.results[0].score > self.detection_threshold:
                    bbox = detection.bbox
                    x = int(bbox.center.position.x - bbox.size_x / 2)
                    y = int(bbox.center.position.y - bbox.size_y / 2)
                    w = int(bbox.size_x)
                    h = int(bbox.size_y)

                    cv2.rectangle(viz_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(viz_img, f"{detection.results[0].class_id}: {detection.results[0].score:.2f}",
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Publish visualization (in a real implementation, you'd publish to an image topic)
            cv2.imshow('Perception Pipeline', viz_img)
            cv2.waitKey(1)

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        """
        Convert a 3x3 rotation matrix to quaternion.
        """
        quat = tf_transformations.quaternion_from_matrix(
            np.vstack([np.hstack([rotation_matrix, np.zeros((3, 1))]),
                      [0, 0, 0, 1]])
        )
        return quat

    def enable_processing(self, enable=True):
        """
        Enable or disable perception processing.
        """
        self.processing_enabled = enable
        state = "enabled" if enable else "disabled"
        self.get_logger().info(f'Perception processing {state}')


def run_perception_lab():
    """
    Run the Isaac perception pipeline lab exercise.
    """
    print("Starting Isaac-based Perception Pipeline Lab...")
    print("This lab demonstrates the integration of SLAM and object detection using Isaac Sim and Isaac ROS.")
    print("Topics to explore:")
    print("1. How SLAM provides localization for object detection")
    print("2. How object detection enhances semantic mapping")
    print("3. Integration with navigation systems")
    print("4. Real-time perception pipeline optimization")

    rclpy.init()

    perception_pipeline = IsaacPerceptionPipeline()

    try:
        rclpy.spin(perception_pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        perception_pipeline.destroy_node()
        rclpy.shutdown()


def create_lab_report():
    """
    Create a template for the lab report.
    """
    report_template = """
Isaac-based Perception Pipeline Lab Report
==========================================

Student Name: ________________
Date: ________________

Objective:
----------
The objective of this lab is to implement and evaluate a complete perception pipeline
that integrates Visual SLAM and object detection for humanoid robots using Isaac Sim and Isaac ROS.

Pre-Lab Questions:
------------------
1. Explain the difference between geometric mapping and semantic mapping.
2. Why is it important to integrate SLAM and object detection?
3. What are the challenges in real-time perception pipeline implementation?

Procedure:
----------
1. Set up Isaac Sim environment with stereo cameras and RGB camera
2. Configure Isaac ROS VSLAM node
3. Configure object detection node
4. Implement perception pipeline integration
5. Test the pipeline with various objects and environments
6. Evaluate performance metrics

Results:
--------
1. Localization accuracy with and without object detection
2. Mapping completeness and semantic content
3. Real-time performance metrics
4. Failure cases and limitations

Analysis:
---------
1. How does object detection improve SLAM performance?
2. What are the computational requirements of the integrated pipeline?
3. How robust is the system to different lighting conditions?
4. What are the limitations of current approach?

Conclusion:
-----------
Summarize the key findings and insights from the lab exercise.
"""

    with open("perception_lab_report_template.txt", "w") as f:
        f.write(report_template)

    print("Lab report template created: perception_lab_report_template.txt")


def main():
    """
    Main function for the perception pipeline lab.
    """
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "run":
            run_perception_lab()
        elif sys.argv[1] == "create_report":
            create_lab_report()
        else:
            print("Usage: python lab_exercise.py [run|create_report]")
    else:
        print("Isaac-based Perception Pipeline Lab")
        print("===================================")
        print("Available commands:")
        print("  python lab_exercise.py run          - Run the perception pipeline")
        print("  python lab_exercise.py create_report - Create lab report template")
        print("")
        print("This lab demonstrates:")
        print("  - Integration of SLAM and object detection")
        print("  - Semantic mapping for humanoid robots")
        print("  - Real-time perception pipeline")


if __name__ == '__main__':
    main()