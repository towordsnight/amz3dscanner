import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os

class CaptureModule:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.sync_flag = False

    def start_stream(self):
        # Configure depth and color streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)

        # Initialize post-processing filters
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()

    def stop_stream(self):
        self.pipeline.stop()

    def capture_frames(self, sync_flag):
        if sync_flag:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None, None

            # Apply post-processing filters
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)

            return depth_frame, color_frame
        else:
            return None, None

    def get_camera_parameters(self):
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_units = depth_sensor.get_option(rs.option.depth_units)
        intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        return {
            "depth_units": depth_units,
            "intrinsics": intrinsics
        }

class FilteringModule:
    def __init__(self, camera_parameters):
        self.depth_units = camera_parameters["depth_units"]
        self.intrinsics = camera_parameters["intrinsics"]

    def remove_background(self, depth_frame, color_frame, min_depth, max_depth):
        depth_data = np.asanyarray(depth_frame.get_data())
        depth_data = depth_data * self.depth_units  # Convert to meters

        # Depth thresholding
        depth_mask = np.logical_and(depth_data >= min_depth, depth_data <= max_depth)

        # Color segmentation (example: based on color range)
        color_data = np.asanyarray(color_frame.get_data())
        hsv_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_data, (0, 0, 0), (180, 255, 50))  # Example: remove black background

        # Combine depth and color masks
        combined_mask = np.logical_and(depth_mask, color_mask)
        return depth_data * combined_mask

    def depth_to_point_cloud(self, depth_data):
        points = []
        for v in range(depth_data.shape[0]):
            for u in range(depth_data.shape[1]):
                Z = depth_data[v, u]
                if Z == 0:  # Ignore invalid points
                    continue
                X = (u - self.intrinsics.ppx) * Z / self.intrinsics.fx
                Y = (v - self.intrinsics.ppy) * Z / self.intrinsics.fy
                points.append([X, Y, Z])
        return np.array(points)

    def clean_point_cloud(self, point_cloud, nb_neighbors=20, std_ratio=2.0):
        # Use statistical outlier removal filter
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return cl

    def process(self, depth_frame, color_frame, min_depth, max_depth):
        depth_data = self.remove_background(depth_frame, color_frame, min_depth, max_depth)
        point_cloud = self.depth_to_point_cloud(depth_data)
        cleared_point_cloud = self.clean_point_cloud(point_cloud)
        return cleared_point_cloud

if __name__ == "__main__":
    # Initialize CaptureModule
    capture_module = CaptureModule()
    capture_module.start_stream()

    # Create output directories
    depth_output_dir = "depth_frames"
    rgb_output_dir = "rgb_frames"
    os.makedirs(depth_output_dir, exist_ok=True)
    os.makedirs(rgb_output_dir, exist_ok=True)

    # Get camera parameters
    camera_parameters = capture_module.get_camera_parameters()

    # Initialize FilteringModule
    filtering_module = FilteringModule(camera_parameters)

    frame_count = 0
    all_point_clouds = []  # List to store point clouds from all frames

    try:
        while True:
            # Capture frames (set sync_flag to True)
            depth_frame, color_frame = capture_module.capture_frames(sync_flag=True)

            if depth_frame and color_frame:
                # Process frames to generate point cloud
                min_depth = 0.5  # Example: 0.5 meters
                max_depth = 2.0  # Example: 2.0 meters
                cleared_point_cloud = filtering_module.process(depth_frame, color_frame, min_depth, max_depth)
                print(f"Processed frame {frame_count}")

                # Add the point cloud to the list
                all_point_clouds.append(cleared_point_cloud)

                frame_count += 1

            # Break the loop after a certain number of frames (e.g., 100)
            if frame_count >= 100:
                break

    finally:
        # Stop the stream
        capture_module.stop_stream()

        # Merge all point clouds into one
        merged_point_cloud = o3d.geometry.PointCloud()
        for pcd in all_point_clouds:
            merged_point_cloud += pcd

        # Remove duplicates and downsample for higher quality
        merged_point_cloud = merged_point_cloud.voxel_down_sample(voxel_size=0.01)  # Adjust voxel size as needed

        # Save the final merged point cloud
        o3d.io.write_point_cloud("merged_point_cloud.ply", merged_point_cloud)
        print("Merged point cloud saved to merged_point_cloud.ply")

        # Visualize the final point cloud
        o3d.visualization.draw_geometries([merged_point_cloud])