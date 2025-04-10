import pyrealsense2 as rs
import numpy as np
import cv2
import os
#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

# Create directories to save depth and RGB frames
depth_output_dir = "depth_frames"
rgb_output_dir = "rgb_frames"
os.makedirs(depth_output_dir, exist_ok=True)
os.makedirs(rgb_output_dir, exist_ok=True)

# Configure depth and RGB streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream

# Start streaming
pipeline.start(config)

frame_count = 0

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert depth frame to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert RGB frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Save the raw depth frame as a binary file
        raw_depth_filename = os.path.join(depth_output_dir, f"depth_frame_{frame_count:04d}.raw")
        depth_image.tofile(raw_depth_filename)

        # Save the RGB frame as an image file
        rgb_filename = os.path.join(rgb_output_dir, f"rgb_frame_{frame_count:04d}.png")
        cv2.imwrite(rgb_filename, color_image)

        # Optionally, save the depth frame as an image for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_image_filename = os.path.join(depth_output_dir, f"depth_frame_{frame_count:04d}.png")
        cv2.imwrite(depth_image_filename, depth_colormap)

        print(f"Saved frame {frame_count}")
        frame_count += 1

        # Break the loop after a certain number of frames (e.g., 100)
        if frame_count >= 100:
            break

finally:
    # Stop streaming
    pipeline.stop()