import threading
import cv2
import numpy as np
import pyrealsense2 as rs
import json
import os
from os.path import exists, join
from os import makedirs, remove
import shutil
from math import tan, radians

class bcolors:
    OK = '\033[92m' #GREEN
    WARNING = '\033[93m' #YELLOW
    FAIL = '\033[91m' #RED
    RESET = '\033[0m' #RESET COLOR

def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics 
    with open(filename, 'w') as outfile:
        json.dump(
            {
            'width': intrinsics.width,
            'height': intrinsics.height,
            'intrinsic_matrix': [
                intrinsics.fx, 0, 0,
                0, intrinsics.fy, 0,
                intrinsics.ppx, intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)

def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder, exist_ok=True)  # Creates parent directories if needed
    else:
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == 'y':
            shutil.rmtree(path_folder)
            makedirs(path_folder)
        else:
            exit()

def guidelines(input_img, width, height):
    # center vertical line (green)
    cv2.line(input_img, (int(width/2), int(0)), (int(width/2), int(height)), (0, 255, 0), 1)
    # left vertical line (white)
    cv2.line(input_img, (int(width/2-height/4), int(0)), (int(width/2-height/4), int(height)), (255, 255, 255), 1)
    # right vertical line (white)
    cv2.line(input_img, (int(width/2+height/4), int(0)), (int(width/2+height/4), int(height)), (255, 255, 255), 1)
    # circle (white)
    cv2.circle(input_img, (int(width/2), int(height/2)), int(height/4), (255, 255, 255), 1)
    # center dot (red)
    cv2.circle(input_img, (int(width/2), int(height/2)), 1, (0, 0, 255), 5)
    # horizontal line (green)
    cv2.line(input_img, (int(0), int(height/2)), (int(width), int(height/2)), (0, 255, 0), 1)
    # outer rectangle (red)
    cv2.rectangle(input_img, (int(width/8), int(height/(8/7))), (int(width/(8/7)), int(height/8)), (0, 0, 255), 1)
    # inner rectangle (green)
    cv2.rectangle(input_img, (int(width/4), int(height/(4/3))), (int(width/(4/3)), int(height/4)), (0, 255, 0), 1)

# Function to display images using OpenCV in a separate thread
def show_image(image):
    cv2.imshow('Recorder Realsense', image)
    cv2.waitKey(1)  # Keeps the window responsive

def capture_images(pipeline, align, path_depth, path_color, width, height):
    frame_count = 0
    imwrite_state = False
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frames = aligned_frames.get_color_frame()

        # Validate frames
        if not aligned_depth_frame or not color_frames:
            continue

        # Convert to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frames.get_data())

        # Display the color image using OpenCV in a separate thread
        threading.Thread(target=show_image, args=(color_image,)).start()

        # Save color and depth image if needed
        if imwrite_state:
            cv2.imwrite("%s/%06d.png" % (path_depth, frame_count), depth_image)
            cv2.imwrite("%s/%06d.jpg" % (path_color, frame_count), color_image)
            print(f"Saved color + depth image {frame_count:06d}")
            frame_count += 1

        # Pause recording and resume based on spacebar press
        key = cv2.waitKey(1)
        if key == 32:  # Spacebar to toggle recording
            imwrite_state = not imwrite_state

        # Handle the escape key for quitting
        if key == 27:  # ESC to quit
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    # Create directories for saving frames
    path_color = join("dataset", "color")
    path_depth = join("dataset", "depth")
    make_clean_folder(path_color)
    make_clean_folder(path_depth)

    # Create RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Load custom preset (ensure JSON format is correct)
    try:
        config_path = "PresetCustom.json"
        with open(config_path, 'r') as f:
            jsonObj = json.load(f)
            json_string = str(jsonObj).replace("'", '\"')
            
            # Convert string values to integers
            width = int(jsonObj['viewer']['stream-width'])
            height = int(jsonObj['viewer']['stream-height'])
            fps = int(jsonObj['viewer']['stream-fps'])
            
            print("Resolution:", width, "x", height)
            print("FPS:", fps)

    except FileNotFoundError:
        print("Error: JSON file not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
    except KeyError as e:
        print(f"Error: Missing expected key {e} in JSON")
    except ValueError:
        print("Error: Could not convert value to integer")

    # Configure RealSense for streaming
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    # Start streaming
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Start the image capture and OpenCV display in separate thread
    capture_thread = threading.Thread(target=capture_images, args=(pipeline, align, path_depth, path_color, width, height))
    capture_thread.start()

    # Wait for the thread to finish (or you can handle it based on your exit conditions)
    capture_thread.join()

    # Stop the pipeline after capture
    pipeline.stop()
