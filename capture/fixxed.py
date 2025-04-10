import pyrealsense2 as rs
import numpy as np
import cv2
from os import listdir, makedirs, remove
from os.path import exists, join
from math import tan, radians
import shutil
import json

class bcolors:
    OK = '\033[92m' #GREEN
    WARNING = '\033[93m' #YELLOW
    FAIL = '\033[91m' #RED
    RESET = '\033[0m' #RESET COLOR

def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics 
    with open(filename, 'w') as outfile:
        obj = json.dump(
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

def guidelines(input_img):
    # center vertical line (green)
    cv2.line(input_img, (int(width/2), int(0)), (int(width/2), int(height)), (0, 255, 0), 1)
    # left vertical line (white)
    cv2.line(input_img, (int(width/2-height/4), int(0)), (int(width/2-height/4), int(height)), (255, 255, 255), 1)
    # right vertical line (white)
    cv2.line(input_img, (int(width/2+height/4), int(0)), (int(width/2+height/4), int(height)), (255, 255, 255), 1)
    # circle (white)
    cv2.circle(input_img, (int(width/2), int(height/2)), int(height/4), (255, 255, 255), 1)
    # cemter dot (red)
    cv2.circle(input_img, (int(width/2), int(height/2)), 1, (0, 0, 255), 5)
    # hotizontal line (green)
    cv2.line(input_img, (int(0), int(height/2)), (int(width), int(height/2)), (0, 255, 0), 1)
    # outer rectangle (red)
    cv2.rectangle(input_img, (int(width/8), int(height/(8/7))), (int(width/(8/7)), int(height/8)), (0, 0, 255), 1)
    # inner rectangle (green)
    cv2.rectangle(input_img, (int(width/4), int(height/(4/3))), (int(width/(4/3)), int(height/4)), (0, 255, 0), 1)

if __name__ == "__main__":
    
    #create folder
    path_color = join("dataset", "color")
    path_depth = join("dataset", "depth")
    make_clean_folder(path_depth)
    make_clean_folder(path_color)

    # Create pipeline and configuration
    pipeline = rs.pipeline()
    config = rs.config()

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

    # Recorded frame parameters
    frame_start = 30  # skip first 30 frames
    frame_loop_at = 506
    frame_per_rotation = frame_loop_at - frame_start
    rotation_period = frame_per_rotation / fps
    rpm = 60 / rotation_period
    rotation_count = int(input("Rotation count (int): "))
    frame_max = frame_start + (frame_per_rotation * rotation_count)
    print("RPM: ", rpm)
    print("Streaming will end at frame number: " + str(frame_max-1) + " (%d frames" % (frame_max-frame_start) + ")")

    # Configure streams
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    # Start pipeline
    profile = pipeline.start(config)
    depth_sensor = profile.get_device()
    depth_first_sensor = profile.get_device().first_depth_sensor()

    # Use advanced mode
    advnc_mode = rs.rs400_advanced_mode(depth_sensor)
    advnc_mode.load_json(json_string)

    # Set optimal disparity
    depth_table_control_group = advnc_mode.get_depth_table()
    distance = float(input("Center distance from camera (cm): "))
    radius = float(input("Radius from center (cm): "))
    max_range = distance + radius
    print("max_range:", max_range)
    x_res = width
    h_fov = 87
    focal_length = 0.5 * (x_res / (tan(radians(h_fov/2))))
    baseline = 50
    disparity_shift = int(((focal_length * baseline) / max_range) / 10)
    print("disparity_shift:", disparity_shift)
    depth_table_control_group.disparityShift = disparity_shift
    advnc_mode.set_depth_table(depth_table_control_group)

    depth_scale = depth_first_sensor.get_depth_scale()
    clip_background_distance = 3  # in meters
    clipping_distance = clip_background_distance / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Initialize recording variables
    frame_count = 0
    rotation_init = 1
    imwrite_state = False

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
                if not frames:
                    print("Error: No frames received.")
                    continue
            except Exception as e:
                print(f"Error: {e}")
                break

            # Align depth and color frames
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frames = aligned_frames.get_color_frame()

            # Validate frames
            if not aligned_depth_frame or not color_frames:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frames.get_data())

            if frame_count == 0:
                save_intrinsic_as_json(join("/home/amazonrob/amz3dscan/capture", "camera_intrinsic.json"), color_frames)

            if imwrite_state:
                cv2.imwrite("%s/%06d.png" % (path_depth, frame_count), depth_image)
                cv2.imwrite("%s/%06d.jpg" % (path_color, frame_count), color_image)
                print("Saved color + depth image %06d" % frame_count)
                frame_count += 1

            grey_color = 153
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
            guidelines(bg_removed)
            guidelines(depth_colormap)
            images = np.hstack((bg_removed, depth_colormap))

            cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Recorder Realsense', images)
            key = cv2.waitKey(1)

            # Toggle recording state with SPACEBAR
            if key == 32:
                imwrite_state = not imwrite_state

            # Check for pause between rotations or exit on ESC (key == 27)
            if (frame_count > frame_start) and (frame_count != frame_max) and ((frame_count - frame_start) % frame_per_rotation == 0) and (frame_count > frame_per_rotation):
                imwrite_state = False  # Pause recording between rotations
                if key == 32:
                    print(f"{bcolors.OK}^ rotation{bcolors.WARNING} {rotation_init} {bcolors.RESET}")
                    imwrite_state = True
                    rotation_init += 1
                elif key == 27:
                    cv2.destroyAllWindows()
                    break
            elif (frame_count == frame_max) or (key == 27):
                print(f"{bcolors.OK}^ rotation{bcolors.WARNING} {rotation_init} {bcolors.RESET}")
                print(f"{frame_count - frame_start} frames recorded")
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()
        if exists(path_color):
            file_color_sorted = sorted(listdir(path_color))
            for file_color in file_color_sorted[:frame_start]:
                remove(join(path_color, file_color))
        else:
            print(f"{bcolors.FAIL}ERROR: File doesn't exist{bcolors.RESET}")
        if exists(path_depth):
            file_depth_sorted = sorted(listdir(path_depth))
            for file_depth in file_depth_sorted[:frame_start]:
                remove(join(path_depth, file_depth))
        else:
            print(f"{bcolors.FAIL}ERROR: File doesn't exist{bcolors.RESET}")
