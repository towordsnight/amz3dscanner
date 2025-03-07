## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#####################################################
## librealsense tutorial #1 - Accessing depth data ##
#####################################################

# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2

try:
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()

    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    while True:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        # UPDATE
        # get color frames
        color_frames = frames.get_color_frame()
        if not depth or not color_frames: continue
        
        # UPDATE
        # convert the depth data to visual image
        # 1. convert to NumPy arrays
        depth_data = np.asanyarray(depth.get_data())
        color_data = np.asanyarray(color_frames.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.03), cv2.COLORMAP_JET)
        # If depth and color resolutions are different, resize color image to match depth image for display
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_data.shape
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_data, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_data, depth_colormap))


        # Show the side-by-side depth and color images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and approximating the coverage of pixels within one meter
        # coverage = [0]*64
        # for y in range(480):
        #     for x in range(640):
        #         dist = depth.get_distance(x, y)
        #         if 0 < dist and dist < 1:
        #             coverage[x//10] += 1
            
        #     if y%20 == 19:
        #         line = ""
        #         for c in coverage:
        #             line += " .:nhBXWW"[c//25]
        #         coverage = [0]*64
        #         print(line)
    # exit(0)
#except rs.error as e:
#    # Method calls agaisnt librealsense objects may throw exceptions of type pylibrs.error
#    print("pylibrs.error was thrown when calling %s(%s):\n", % (e.get_failed_function(), e.get_failed_args()))
#    print("    %s\n", e.what())
#    exit(1)
# except Exception as e:
#     print(e)
#     pass

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()