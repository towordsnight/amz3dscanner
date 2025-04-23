# make_fragments.py — final serial implementation
# Open3D: www.open3d.org
# MIT License

import math
import sys
import numpy as np
import open3d as o3d
import faulthandler
# Enable C-level crash dumps
faulthandler.enable()

# Project imports
sys.path.append("/home/ridan/Development/3D-Scanner/files")
from file import join, make_clean_folder, get_rgbd_file_lists
from opencv import initialize_opencv
sys.path.append(".")
from optimize_posegraph import optimize_posegraph_for_fragment

# Check OpenCV availability
with_opencv = initialize_opencv()
if with_opencv:
    from opencv_pose_estimation import pose_estimation


def read_rgbd_image(color_file, depth_file, convert_rgb_to_intensity, config):
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=config["depth_scale"],
        depth_trunc=config["max_depth"],
        convert_rgb_to_intensity=convert_rgb_to_intensity)

# create an odometryOption for frame-to-frame alignment
def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic,
                           with_opencv, config):
    src = read_rgbd_image(color_files[s], depth_files[s], True, config)
    tgt = read_rgbd_image(color_files[t], depth_files[t], True, config)
    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = config["max_depth_diff"]

    # Consecutive-frame odometry only
    odo_init = np.eye(4)
    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        src, tgt, intrinsic, odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
    return [success, trans, info]


def make_posegraph_for_fragment(path_dataset, sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pg = o3d.pipelines.registration.PoseGraph()
    trans_accum = np.eye(4)
    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_accum))

    for s in range(sid, eid):
        for t in range(s + 1, eid):
            if t == s + 1:
                success, trans, info = register_one_rgbd_pair(
                    s, t, color_files, depth_files,
                    intrinsic, with_opencv, config)
                trans_accum = trans @ trans_accum
                inv = np.linalg.inv(trans_accum)
                pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(inv))
                pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                    s - sid, t - sid, trans, info, uncertain=False))

            # Optional loop closure
            if s % config['n_keyframes_per_n_frame'] == 0 and t % config['n_keyframes_per_n_frame'] == 0:
                success, trans, info = register_one_rgbd_pair(
                    s, t, color_files, depth_files,
                    intrinsic, with_opencv, config)
                if success:
                    pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                        s - sid, t - sid, trans, info, uncertain=True))

    out = join(path_dataset, config["template_fragment_posegraph"] % fragment_id)
    o3d.io.write_pose_graph(out, pg)


def integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id,
                                      n_fragments, pg_path, intrinsic, config):
    pg = o3d.io.read_pose_graph(pg_path)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=config.get("sdf_trunc", 0.04),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i, node in enumerate(pg.nodes):
        idx = fragment_id * config['n_frames_per_fragment'] + i
        rgbd = read_rgbd_image(color_files[idx], depth_files[idx], False, config)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(node.pose))

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def make_pointcloud_for_fragment(path_dataset, color_files, depth_files,
                                 fragment_id, n_fragments, intrinsic, config):
    pg_file = join(path_dataset, config["template_fragment_posegraph_optimized"] % fragment_id)
    mesh = integrate_rgb_frames_for_fragment(
        color_files, depth_files, fragment_id,
        n_fragments, pg_file, intrinsic, config)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    out_pcl = join(path_dataset, config["template_fragment_pointcloud"] % fragment_id)
    o3d.io.write_point_cloud(out_pcl, pcd, write_ascii=False, compressed=True)


def process_single_fragment(fragment_id, color_files, depth_files, n_files,
                            n_fragments, config):
    if config.get("path_intrinsic"):
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    sid = fragment_id * config['n_frames_per_fragment']
    eid = min(sid + config['n_frames_per_fragment'], n_files)

    make_posegraph_for_fragment(
        config["path_dataset"], sid, eid,
        color_files, depth_files,
        fragment_id, n_fragments,
        intrinsic, with_opencv, config)

    optimize_posegraph_for_fragment(
        config["path_dataset"], fragment_id, config)

    make_pointcloud_for_fragment(
        config["path_dataset"], color_files, depth_files,
        fragment_id, n_fragments, intrinsic, config)


def run(config):
    print("making fragments from RGBD sequence.")
    make_clean_folder(join(config["path_dataset"], config["folder_fragment"]))

    color_files, depth_files = get_rgbd_file_lists(config["path_dataset"])
    n_files = len(color_files)
    # n_fragments = int(math.ceil(n_files / config['n_frames_per_fragment']))
    n_fragments = int(math.ceil(n_files / 330))


    print(f"[INFO] Processing {n_fragments} fragments serially.")
    for fid in range(n_fragments):
        print(f"[INFO] Fragment {fid+1}/{n_fragments}")
        process_single_fragment(fid, color_files, depth_files, n_files, n_fragments, config)
