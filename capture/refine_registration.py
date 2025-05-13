# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/refine_registration.py

import numpy as np
import open3d as o3d
import sys
sys.path.append("/home/ridan/Development/3D-Scanner/files")
sys.path.append(".")
from file import join, get_file_list, write_poses_to_log
from visualization import draw_registration_result_original_color
from optimize_posegraph import optimize_posegraph_for_refined_scene

def update_posegraph_for_scene(s, t, transformation, information, odometry,
                               pose_graph):
    if t == s + 1:  # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s,
                                                     t,
                                                     transformation,
                                                     information,
                                                     uncertain=False))
    else:  # loop closure case
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s,
                                                     t,
                                                     transformation,
                                                     information,
                                                     uncertain=True))
    return (odometry, pose_graph)


def multiscale_icp(source,
                   target,
                   voxel_size,
                   max_iter,
                   config,
                   init_transformation=np.identity(4)):
    current_transformation = init_transformation
    for i, scale in enumerate(range(len(max_iter))):  # multi-scale approach
        iter = max_iter[scale]
        distance_threshold = config["voxel_size"] * 1.4
        print("voxel_size {}".format(voxel_size[scale]))
        source_down = source.voxel_down_sample(voxel_size[scale])
        target_down = target.voxel_down_sample(voxel_size[scale])
        if config["icp_method"] == "point_to_point":
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down, distance_threshold,
                current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=iter))
        else:
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] *
                                                     2.0,
                                                     max_nn=30))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] *
                                                     2.0,
                                                     max_nn=30))
            if config["icp_method"] == "point_to_plane":
                result_icp = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, distance_threshold,
                    current_transformation,
                    o3d.pipelines.registration.
                    TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=iter))
            if config["icp_method"] == "color":
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, distance_threshold,
                    current_transformation,
                    o3d.pipelines.registration.
                    TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=iter))
        current_transformation = result_icp.transformation
        if i == len(max_iter) - 1:
            information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source_down, target_down, voxel_size[scale] * 1.4,
                result_icp.transformation)

    return (result_icp.transformation, information_matrix)

def multiscale_icp_small_objects(source, target, voxel_size_init, config):
    """
    Modified multiscale ICP specifically for small objects
    """
    # Make a copy of point clouds to avoid modifying originals
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    
    # Add points if clouds are too sparse (helps with registration)
    if len(source_copy.points) < 1000 or len(target_copy.points) < 1000:
        source_copy.estimate_normals()
        target_copy.estimate_normals()
        
        # Densify if needed
        if len(source_copy.points) < 1000:
            source_copy = source_copy.uniform_down_sample(1)  # Keep all points
        if len(target_copy.points) < 1000:
            target_copy = target_copy.uniform_down_sample(1)  # Keep all points
    
    # Use three scales for small objects
    voxel_sizes = [voxel_size_init * 4, voxel_size_init * 2, voxel_size_init]
    
    # Start with identity matrix or previous transformation
    current_transformation = np.identity(4)
    
    for i, voxel_size in enumerate(voxel_sizes):
        # Downsample at current scale
        source_down = source_copy.voxel_down_sample(voxel_size)
        target_down = target_copy.voxel_down_sample(voxel_size)
        
        # Update normals at each scale
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        # Enhanced ICP parameters for small objects
        max_iter = 50 if i < 2 else 100
        
        try:
            # First try point-to-plane ICP (more robust for small objects)
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down, voxel_size * 3, current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
            
            # Check if registration was successful
            if result_icp.fitness > 0.05:
                current_transformation = result_icp.transformation
            else:
                # Try point-to-point as backup
                result_icp = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, voxel_size * 3, current_transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
                
                if result_icp.fitness > 0.05:
                    current_transformation = result_icp.transformation
        except Exception as e:
            print(f"ICP failed at scale {i}: {e}")
            # Continue to next scale with current transformation
    
    # Only try colored ICP if both point clouds have colors
    if (source.has_colors() and target.has_colors()):
        try:
            # Try colored ICP only at final scale
            result_colored = o3d.pipelines.registration.registration_colored_icp(
                source, target, voxel_size_init * 0.5, current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                 relative_rmse=1e-6,
                                                                 max_iteration=50))
            
            if result_colored.fitness > 0.1:  # Only use if it improves the result
                current_transformation = result_colored.transformation
        except Exception as e:
            print(f"Colored ICP failed: {e}")
            # Keep the current transformation from regular ICP
    
    return current_transformation

def local_refinement(source, target, transformation_init, config):
    voxel_size = config["voxel_size"]
    (transformation, information) = \
            multiscale_icp(
            source, target,
           # [voxel_size, voxel_size/2.0, voxel_size/4.0], [50, 30, 14],
             [voxel_size, voxel_size/2.0, voxel_size/4.0, voxel_size/8.0], [50, 30, 14, 10],
            config, transformation_init)
    return (transformation, information)


def register_point_cloud_pair(ply_file_names, s, t, transformation_init,
                              config):
    print("reading %s ..." % ply_file_names[s])
    source = o3d.io.read_point_cloud(ply_file_names[s])
    print("reading %s ..." % ply_file_names[t])
    target = o3d.io.read_point_cloud(ply_file_names[t])

    if config["debug_mode"]:
        draw_registration_result_original_color(source, target,
                                                transformation_init)

    (transformation, information) = \
        local_refinement(source, target, transformation_init, config)

    if config["debug_mode"]:
        draw_registration_result_original_color(source, target, transformation)
        print(transformation)
        print(information)
    return (transformation, information)


# other types instead of class?
class matching_result:

    def __init__(self, s, t, trans):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = trans
        self.infomation = np.identity(6)


def make_posegraph_for_refined_scene(ply_file_names, config):
    pose_graph = o3d.io.read_pose_graph(
        join(config["path_dataset"],
             config["template_global_posegraph_optimized"]))

    n_files = len(ply_file_names)
    matching_results = {}
    for edge in pose_graph.edges:
        s = edge.source_node_id
        t = edge.target_node_id

        transformation_init = edge.transformation
        matching_results[s * n_files + t] = \
            matching_result(s, t, transformation_init)

    if config["python_multi_threading"] == True:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(),
                         max(len(pose_graph.edges), 1))
        results = Parallel(n_jobs=MAX_THREAD)(
            delayed(register_point_cloud_pair)(
                ply_file_names, matching_results[r].s, matching_results[r].t,
                matching_results[r].transformation, config)
            for r in matching_results)
        for i, r in enumerate(matching_results):
            matching_results[r].transformation = results[i][0]
            matching_results[r].information = results[i][1]
    else:
        for r in matching_results:
            (matching_results[r].transformation,
                    matching_results[r].information) = \
                    register_point_cloud_pair(ply_file_names,
                    matching_results[r].s, matching_results[r].t,
                    matching_results[r].transformation, config)

    pose_graph_new = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph_new.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(odometry))
    for r in matching_results:
        (odometry, pose_graph_new) = update_posegraph_for_scene(
            matching_results[r].s, matching_results[r].t,
            matching_results[r].transformation, matching_results[r].information,
            odometry, pose_graph_new)
    print(pose_graph_new)
    o3d.io.write_pose_graph(
        join(config["path_dataset"], config["template_refined_posegraph"]),
        pose_graph_new)


def run(config):
    print("refine rough registration of fragments.")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    ply_file_names = get_file_list(
        join(config["path_dataset"], config["folder_fragment"]), ".ply")
    make_posegraph_for_refined_scene(ply_file_names, config)
    optimize_posegraph_for_refined_scene(config["path_dataset"], config)

    path_dataset = config['path_dataset']
    n_fragments = len(ply_file_names)

    # Save to trajectory
    poses = []
    pose_graph_fragment = o3d.io.read_pose_graph(
        join(path_dataset, config["template_refined_posegraph_optimized"]))
    for fragment_id in range(len(pose_graph_fragment.nodes)):
        pose_graph_rgbd = o3d.io.read_pose_graph(
            join(path_dataset,
                 config["template_fragment_posegraph_optimized"] % fragment_id))
        for frame_id in range(len(pose_graph_rgbd.nodes)):
            frame_id_abs = fragment_id * \
                    config['n_frames_per_fragment'] + frame_id
            pose = np.dot(pose_graph_fragment.nodes[fragment_id].pose,
                          pose_graph_rgbd.nodes[frame_id].pose)
            poses.append(pose)

    traj_name = join(path_dataset, config["template_global_traj"])
    write_poses_to_log(traj_name, poses)
