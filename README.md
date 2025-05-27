import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pymeshlab
import os
import importlib.resources


class MeshingNode(Node):
    def __init__(self):
        super().__init__('meshing_node')
        self.subscription = self.create_subscription(
            String,
            'ply_file_input',  # Topic message format: "/path/to/file.ply|ball"
            self.listener_callback,
            10)
        self.get_logger().info("Meshing Node Initialized. Format: /path/to/file.ply|ball")

    def listener_callback(self, msg):
        try:
            # Expect input like: "/path/to/pointcloud.ply|ball"
            input_file, script_choice = msg.data.strip().split('|')
        except ValueError:
            self.get_logger().error("Invalid format. Use: '/path/to/file.ply|ball' or '|poisson'")
            return

        if not os.path.exists(input_file):
            self.get_logger().error(f"Point cloud file does not exist: {input_file}")
            return

        # Choose .mlx script based on user input
        script_map = {
            'ball': 'ball.mlx',
            'poisson': 'poisson.mlx',
        }

        if script_choice not in script_map:
            self.get_logger().error(f"Unsupported script '{script_choice}'. Options: {list(script_map.keys())}")
            return

        mlx_name = script_map[script_choice]

        try:
            with importlib.resources.path('mesh', mlx_name) as mlx_path:
                mlx_script = str(mlx_path)
                output_file = os.path.join(os.path.dirname(mlx_script), f'{script_choice}_output.ply')
        except FileNotFoundError:
            self.get_logger().error(f"MLX script not found: {mlx_name}")
            return

        # Apply Meshing
        self.get_logger().info(f"Loading point cloud: {input_file}")
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_file)

        self.get_logger().info(f"Applying script: {mlx_script}")
        ms.load_filter_script(mlx_script)
        ms.apply_filter_script()
        ms.set_current_mesh(ms.number_meshes() - 1)
        ms.save_current_mesh(output_file)

        self.get_logger().info(f"Mesh reconstruction complete. Output saved to: {output_file}")


def main(args=None):
    rclpy.init(args=args)
    node = MeshingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
