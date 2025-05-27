    def listener_callback(self, msg):
        try:
            input_file, script_choice = msg.data.strip().split('|')
        except ValueError:
            self.get_logger().error("Invalid message format. Use: /path/to/file.ply|ball or |poisson")
            return

        if not os.path.exists(input_file):
            self.get_logger().error(f"File {input_file} does not exist.")
            return

        if script_choice == 'ball':
            mlx_name = 'ball.mlx'
        elif script_choice == 'poisson':
            mlx_name = 'poisson.mlx'
        else:
            self.get_logger().error(f"Unknown script type: {script_choice}. Use 'ball' or 'poisson'.")
            return

        try:
            with importlib.resources.path('mesh', mlx_name) as mlx_path:
                mlx_script = str(mlx_path)
                output_file = os.path.join(os.path.dirname(mlx_script), f'{script_choice}_output.ply')
        except FileNotFoundError:
            self.get_logger().error(f"MLX script {mlx_name} not found in package.")
            return

        self.get_logger().info(f"Loading point cloud: {input_file}")
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_file)

        self.get_logger().info(f"Applying MLX script: {mlx_script}")
        ms.load_filter_script(mlx_script)
        ms.apply_filter_script()
        ms.set_current_mesh(ms.number_meshes() - 1)
        ms.save_current_mesh(output_file)

        self.get_logger().info(f"Mesh reconstruction complete. Output saved to: {output_file}")

