import msplat

import msplat

def train_3dgs_metal(workspace_dir, output_ply_path):
    print(f"Loading dataset from {workspace_dir}...")
    
    # Load dataset
    dataset = msplat.load_dataset(workspace_dir, eval_mode=True)
    
    # Configure training: 7000 iterations is the msplat default
    # num_downscales=0 prevents it from starting at a lower resolution
    config = msplat.TrainingConfig(iterations=7000, num_downscales=0)
    trainer = msplat.GaussianTrainer(dataset, config)
    
    print("Starting Metal-accelerated optimization loop...")
    
    # Pass a lambda function as the callback to print the progress
    trainer.train(
        lambda s: print(f"step={s.iteration} splats={s.splat_count:,}"), 
        callback_every=100
    )

    # Save the final optimized point cloud
    trainer.export_ply(output_ply_path)
    print(f"Success! 3DGS scene saved to {output_ply_path}")

# Call the training function
train_3dgs_metal("3d_visualizer/test_workspace", "3d_visualizer/test_scene.ply")