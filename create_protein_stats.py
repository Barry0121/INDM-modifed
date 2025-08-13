#!/usr/bin/env python3
"""
Create protein contact map statistics file using the exact same data loading process as evaluation.
"""
import numpy as np
import sys
import os
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# Add current directory to path to import modules
sys.path.append('/mnt/data/gpu-server/learning/INDM')
from pdb_dataset import PDB
from cleanfid.features import build_feature_extractor
from cleanfid.resize import build_resizer

def get_batch_features(batch, model, device):
    """Extract features from a batch using Inception model."""
    with torch.no_grad():
        feat = model(batch.to(device).permute(0,3,1,2))
    return feat.detach().cpu().numpy()

def create_protein_stats(output_path, mode="clean", device=torch.device("cuda")):
    """Create statistics file using the exact same Inception feature extraction as evaluation."""
    
    print("Creating protein contact map statistics...")
    print("Using the same Inception model feature extraction as evaluation")
    
    try:
        # Step 1: Initialize Inception model (same as used in evaluation)
        print("Initializing Inception model...")
        model = build_feature_extractor(mode, device)
        fn_resize = build_resizer(mode)
        transforms_fn = transforms.ToTensor()
        
        # Step 2: Load data exactly as evaluation does
        print("Loading evaluation dataset...")
        eval_dataset = PDB(train=False)  # Use evaluation split
        
        print("Loading training dataset...")
        train_dataset = PDB(train=True)   # Also load training split for complete statistics
        
        # Step 3: Get the normalized matrices (exactly as used in evaluation)
        eval_matrices = eval_dataset.get_matrices()  # Already normalized to [-1, 1]
        train_matrices = train_dataset.get_matrices()
        
        print(f"Training matrices shape: {train_matrices.shape}")
        print(f"Evaluation matrices shape: {eval_matrices.shape}")
        
        # Step 4: Combine both splits for comprehensive statistics
        all_matrices = np.concatenate([train_matrices, eval_matrices], axis=0)
        print(f"Combined matrices shape: {all_matrices.shape}")
        
        # Step 5: Process through Inception model (same as evaluation)
        print("Processing matrices through Inception model...")
        all_features = []
        batch_size = 128  # Process in batches to avoid memory issues
        
        for i in tqdm(range(0, len(all_matrices), batch_size), desc="Processing batches"):
            batch_matrices = all_matrices[i:i+batch_size]
            
            # Convert single-channel protein maps to 3-channel (same as evaluation)
            if len(batch_matrices.shape) == 3:  # (batch, height, width)
                batch_matrices = np.stack([batch_matrices, batch_matrices, batch_matrices], axis=-1)
            elif len(batch_matrices.shape) == 4 and batch_matrices.shape[-1] == 1:  # (batch, height, width, 1)
                batch_matrices = np.repeat(batch_matrices, 3, axis=-1)
            
            # Process each image in the batch
            batch_tensor = torch.zeros((batch_matrices.shape[0], 299, 299, 3), device=device)
            for j, img in enumerate(batch_matrices):
                # Normalize to [0, 255] range (same as evaluation)
                if np.max(img) <= 1.0:  # If in [-1, 1] range
                    img = (img + 1.0) / 2.0 * 255.0  # Convert to [0, 255]
                
                # Resize to 299x299 for Inception
                img_resized = fn_resize(img.astype(np.uint8))
                img_t = transforms_fn(img_resized) * 255
                img_t = img_t.permute(1, 2, 0)
                batch_tensor[j] = img_t
            
            # Extract features using Inception model
            features = get_batch_features(batch_tensor, model, device)
            all_features.append(features)
        
        # Step 6: Combine all features
        all_features = np.concatenate(all_features, axis=0)
        print(f"Extracted features shape: {all_features.shape}")
        
        # Step 7: Compute mu and sigma from Inception features
        print("Computing mean (mu) from Inception features...")
        mu = np.mean(all_features, axis=0).astype(np.float32)
        
        print("Computing covariance matrix (sigma) from Inception features...")
        sigma = np.cov(all_features, rowvar=False).astype(np.float32)
        
        print(f"Mu shape: {mu.shape}")
        print(f"Sigma shape: {sigma.shape}")
        print(f"Mu range: [{np.min(mu):.6f}, {np.max(mu):.6f}]")
        print(f"Sigma range: [{np.min(sigma):.6f}, {np.max(sigma):.6f}]")
        
        # Step 6: Save to file
        print(f"Saving statistics to: {output_path}")
        np.savez(output_path, mu=mu, sigma=sigma)
        
        print("Successfully created protein contact map statistics!")
        print(f"File saved at: {output_path}")
        
        # Verify the file was created correctly
        test_load = np.load(output_path)
        print("Verification:")
        print(f"  Loaded mu shape: {test_load['mu'].shape}")
        print(f"  Loaded sigma shape: {test_load['sigma'].shape}")
        
    except Exception as e:
        print(f"Error creating statistics: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    # Create statistics in the location we have access to
    output_path = "/mnt/data/gpu-server/learning/assets/stats/protein_contact_map_stats.npz"
    
    # Make sure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    success = create_protein_stats(output_path, mode="clean", device=device)
    
    if success:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("Run the evaluation with custom assetdir:")
        print("python main.py --config=configs/vp/PROTEIN/indm_fid.py \\")
        print("  --workdir=outputs/protein_vp_fid_run \\")
        print("  --mode=eval \\")
        print("  --eval_folder=test_eval \\")
        print("  --assetdir=/mnt/data/gpu-server/learning/assets/stats/")
        print("="*60)
    else:
        print("Failed to create statistics file!")
        sys.exit(1)