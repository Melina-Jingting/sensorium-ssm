import os
import tarfile
import numpy as np
import argparse
from src import constants
from tqdm import tqdm

def convert_to_webdataset(root_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # File types to process
    file_types = ["behavior", "pupil_center", "responses", "videos"]

    # Loop through each mouse dataset folder
    for folder in os.listdir(root_dir):
        if folder == ".DS_Store":
            continue 
        folder_path = os.path.join(root_dir, folder)
        tiers_path = os.path.join(folder_path, "meta", "trials", "tiers.npy")
        tiers = np.load(tiers_path)

        # mouse_short_name is computed but not used here
        mouse_short_name = folder.split("-")[0]

        # Loop through trials
        for trial_id, tier in tqdm(enumerate(tiers)):
            # For new mice, process only "train" or "oracle" trials
            if folder in constants.new_mice and tier not in ("train", "oracle"):
                continue 
            elif tier == "none":
                continue

            # Create output subdirectory for the given tier
            os.makedirs(os.path.join(output_dir, tier), exist_ok=True)
            tar_filename = os.path.join(output_dir, tier, f"{folder}.tar")
            
            # Open the tar file in append mode
            with tarfile.open(tar_filename, "a") as tar:
                sample_length = 0
                for file_type in file_types:
                    file_path = os.path.join(folder_path, "data", file_type, f"{trial_id}.npy")
                    if os.path.exists(file_path):
                        sample_length = np.load(file_path).shape[-1]
                        tar.add(file_path, arcname=f"{trial_id}.{file_type}.npy")
                    elif file_type == "responses":
                        # If responses file doesn't exist, create a dummy file
                        np.save(file_path, np.zeros((1, sample_length)))
                        tar.add(file_path, arcname=f"{trial_id}.{file_type}.npy")
            
        print(f"✅ Finished processing {folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process the sensorium dataset into tar files."
    )
    parser.add_argument(
        "--root_dir", type=str, required=True,
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for the generated tar files"
    )
    args = parser.parse_args()
    
    convert_to_webdataset(args.root_dir, args.output_dir)
