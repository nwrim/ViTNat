"""
Applies the ViTNat model to compute naturalness predictions for all images in a directory.

Usage:
    python vitnat_predict_directory.py --image_dir /path/to/the/directory --output_path /path/to/output.csv

Outputs:
- CSV file with ViTNat predictions
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

def list_all_files(dir):
    """
    List all files (excluding directories) in the specified folder.

    Parameters
    ----------
    dir : str
        Path to the directory to list files from.

    Returns
    -------
    list of str
        List of filenames (not full paths) corresponding to regular files in the directory.

    Notes
    -----
    Adapted from: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    """

    # Filter and return only regular files (exclude directories)
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

def main(image_dir, output_path):
    # load the pre-trained model
    processor = ViTImageProcessor.from_pretrained('nwrim/ViTNat')
    model = ViTForImageClassification.from_pretrained('nwrim/ViTNat', num_labels=1)
    model.eval()

    # list all files in the directory
    image_names = list_all_files(image_dir)

    predictions = []
    for image_name in tqdm(image_names):
        image_path = os.path.join(image_dir, image_name)
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                pred = outputs.logits.item()
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            pred = np.nan
        predictions.append(pred)
    # create a DataFrame with the results
    df = pd.DataFrame({
        'image_name': image_names,
        'vitnat': predictions,
    })
    df.sort_values(by='image_name', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # save the DataFrame to a CSV file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply ViTNat model to compute naturalness predictions for images.")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing images.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output CSV file.')
    
    args = parser.parse_args()
    print(args.image_dir, args.output_path)
    main(args.image_dir, args.output_path)
