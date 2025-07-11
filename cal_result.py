import argparse
from argparse import ArgumentParser
import sys
import os
import pandas as pd
from datetime import datetime
import json


def get_latest_folder(directory):
    # Get all subfolders in the directory
    dataset_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    if not dataset_folders:
        return None  
    
    # Get the last modified time of each folder
    latest_folder = max(dataset_folders, key=lambda folder: os.path.getmtime(os.path.join(directory, folder)))
    
    return latest_folder

def extract_results(directory, dataset_name):
    # Store results for each sequence
    results = {}

    dataset_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f)) and dataset_name in f]
    
    for folder in dataset_folders:
        folder_path = os.path.join(directory, folder)
        
        latest_folder = get_latest_folder(folder_path)
        
        if latest_folder is None:
            print(f"No folder matching the criteria was found: {folder}")
            continue
        # Read evaluation metrics
        latest_folder_path = os.path.join(folder_path, latest_folder)
        stats_file = os.path.join(latest_folder_path, 'plot', 'stats_final.json')
        final_result_file = os.path.join(latest_folder_path, 'psnr', 'after_opt', 'final_result.json')
        
        if not os.path.exists(stats_file) or not os.path.exists(final_result_file):
            print(f"Required files not found in {latest_folder_path}")
            continue
        
        with open(stats_file, 'r') as f:
            stats_data = json.load(f)
        
        with open(final_result_file, 'r') as f:
            final_result_data = json.load(f)
        
        result = {
            'rmse': stats_data.get('rmse'),
            'mean_psnr': final_result_data.get('mean_psnr'),
            'mean_ssim': final_result_data.get('mean_ssim'),
            'mean_lpips': final_result_data.get('mean_lpips')
        }
        sequence_number = folder.split('_')[1]
        results[sequence_number] = result
    
    return results

# Save mean values
def save_results_to_csv(results, dataset_name):
    
    df = pd.DataFrame(results)
    df['Avg'] = df.mean(axis=1)
    
    date_str = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f"{dataset_name}_{date_str}.csv"
    
    df.to_csv(filename)
    print(f"Results saved as {filename}")


def main():
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--dataname", type=str, default='waymo')

    args = parser.parse_args(sys.argv[1:])
    dataset_name = args.dataname  
    
    directory = './results'  
    results = extract_results(directory, dataset_name)
    
    if results:
        save_results_to_csv(results, dataset_name)

if __name__ == "__main__":
    main()
