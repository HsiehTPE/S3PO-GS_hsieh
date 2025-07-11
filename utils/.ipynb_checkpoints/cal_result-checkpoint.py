import argparse
from argparse import ArgumentParser
import sys
import os
import pandas as pd
from datetime import datetime
import json


def get_latest_folder(directory):
    # 获取目录下的所有文件夹
    dataset_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    if not dataset_folders:
        return None  # 如果没有符合条件的文件夹，返回None
    
    # 获取每个文件夹的最后修改时间
    latest_folder = max(dataset_folders, key=lambda folder: os.path.getmtime(os.path.join(directory, folder)))
    
    return latest_folder

def extract_results(directory, dataset_name):
    # 存储每个序列的结果
    results = {}

    # 查找所有与数据集名称匹配的文件夹
    dataset_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f)) and dataset_name in f]
    
    for folder in dataset_folders:
        folder_path = os.path.join(directory, folder)
        
        # 获取每个序列下最新的文件夹
        latest_folder = get_latest_folder(folder_path)
        
        if latest_folder is None:
            print(f"没有找到符合条件的文件夹: {folder}")
            continue
        
        # 更新路径：进入“results/waymo_序列号/最新的结果文件夹”
        latest_folder_path = os.path.join(folder_path, latest_folder)
        
        # 读取 stats_final.json 和 final_result.json
        stats_file = os.path.join(latest_folder_path, 'plot', 'stats_final.json')
        final_result_file = os.path.join(latest_folder_path, 'psnr', 'after_opt', 'final_result.json')
        
        if not os.path.exists(stats_file) or not os.path.exists(final_result_file):
            print(f"在 {latest_folder_path} 中没有找到需要的文件")
            continue
        
        with open(stats_file, 'r') as f:
            stats_data = json.load(f)
        
        with open(final_result_file, 'r') as f:
            final_result_data = json.load(f)
        
        # 提取需要的字段
        result = {
            'rmse': stats_data.get('rmse'),
            'mean_psnr': final_result_data.get('mean_psnr'),
            'mean_ssim': final_result_data.get('mean_ssim'),
            'mean_lpips': final_result_data.get('mean_lpips')
        }
        
        # 提取序列号
        sequence_number = folder.split('_')[1]
        
        # 将数据存储到结果字典中
        results[sequence_number] = result
    
    return results


def save_results_to_csv(results, dataset_name):
    # 创建一个DataFrame
    df = pd.DataFrame(results)
    
    # 计算每行的平均值
    df['Avg'] = df.mean(axis=1)
    
    # 生成文件名
    date_str = datetime.now().strftime('%Y%m%d')
    filename = f"{dataset_name}_{date_str}.csv"
    
    # 保存到CSV
    df.to_csv(filename)
    print(f"结果已保存为 {filename}")

# 主函数
def main():
    # 解析命令行参数
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--dataname", type=str, default='waymo')

    args = parser.parse_args(sys.argv[1:])
    dataset_name = args.dataname  # 获取数据集名称
    
    directory = './results'  # 设置您实际的文件夹路径
    results = extract_results(directory, dataset_name)
    
    if results:
        save_results_to_csv(results, dataset_name)

if __name__ == "__main__":
    main()
