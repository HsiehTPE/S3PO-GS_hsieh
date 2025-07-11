from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from .init_pose import _resize_pil_image, torch_images_to_dust3r_format

def find_scale(im1, im2, depth1, depth2, model):
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    
    ## 从图像中提取特征，并进行点匹配
    images = torch_images_to_dust3r_format([im1, im2], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()     # 用于点匹配的图像特征

    matches_im1, matches_im2 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)
    
    H1 = view1['img'].shape[2]      # 输入到特征提取网络中的尺寸
    W1 = view1['img'].shape[3]
    
    depth1_resize = cv2.resize(depth1, (W1,H1), interpolation=cv2.INTER_LINEAR)
    depth2_resize = cv2.resize(depth2, (W1,H1), interpolation=cv2.INTER_LINEAR)

    ## 最近邻插值
    #depth1_resize = cv2.resize(depth1, (W1, H1), interpolation=cv2.INTER_NEAREST)
    #depth2_resize = cv2.resize(depth2, (W1, H1), interpolation=cv2.INTER_NEAREST)

    
    #depth_values_current = [depth2_resize[v, u] for (u, v) in matches_im2]
    #depth_values_previous = [depth1_resize[v, u] for (u, v) in matches_im1]
    
    # 将 matches_im2 和 matches_im1 转换为 NumPy 数组并拆分为索引
    #matches_im2 = np.array(matches_im2)
    #matches_im1 = np.array(matches_im1)

    # 获取列和行索引
    u2, v2 = matches_im2[:, 0], matches_im2[:, 1]
    u1, v1 = matches_im1[:, 0], matches_im1[:, 1]

    # 使用 NumPy 的索引方式直接提取深度值
    depth_values_current = depth2_resize[v2, u2]
    depth_values_previous = depth1_resize[v1, u1]

    # 生成有效掩码，确保两个深度数组中的像素都是有效的
    valid_mask = (depth_values_current > 0) & ~np.isnan(depth_values_current) & (depth_values_previous > 0) & ~np.isnan(depth_values_previous)

    # 应用掩码，确保两个数组使用相同的有效像素
    depth_values_current = depth_values_current[valid_mask]
    depth_values_previous = depth_values_previous[valid_mask]

    #scale_factor = np.median(depth_values_previous/(depth_values_current + 1e-8))
    #scale_factor = np.mean(depth_values_previous/(depth_values_current + 1e-8))

    scale_factor = np.mean(depth_values_previous) / (np.mean(depth_values_current))
    
    return scale_factor

def process_depth(render_depth, mono_depth, last_depth, im1, im2, model, patch_size=10, mean_threshold=0.25, std_threshold=0.3, error_threshold=0.1, final_error_threshold=0.15, max_iter=4, epsilon=0.01, min_accurate_pixels_ratio=0.01):
    # Step 1: 确保 render_depth 是 (H, W) 格式
    if render_depth.ndim == 3:
        render_depth = render_depth[0]  # 假设第一个通道为深度

    H, W = render_depth.shape
    scale_factor = 1.0
    prev_scale_factor = 0.0  # 用于比较前后尺度因子的变化
    final_mask = np.zeros((H, W), dtype=bool)
    
    total_pixels = H * W
    min_accurate_pixels = int(min_accurate_pixels_ratio * total_pixels)
    
    num_accurate_pixels = 0

    for k in range(max_iter):
        # 检查尺度因子的变化是否小于阈值
        if (abs(scale_factor - prev_scale_factor) < epsilon) and (scale_factor != 1.0):
            break
        prev_scale_factor = scale_factor
        
        patch_num = 0

        # Step 2: 初步筛选
        accurate_pixels = np.zeros((H, W), dtype=bool)
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                render_patch = render_depth[i:i+patch_size, j:j+patch_size]
                mono_patch = mono_depth[i:i+patch_size, j:j+patch_size] * scale_factor

                # 检查patch大小，以防超出边界
                if render_patch.size == 0 or mono_patch.size == 0:
                    continue

                # 判断均值和标准差的条件
                mean_condition = abs(np.mean(render_patch) - np.mean(mono_patch)) < mean_threshold * np.mean(mono_patch)
                std_condition = abs(np.std(render_patch) - np.std(mono_patch)) < std_threshold * np.std(mono_patch)

                if mean_condition and std_condition:  
                    patch_num = patch_num + 1
                    # Step 3: 标准化
                    render_norm = (render_patch - np.mean(render_patch)) / (np.std(render_patch) + 1e-6)
                    mono_norm = (mono_patch - np.mean(mono_patch)) / (np.std(mono_patch) + 1e-6)

                    # Step 4: 准确像素筛选
                    patch_mask = np.abs(render_norm - mono_norm) < error_threshold
                    accurate_pixels[i:i+patch_size, j:j+patch_size] = patch_mask

        if (np.sum(accurate_pixels) < min_accurate_pixels) and (k==2):
            num_accurate_pixels = np.sum(accurate_pixels) 
            
            scale_factor = find_scale(im1, im2, last_depth, mono_depth, model)
            continue
            #if (k>=2) or (scale_factor==prev_scale_factor):
            #print("找到的准确像素太少，基于前一关键帧做尺度修正")
            #break
        #else:
        if (np.sum(accurate_pixels) < min_accurate_pixels) and (k==3):
            num_accurate_pixels = np.sum(accurate_pixels) 
            scale_factor = find_scale(im1, im2, last_depth, mono_depth, model)
            print("找到的准确像素太少，基于前一关键帧做尺度修正")
            break
            
        num_accurate_pixels = 0
        if np.any(accurate_pixels) and ((k<2) or (np.sum(accurate_pixels) >= min_accurate_pixels)):
            scale_factor = np.mean(render_depth[accurate_pixels]) / np.mean(mono_depth[accurate_pixels])
            #scale_factor = np.median(render_depth[accurate_pixels] / (mono_depth[accurate_pixels] + 1e-8))
            final_mask = accurate_pixels.copy()  # 记录最后的准确像素mask
            num_accurate_pixels = np.sum(final_mask)

    # Step 7: 填补错误像素，使用相对误差
    mono_depth_scaled = mono_depth * scale_factor
    relative_error = np.abs(render_depth - mono_depth_scaled) / (mono_depth_scaled + 1e-8)  # 避免除以0
    error_mask = relative_error > final_error_threshold

    # 如果渲染深度为0，也需要填补
    error_mask[render_depth == 0] = True

    # 填补错误像素
    final_depth = np.where(error_mask, mono_depth_scaled, render_depth)
    
    print("通过第一轮筛选的patch数量: ", patch_num)

    return final_depth, scale_factor, error_mask, num_accurate_pixels 


def process_depth_torch(render_depth, mono_depth, patch_size=10, mean_threshold=0.15, std_threshold=0.25, error_threshold=0.05, final_error_threshold=0.1, max_iter=3, epsilon=0.01, min_accurate_pixels_ratio=0.0002):
    # Step 1: 确保 render_depth 是 (H, W) 格式，使用第一个通道
    if render_depth.ndim == 3:
        render_depth = render_depth[0]  # 假设第一个通道为深度

    H, W = render_depth.shape
    mono_depth = torch.from_numpy(mono_depth).float().cuda()  # 将 mono_depth 转为 torch 并放到 CUDA

    scale_factor = 1.0
    prev_scale_factor = 0.0  # 用于比较前后尺度因子的变化
    final_mask = torch.zeros((H, W), dtype=torch.bool, device=render_depth.device)

    total_pixels = H * W
    min_accurate_pixels = int(min_accurate_pixels_ratio * total_pixels)

    for _ in range(max_iter):
        if abs(scale_factor - prev_scale_factor) < epsilon:
            break
        prev_scale_factor = scale_factor

        # Step 2: 分patch处理
        render_patches = F.unfold(render_depth.unsqueeze(0).unsqueeze(0), kernel_size=patch_size, stride=patch_size)
        mono_patches = F.unfold(mono_depth.unsqueeze(0).unsqueeze(0), kernel_size=patch_size, stride=patch_size) * scale_factor

        # Step 3: 计算每个patch的均值和标准差
        render_means = render_patches.mean(dim=1)
        render_stds = render_patches.std(dim=1)
        mono_means = mono_patches.mean(dim=1)
        mono_stds = mono_patches.std(dim=1)

        # Step 4: 判断均值和标准差条件
        mean_condition = (torch.abs(render_means - mono_means) < mean_threshold * mono_means).all(dim=0)
        std_condition = (torch.abs(render_stds - mono_stds) < std_threshold * mono_stds).all(dim=0)

        # 判断是否满足条件的patch索引
        valid_patch_mask = mean_condition & std_condition

        # 选取满足条件的patch对应的像素位置
        accurate_patches = render_patches[:, :, valid_patch_mask]
        mono_patches_accurate = mono_patches[:, :, valid_patch_mask]

        if valid_patch_mask.sum() < min_accurate_pixels:
            print("找到的正确像素过少，尺度因子估计不鲁棒，退出循环")
            break

        # 计算新的尺度因子
        scale_factor = accurate_patches.mean() / (mono_patches_accurate.mean() + 1e-6)


        # 将patch还原到图像
        #accurate_pixels = F.fold(
        #    torch.ones_like(accurate_patches, dtype=torch.bool, device=render_depth.device),
        #    output_size=(H, W),
        #    kernel_size=patch_size,
        #    stride=patch_size
        #).bool()
        #final_mask.view(-1)[valid_patch_mask] = True
        #final_mask = accurate_pixels.clone()

    # Step 7: 填补错误像素，使用相对误差
    mono_depth_scaled = mono_depth * scale_factor
    relative_error = torch.abs(render_depth - mono_depth_scaled) / (mono_depth_scaled + 1e-6)  # 避免除以0
    error_mask = relative_error > final_error_threshold

    # 如果渲染深度为0，也需要填补
    error_mask[render_depth == 0] = True

    # 填补错误像素
    final_depth = torch.where(error_mask, mono_depth_scaled, render_depth)

    return final_depth, scale_factor, error_mask, valid_patch_mask.sum().item()