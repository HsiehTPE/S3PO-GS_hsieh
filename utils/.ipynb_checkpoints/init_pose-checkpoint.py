from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import os
import numpy as np
import time
import cv2
import torch
from scipy.spatial.transform import Rotation as R
import PIL.Image
from PIL.ImageOps import exif_transpose
import matplotlib
matplotlib.use('Agg')  # 使用无图形界面的后端
import matplotlib.pyplot as plt

from gaussian_splatting.gaussian_renderer import render_with_custom_resolution

import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def torch_images_to_dust3r_format(tensor_images, size, square_ok=False, verbose=False):
    """
    Convert a list of torch tensor images to the format required by the DUSt3R model.
    
    Args:
    - tensor_images (list of torch.Tensor): List of RGB images in torch tensor format.
    - size (int): Target size for the images.
    - square_ok (bool): Whether square images are acceptable.
    - verbose (bool): Whether to print verbose messages.

    Returns:
    - list of dict: Converted images in the required format.
    """
    imgs = []
    for idx, image in enumerate(tensor_images):
        image = image.permute(1, 2, 0).cpu().numpy() * 255  # Convert to HWC format and scale to [0, 255]
        image = image.astype(np.uint8)
        
        img = PIL.Image.fromarray(image, 'RGB')
        img = exif_transpose(img).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not square_ok and W == H:
                halfh = 3 * halfw // 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        #if verbose:
        #    print(f' - processed image {idx} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32([img.size[::-1]]), idx=idx, instance=str(idx)))

    assert imgs, 'no images found'
    #if verbose:
    #    print(f' (Processed {len(imgs)} images)')
    return imgs

# 从深度图转到3D点云坐标
def depth_to_3d(depth_map, K, dist_coeffs):
    """
    将深度图转换为3D点，考虑相机畸变。
    
    参数:
    - depth_map: 深度图
    - K: 相机内参矩阵
    - dist_coeffs: 相机畸变系数 [k1, k2, p1, p2, k3]

    返回:
    - points_3d: 3D点云
    """
    if len(depth_map.shape) == 3:
        # 如果是 (C, H, W)，我们去掉通道维度
        depth_map = depth_map.squeeze(0)  # 去掉第一个维度 C
    
    h, w = depth_map.shape

    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack((u, v), axis=-1).reshape(-1, 2).astype(np.float32)  # 确保为 float32 类型

    # 校正畸变后的像素坐标
    undistorted_pixels = cv2.undistortPoints(pixels, K, dist_coeffs, P=K).reshape(h, w, 2)

    # 获取校正后的u', v'
    u_undistorted = undistorted_pixels[..., 0]
    v_undistorted = undistorted_pixels[..., 1]

    # 获取深度值
    Z = depth_map

    # 根据内参矩阵将像素坐标转换为3D点
    X = (u_undistorted - K[0, 2]) * Z / K[0, 0]
    Y = (v_undistorted - K[1, 2]) * Z / K[1, 1]

    # 将X, Y, Z堆叠为3D点
    points_3d = np.stack((X, Y, Z), axis=-1)

    return points_3d

def depth_to_3d1(depth_map, K):
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_map
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    points_3d = np.stack((X, Y, Z), axis=-1)
    return points_3d

## 估计相对位姿，并返回渲染深度
def get_pose(img1, img2, model, dist_coeffs, viewpoint, gaussians, pipeline_params, background):
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    
    ## 从图像中提取特征，并进行点匹配
    images = torch_images_to_dust3r_format([img1, img2], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()     # 用于点匹配的图像特征

    #点云
    #scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
    #pts = scene.get_pts3d()
    #pts1 = pts[0]
    #pts2 = pts[1]
    #pts1 = pred1['pts3d'].squeeze(0)
    #pts2 = pred2['pts3d_in_other_view'].squeeze(0)
    #z1 = pts1[...,2]
    #z2 = pts2[...,2]
    
    # find 2D-2D matches between the two images 特征点匹配
    matches_im1, matches_im2 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)
    
    H1 = view1['img'].shape[2]      # 输入到特征提取网络中的尺寸
    W1 = view1['img'].shape[3]
    #print(img1.size)
    scale_H = H1 / viewpoint.image_height
    scale_W = W1 / viewpoint.image_width
    
    render_pkg = render_with_custom_resolution(viewpoint, gaussians, pipeline_params, background, target_width=W1, target_height=H1)
    render_depth = render_pkg["depth"]
    #print("不为0的像素数量:", torch.count_nonzero(render_depth))
    #print("不为0的像素坐标:", torch.nonzero(render_depth))

    
    # 调整相机的内参矩阵
    fx_new = viewpoint.fx * scale_W
    fy_new = viewpoint.fy * scale_H
    cx_new = viewpoint.cx * scale_W
    cy_new = viewpoint.cy * scale_H
    
    K_new = np.array([
        [fx_new, 0, cx_new],
        [0, fy_new, cy_new],
        [0, 0, 1]
    ])
    
    # 生成逐像素点云
    pts3d = depth_to_3d(render_depth.detach().cpu().numpy(), K_new, dist_coeffs=dist_coeffs)
    #print(pts3d)

    # 提取图1中的3D点和图2中的对应2D点
    objectPoints = pts3d[matches_im1[:, 1].astype(int), matches_im1[:, 0].astype(int), :]
    objectPoints = objectPoints.astype(np.float32)
    imagePoints = matches_im2.astype(np.float32)

    if len(objectPoints) < 6 or len(imagePoints) < 6:
        print("点数不足，无法执行PnP估计。")
        print("点数为：", len(objectPoints))
        success = False
    else:
        # 执行PnP估计
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints, imagePoints, K_new, dist_coeffs, iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP
        )
    
    if success:
        # 构造4x4齐次变换矩阵
        R, _ = cv2.Rodrigues(rvec)
        pose_w2c = np.eye(4)
        pose_w2c[:3, :3] = R
        pose_w2c[:3, 3] = tvec[:, 0]
        #pose_c2w = np.linalg.inv(pose_w2c)
        return pose_w2c, render_depth.detach().cpu().numpy()   # 返回相对位姿和渲染深度
    else:
        print("PnP估计失败")
        pose_w2c = np.eye(4)
        return pose_w2c, render_depth.detach().cpu().numpy()   # 返回相对位姿和渲染深度
        #raise RuntimeError("PnP估计失败")

## 从MASt3R提取深度
def get_depth(img1, img2, model):
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    
    ## 从图像中提取特征，并进行点匹配
    H1 = img1.shape[1]
    W1 = img1.shape[2]
    
    images = torch_images_to_dust3r_format([img1, img2], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    pts1 = pred1['pts3d'].squeeze(0)
    z1 = pts1[...,2]
    z1 = z1.detach().cpu().numpy()
    z1_resized = cv2.resize(z1, (W1,H1), interpolation=cv2.INTER_NEAREST)
   
    return z1_resized
    
## 绘制渲染深度对比图
def save_depth_comparison(render_depth, mono_depth, rgb, cur_frame_idx, save_dir):
    '''
    输入：
        - render_depth: 渲染深度, (H,W)或(C,H,W) numpy
        - mono_depth: 单目深度估计, (H,W) numpy
        - rgb: rgb图像, (C,H,W) torch
        - cur_frame_idx: 当前帧索引
        - save_dir: 结果保存路径 
    '''
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 将 render_depth 转换为 [H, W] 格式
    if render_depth.ndim == 3:
        render_depth = render_depth.squeeze(0)
    
    # 将 RGB 图像从 [C, H, W] 转换为 [H, W, C] 并转换为 numpy 格式
    rgb_image = rgb.permute(1, 2, 0).cpu().numpy()  # 将 torch.Tensor 转为 numpy
    
    # 确保 RGB 图像尺寸与渲染深度一致
    H, W = render_depth.shape
    if rgb_image.shape[:2] != (H, W):
        rgb_image = cv2.resize(rgb_image, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # 确保 mono_depth 尺寸与渲染深度一致
    if mono_depth.shape != (H, W):
        mono_depth = cv2.resize(mono_depth, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # 归一化深度图
    render_depth_norm = (render_depth - render_depth.min()) / (render_depth.max() - render_depth.min())
    mono_depth_norm = (mono_depth - mono_depth.min()) / (mono_depth.max() - mono_depth.min())
    
    # 计算深度误差
    depth_error = np.abs(render_depth - mono_depth)
    depth_error_norm = (depth_error - depth_error.min()) / (depth_error.max() - depth_error.min())
    
    # 设置标题和布局
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Frame {cur_frame_idx}", fontsize=20, y=0.93)
    
    # 绘制渲染深度
    render0 = axes[0,0].imshow(render_depth_norm, cmap="viridis", vmin=0, vmax=1)
    axes[0,0].set_title("Rendered Depth", fontsize=15)
    axes[0,0].axis("off")
    
    # 绘制单目估计深度
    axes[0,1].imshow(mono_depth_norm, cmap="viridis", vmin=0, vmax=1)
    axes[0,1].set_title("MASt3R Mono Depth", fontsize=15)
    axes[0,1].axis("off")
    
    # 添加一个共享颜色条到上方两张深度图之间
    cbar = fig.colorbar(render0, ax=axes[0, :], orientation="horizontal", fraction=0.05, pad=0.1)
    cbar.set_label("Normalized Depth Value", fontsize=12)
    
    # 绘制深度误差图
    error = axes[1,0].imshow(depth_error_norm, cmap="magma", vmin=0, vmax=1)
    axes[1,0].set_title("Depth Error", fontsize=15)
    axes[1,0].axis("off")
    
    # 为误差图添加单独的颜色条
    cbar_error = fig.colorbar(error, ax=axes[1, 0], orientation="horizontal", fraction=0.05, pad=0.1)
    cbar_error.set_label("Normalized Depth Error", fontsize=12)
    
    # 绘制 RGB 图像
    axes[1,1].imshow(rgb_image)
    axes[1,1].set_title("RGB", fontsize=15)
    axes[1,1].axis("off")
    
    # 保存图片
    save_path = os.path.join(save_dir, f"{cur_frame_idx}.png")
    plt.savefig(save_path)
    plt.close(fig)
    
    return save_path


## 计算逐帧估计的误差
def compute_pose_error(T1, T2, T3, T_gt):
    """
    计算T1, T2, T3和T_gt之间的平移误差和角度误差，并返回6维数组。
    
    参数:
    - T1: 上一帧位姿的4x4 W2C矩阵 (torch.Tensor)
    - T2: 经过PNP估计的初始化位姿4x4 W2C矩阵 (torch.Tensor)
    - T3: 经过优化的位姿4x4 W2C矩阵 (torch.Tensor)
    - T_gt: 这一帧的GT位姿4x4 W2C矩阵 (torch.Tensor)
    
    返回:
    - 一个包含6个值的数组：
      [T1的平移误差, T1的角度误差, T2的平移误差, T2的角度误差, T3的平移误差, T3的角度误差]
    """

    def calculate_angle_translation_error(T_est, T_gt):
        # 提取平移向量
        t_est = T_est[:3, 3]
        t_gt = T_gt[:3, 3]

        # 计算平移误差（L2范数）
        translation_error = torch.norm(t_est - t_gt)

        # 提取旋转矩阵
        R_est = T_est[:3, :3]
        R_gt = T_gt[:3, :3]

        # 计算相对旋转矩阵
        R_error = R_est @ R_gt.T

        # 使用torch的罗德里格斯变换来获得旋转向量（轴角表示）
        rvec, _ = cv2.Rodrigues(R_error.cpu().numpy())
        rvec = torch.tensor(rvec, device=T_est.device)

        # 旋转向量的模即为角度误差（弧度）
        angle_error = torch.norm(rvec)

        # 将弧度转换为角度
        angle_error_degrees = torch.rad2deg(angle_error)

        return translation_error, angle_error_degrees

    # 计算 T1 和 T_gt 之间的误差
    trans_error_T1, angle_error_T1 = calculate_angle_translation_error(T1, T_gt)

    # 计算 T2 和 T_gt 之间的误差
    trans_error_T2, angle_error_T2 = calculate_angle_translation_error(T2, T_gt)

    # 计算 T3 和 T_gt 之间的误差
    trans_error_T3, angle_error_T3 = calculate_angle_translation_error(T3, T_gt)

    # 返回6维数组 [T1的平移误差, T1的角度误差, T2的平移误差, T2的角度误差, T3的平移误差, T3的角度误差]
    return [trans_error_T1, angle_error_T1, trans_error_T2, angle_error_T2, trans_error_T3, angle_error_T3]