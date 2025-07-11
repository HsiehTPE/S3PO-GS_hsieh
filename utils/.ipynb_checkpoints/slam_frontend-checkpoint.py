import time

import numpy as np
import torch
import torch.multiprocessing as mp
import os

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
#from utils.dust3r_utils import get_result, get_scale
from utils.init_pose import save_depth_comparison, get_pose, compute_pose_error, get_depth
from utils.depth_utils import process_depth, process_depth_torch

# SLAM的前端类，功能包括：
class FrontEnd(mp.Process):
    def __init__(self, config,model, save_dir=None):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None
        self.save_dir = save_dir

        self.initialized = False            # 是否已经初始化：RGD-D只要做了第一帧地图初始化即可；RGD单目需要填满keyframe window且没有remove过kf才算，即要有足够的overlap
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        
        # 位姿确定相关
        self.model = model 
        self.pose_error = []
        
        # 存储dust3r相关的信息
        #self.d3r_model = d3r_model
        self.last_color = None
        self.pts3d = None
        self.imgs = None
        self.mask = None
        self.matches_im0 = None
        self.matches_im1 = None
        self.matches_3d0 = None
        self.scale = 1
        self.scale1 = 1
        self.theta = 0
    # 从配置文件中读取和设置超参数
    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]       # ？
    # 添加新的关键帧。根据配置中的 RGB 边界阈值生成有效像素mask，然后生成初始深度图
    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        if len(self.kf_indices) > 0:
            last_kf = self.kf_indices[-1]
            viewpoint_last = self.cameras[last_kf]
            R_last = viewpoint_last.R
        self.kf_indices.append(cur_frame_idx)
        #print("key frame is:",cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        R_now = viewpoint.R
        if len(self.kf_indices) > 1:
            # 计算和上一帧的角度差
            R_now = R_now.to(torch.float32)
            R_last = R_last.to(torch.float32)
            R_diff = torch.matmul(R_last.T, R_now)
            trace_R_diff = torch.trace(R_diff)
            theta_rad = torch.acos((trace_R_diff - 1) / 2)
            theta_deg = torch.rad2deg(theta_rad)
            self.theta = theta_deg
        #print("角度差为:",self.theta)
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]      # 判断三通道的像素值和是否大于边界阈值，添加一个新的维度以匹配后续操作的形状。
        a_invalid = self.config["Training"]["a_invalid"]
        a_valid = self.config["Training"]["a_valid"]
        if self.monocular:
            if depth is None:
                #initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2]) # 创建全为2的深度图
                #initial_depth += torch.randn_like(initial_depth) * 0.3              # 添加随机噪声以模拟初始深度的不确定性
                initial_depth = torch.from_numpy(viewpoint.mono_depth).unsqueeze(0)     # 初始化地图时使用MASt3R估计深度
                print("第", cur_frame_idx, "帧的初始化深度图信息:", f"最大值: {torch.max(initial_depth).item()}",  f"最小值: {torch.min(initial_depth).item()}", 
                        f"均值: {torch.mean(initial_depth).item()}", f"中位数: {torch.median(initial_depth).item()}", f"标准差: {torch.std(initial_depth).item()}")
                initial_depth[~valid_rgb.cpu()] = 0
                return initial_depth[0].numpy()
            else:    
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:       # 逆深度这种表示方法在计算机视觉和 SLAM 中有时更为稳定；深度值在较大的范围内变化，逆深度能将其缩放到一个较小的范围。处理和筛选无效深度值时，逆深度可以更有效地减少极值对结果的影响？
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                #elif self.config["Dataset"]["type"] == 'waymo':
                else:
                    #invalid_depth_mask = depth < 0
                    #depth[invalid_depth_mask] = 0
                    initial_depth = depth
                #else:      # 对无效的depth使用depth_median，然后加一个随机噪声
                #    median_depth, std, valid_mask = get_median_depth(
                #        depth, opacity, mask=valid_rgb, return_std=True
                #    )
                #    invalid_depth_mask = torch.logical_or(
                #        depth > median_depth + std, depth < median_depth - std
                #    )
                #    invalid_depth_mask = torch.logical_or(
                #        invalid_depth_mask, ~valid_mask
                #    )
                #    depth[invalid_depth_mask] = median_depth
                #    initial_depth = depth + torch.randn_like(depth) * torch.where(
                #        invalid_depth_mask, std * a_invalid, std * a_valid        # std * 0.5：无效深度值对应的噪声幅度；0.2是有效
                #    )
                #print("第", cur_frame_idx, "帧的初始化深度图信息:", f"最大值: {torch.max(initial_depth).item()}",  f"最小值: {torch.min(initial_depth).item()}", 
                #        f"均值: {torch.mean(initial_depth).item()}", f"中位数: {torch.median(initial_depth).item()}",f"标准差: {torch.std(initial_depth).item()}")
                
                # 调整渲染深度和深度估计尺度
                render_depth = initial_depth.cpu().numpy()[0]
                
                initial_depth, scale_factor, error_mask, num_accurate_pixels = process_depth(render_depth, viewpoint.mono_depth, last_depth = viewpoint_last.mono_depth, 
                                                                                             im1 = viewpoint_last.original_image, im2 = viewpoint.original_image, model = self.model,
                                                                                             patch_size = self.config["depth"]["patch_size"], 
                                                                                             mean_threshold = self.config["depth"]["mean_threshold"], std_threshold = self.config["depth"]["std_threshold"],
                                                                                             error_threshold = self.config["depth"]["error_threshold"], final_error_threshold = self.config["depth"]["final_error_threshold"],
                                                                                             min_accurate_pixels_ratio = self.config["depth"]["min_accurate_pixels_ratio"])
                #render_depth = initial_depth.float().cuda() if not initial_depth.is_cuda else initial_depth
                #initial_depth, scale_factor, error_mask, num_accurate_pixels = process_depth_torch(render_depth, viewpoint.mono_depth)
                #scale_factor_np = scale_factor.item() if isinstance(scale_factor, torch.Tensor) else scale_factor

                ### 消融
                viewpoint.mono_depth = viewpoint.mono_depth * scale_factor
                
                #initial_depth = render_depth
                pixel_num = viewpoint.image_height * viewpoint.image_width
                
                print("第", cur_frame_idx, "帧的初始化深度图信息:", f"最大值: {np.max(initial_depth)}", f"最小值: {np.min(initial_depth)}", f"均值: {np.mean(initial_depth)}",
                      f"中位数: {np.median(initial_depth)}", f"标准差: {np.std(initial_depth)}", f"尺度因子: {scale_factor}" ,f"准确像素比例: {num_accurate_pixels / pixel_num}", f"填补像素比例: {np.sum(error_mask) / pixel_num}")
                initial_depth_tensor = torch.from_numpy(initial_depth).cuda()
                
                #print("第", cur_frame_idx, "帧的初始化深度图信息:", f"最大值: {torch.max(initial_depth).item()}",  f"最小值: {torch.min(initial_depth).item()}", 
                #        f"均值: {torch.mean(initial_depth).item()}", f"中位数: {torch.median(initial_depth).item()}",f"标准差: {torch.std(initial_depth).item()}",
                #        f"尺度因子: {scale_factor}" , f"准确像素比例: {num_accurate_pixels/pixel_num}", f"填补像素比例: {error_mask.sum().item()/pixel_num}")
                
                #initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
                
                valid_rgb_np = valid_rgb.cpu().numpy() if isinstance(valid_rgb, torch.Tensor) else valid_rgb

                # 检查尺寸是否匹配并修改 initial_depth
                if initial_depth.shape == valid_rgb_np.shape[1:]:
                    initial_depth[~valid_rgb_np[0]] = 0  # 处理 valid_rgb[None] 的切片问题
            return initial_depth
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)      # 从 viewpoint 中获取深度图，并添加一个新的维度
        print("第", cur_frame_idx, "帧的初始化深度图信息:", f"最大值: {torch.max(initial_depth).item()}",  f"最小值: {torch.min(initial_depth).item()}", f"均值: {torch.mean(initial_depth).item()}", f"标准差: {torch.std(initial_depth).item()}")
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()      # initial_depth 是一个 4D 张量 (1, C, H, W)，取第一个通道的数据 (C, H, W)
    # 初始化 SLAM 系统。在初始化过程中，清空后端队列，重置状态，并将当前帧设置为groundtruth姿态，然后添加一个新的关键帧，并将相关信息放入后端队列。
    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose ？？
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        # 得到估计深度
        img = viewpoint.original_image
        viewpoint.mono_depth = get_depth(img, img, self.model)
        
        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)      # 请求初始化，并将相关信息放入后端队列
        self.reset = False
    # 执行当前帧的跟踪。更新当前视角的旋转和平移矩阵，然后设置优化参数，并使用 Adam 优化器进行迭代优化；每 10 次迭代，将当前状态发送到可视化队列；记录深度渲染中值；返回最后一次迭代的render结果。
    def tracking(self, cur_frame_idx, viewpoint):
        #print("current frame is:",cur_frame_idx)
        
        ##=========================================位姿初始化部分==================================================
        
        #print("当前viewpoint的projection matrix:", viewpoint.projection_matrix)
        #print("当前viewpoint的world_view_transform:", viewpoint.world_view_transform)
        #print("当前viewpoint的full_proj_transform:", viewpoint.full_proj_transform)
        
        # 上一帧的位姿
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        pose_prev = getWorld2View2(prev.R, prev.T)
        
        # 上一关键帧的信息
        last_keyframe_idx = self.current_window[0]
        last_kf = self.cameras[last_keyframe_idx]
        pose_last_kf = getWorld2View2(last_kf.R, last_kf.T)
        img1 = last_kf.original_image
        
        # 进行相对位姿推断
        img2 = viewpoint.original_image
        #print(img1.shape)
        #print(self.dataset.dist_coeffs)
        rel_pose, render_depth = get_pose(img1=img1, img2=img2, model=self.model, dist_coeffs=self.dataset.dist_coeffs, 
                            viewpoint=last_kf, gaussians=self.gaussians, pipeline_params=self.pipeline_params, background=self.background)
        # 得到MASt3R估计深度
        #if self.config["Dataset"]["type"]=='waymo':
        viewpoint.mono_depth = get_depth(img2, img2, self.model)
        
        #_, depth_now2, depth_now3, = get_pose(img1=img2, img2=img2, model=self.model, dist_coeffs=self.dataset.dist_coeffs, 
        #                    viewpoint=last_kf, gaussians=self.gaussians, pipeline_params=self.pipeline_params, background=self.background)
        #print("不同帧输入得到当前帧深度:", depth_now1)
        #print("同样图片输入得到当前帧深度1:", depth_now2)
        #print("同样图片输入得到当前帧深度2:", depth_now3)
        #print("一二差距:",depth_now1-depth_now2)
        #print("一三差距:",depth_now1-depth_now3)
        #print("二三差距:",depth_now2-depth_now3)
        #print("一二差距均值:",torch.mean(abs(depth_now1-depth_now2)))
        #print("一三差距均值:",torch.mean(abs(depth_now1-depth_now3)))
        #print("二三差距均值:",torch.mean(abs(depth_now2-depth_now3)))

        ### 位姿初始化
        identity_matrix = torch.eye(4, device=self.device)
        rel_pose = torch.from_numpy(rel_pose).to(self.device).float()
        if torch.allclose(rel_pose, identity_matrix, atol=1e-6):  # atol 是绝对容忍误差
            # 如果是单位矩阵，执行更新 prev.R 和 prev.T
            pose_init = rel_pose @ pose_last_kf
            viewpoint.update_RT(prev.R, prev.T)
        else:
            # 否则执行原始的相对位姿更新
            #rel_pose = torch.from_numpy(rel_pose).to(self.device).float()
            pose_init = rel_pose @ pose_last_kf
            viewpoint.update_RT(pose_init[:3, :3], pose_init[:3, 3])
        
        ##pose_init = rel_pose @ pose_last_kf
        ##viewpoint.update_RT(pose_init[:3,:3], pose_init[:3,3])

        ### 使用上一帧位姿初始化(消融)
        #viewpoint.update_RT(prev.R, prev.T)
        
        ## 绘制渲染深度对比图
        ##save_depth_comparison(render_depth, viewpoint.mono_depth, img2, cur_frame_idx, os.path.join(self.save_dir, "位姿估计depth"))
        
        ## dust3r
        #trans_pose ,pts3d, imgs, matches_im0, matches_im1, matches_3d0=get_result(viewpoint.original_image, self.last_color, model=self.d3r_model, device=self.device)
        #self.pts3d = pts3d
        #self.imgs = imgs
        #scale1, scale = get_scale(self.matches_im1, self.matches_im0, matches_im1, matches_im0, self.matches_3d0, matches_3d0)
        #self.scale = self.scale * scale
        #self.matches_im0 = matches_im0
        #self.matches_im1 = matches_im1
        #self.matches_3d0 = matches_3d0
        # self.mask = 
        #trans_pose[:3,3] = trans_pose[:3,3]/self.scale
        #trans_pose_inv = np.linalg.inv(trans_pose)
        #trans_pose_inv_torch = torch.from_numpy(trans_pose_inv).to(self.device)
        
        #w2c1 = getWorld2View2(prev.R, prev.T)
        #w2c2 = trans_pose_inv_torch @ w2c1
        
        # 计算 C2W 矩阵
        #c2w1 = torch.linalg.inv(w2c1)
        #c2w2 = torch.linalg.inv(w2c2)
        
        #c2w1 = torch.linalg.inv(getWorld2View2(prev.R, prev.T))
        #print("last frame c2w is:",'\n' , c2w1)
        #c2w2 = c2w1 @ trans_pose
        #print("after dust3r init c2w is:", '\n', c2w2)
        #print("dust3r初始化后位移距离为:", c2w2[:3,3]- c2w1[:3,3])
        #w2c2 = torch.linalg.inv(c2w2)
        
        #viewpoint.update_RT(w2c2[:3,:3],w2c2[:3,3])         # 基于dust3r得到的pose作为初始化
        
        ## MonoGS
        #viewpoint.update_RT(prev.R, prev.T)         # 将前一个帧的旋转矩阵和平移矩阵赋值给当前视点的旋转矩阵和平移矩阵。初始化？？⭐
        # viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)         # pose使用ground truth

        ## ======================================位姿优化部分========================================================
        opt_params = []     # 曝光参数 a 和 b，用于调整图像的曝光效果。通过优化这些参数，使得渲染的图像更符合实际拍摄图像
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint) 
                #converged = True

            if tracking_itr % 10 == 0:              
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break
           
        #c2w3 = torch.linalg.inv(getWorld2View2(viewpoint.R, viewpoint.T))
        
        ## 提取优化后的位姿并计算逐步误差
        pose_opti = getWorld2View2(viewpoint.R, viewpoint.T)
        pose_gt = getWorld2View2(viewpoint.R_gt, viewpoint.T_gt)
        
        pose_error = compute_pose_error(pose_prev, pose_init, pose_opti, pose_gt)
        self.pose_error.append(pose_error)
        
        #print("after tracking optimization, c2w is:", '\n', c2w3)
        ## dust3r需要恢复
        #self.scale1 = self.scale1 * scale1
        #print("tracking优化后，位移距离为:", c2w3[:3,3]- c2w2[:3,3])
        #print("当前帧算得的平均数scale比例为:",scale1)
        #print("当前帧的累积平均数scale比例为:", self.scale1)
        #print("当前帧算得的中位数scale比例为:",scale)
        #print("当前帧的累积中位数scale比例为:", self.scale)
        
        # viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)         # pose使用ground truth
        #pose_dpvo = self.poses_dpvo[cur_frame_idx]
        #pose_dpvo = torch.from_numpy(pose_dpvo).to(self.device)
        #viewpoint.update_RT(pose_dpvo[:3,:3], pose_dpvo[:3,3])
        
        self.median_depth = get_median_depth(depth, opacity)        # 渲染深度的中值，用于后续判断是否为关键帧
        return render_pkg
    
    # 判断当前帧是否为关键帧。通过计算当前帧和上一个关键帧之间的平移距离和重叠区域的比例，来确定是否需要将当前帧设为关键帧
    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)        # 这里应该还是W2C矩阵
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)                   # C2W矩阵
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])           # 得到一个从当前帧到上一个关键帧的变换矩阵，然后提取平移部分，求距离
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check       # 共视区域小或相机位移大
    
    # 将当前帧添加到窗口中，并根据与其他关键帧的重叠比例移除窗口中重叠较少的关键帧。确保窗口大小不超过设定值，并返回更新后的窗口和移除的关键帧
    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if (point_ratio_2 <= cut_off) and (len(window) > self.config["Training"]["window_size"]):        
            #if (point_ratio_2 <= cut_off):
                to_remove.append(kf_idx)
        # 移除overlap小于阈值的关键帧中最早的那个帧
        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))
        # 如果window内数量超了（前一步没有删掉），则寻找与当前帧最远的那个帧踢出，这里距离乘了一个权重，倾向于删除和其他候选帧逆距离最大（距离最小）的
        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)
        #print("current keyframe ",cur_frame_idx,'window is ',window)
        return window, removed_frame
    
    # 请求添加新的关键帧，并将相关信息放入后端队列
    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, self.pts3d, self.imgs, self.mask, self.scale1, self.theta]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1
    # 请求mapping，并将相关信息放入后端队列
    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)
    # 请求初始化，并将相关信息放入后端队列
    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map, self.pts3d, self.imgs, self.mask, self.scale1]
        self.backend_queue.put(msg)
        self.requested_init = True
    # 从后端同步数据，包括高斯分布数据、遮挡感知可见性和关键帧信息。更新关键帧的旋转和平移矩阵
    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())
    # 清理当前帧的相机数据，并每 10 帧清空一次 CUDA 缓存
    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()
            
    # 主运行循环。处理前端和后端队列中的消息，执行tracking、关键帧管理、初始化请求等操作。同步数据、清理资源并保存结果。
    # 每个帧都进行tracking并判断是否为关键帧，但只对关键帧eval并保存结果
    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(       # 生成并转置投影矩阵 projection_matrix,将三维坐标变换为相机坐标系并进行透视投影，从而获得在图像平面上的二维坐标。
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)      # 创建 CUDA 事件 tic 和 toc 以进行时间测量
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():         # 处理可视化队列
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():     # 检查前端队列 frontend_queue 是否为空，如果为空，则开始处理当前帧
                tic.record()
                if cur_frame_idx >= len(self.dataset):  # 如果当前帧索引超过数据集长度，则评估结果并保存，然后退出循环；
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break
                # 如果请求了初始化，或在单线程模式下有关键帧请求，或尚未初始化但有关键帧请求，则暂停 0.01 秒并继续下一次循环。让线程休眠 0.01 秒以避免高频率的无效循环。这样可以减少 CPU 占用，给其他线程和操作更多的执行机会？
                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                # 使用数据集和投影矩阵初始化当前帧的camera viewpoint，并计算梯度掩码，然后存储到 cameras 列表中。梯度掩码用于确定图像或深度图中哪些部分应该参与梯度计算。它可以用于加速优化过程，忽略那些对优化结果影响较小的区域。
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint
                # 如果需要重置，调用 initialize 函数初始化当前帧，并将其添加到当前窗口，增加帧索引并继续下一次循环
                if self.reset:
                    self.last_color = self.cameras[cur_frame_idx].original_image
                    #_ ,pts3d, imgs, self.matches_im0, self.matches_im1, self.matches_3d0=get_result(self.last_color,self.last_color, model=self.d3r_model, device=self.device)
                    #self.pts3d = pts3d
                    #self.imgs = imgs 
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                self.last_color = self.cameras[cur_frame_idx].original_image
                # 创建current window字典并获取关键帧列表，将这些数据放入 q_main2vis 队列
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                
                #print("pose at this time:",cur_frame_idx,torch.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T)))     # 打印此时此刻可视化用的pose
                #print("gt pose:",cur_frame_idx,torch.linalg.inv(getWorld2View2(viewpoint.R_gt,viewpoint.T_gt)))         # 打印此时此刻可视化用的pose
                
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )
                
                # 如果有关键帧请求，调用 cleanup 函数清理当前帧数据，并增加帧索引？
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval    #帧间隔作为关键帧选择的考虑依据
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,          # 这东西从哪来？
                )
                if len(self.current_window) < self.window_size:     # 如果窗口不满，看共视程度+和上个关键帧有一定距离。是否和上面重复？？
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:      # 如果是单线程模式
                    create_kf = check_time and create_kf
                if create_kf:       # 如果需要添加关键帧，则调用 add_to_window 函数管理窗口中的关键帧，并请求新的关键帧；否则清理当前帧
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )       
                    #if self.monocular and not self.initialized and removed is not None:     # 如果在单目模式下，尚未初始化并且有关键帧被移除，说明关键帧之间缺乏足够的重叠，无法进行有效的初始化，因此需要重置系统。单目需要足够overlap
                    #    self.reset = True
                    #    Log(
                    #        "Keyframes lacks sufficient overlap to initialize the map, resetting."
                    #    )
                    #    continue
                    depth_map = self.add_new_keyframe(      # 增加新的关键帧，并得到深度图，用于后面request_keyframe
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(      # 将新关键帧信息同步到后端
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1              # 增加帧index，这代表当前帧

                if (                # 如果需要保存结果且创建了关键帧，并且关键帧索引满足保存间隔，则调用 eval_ate 进行评估
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()        # 等待当前设备上所有流中的所有 CUDA 内核完成。这是一个同步操作，确保在继续执行后续代码之前，所有的 CUDA 操作都已完成
                if create_kf:
                    # throttle at 3fps when keyframe is added   意义何在？
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:       # 如果前端队列不为空，获取消息并根据消息类型执行相应操作，包括同步后端数据、处理关键帧请求、处理初始化请求和停止前端。（队列信息由后端发送来）
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
