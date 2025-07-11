import random
import time

import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
import os

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping, get_loss_mapping_rgbd, get_loss_mapping_rgb
from utils.init_pose import save_depth_comparison
from utils.depth_utils import process_depth, process_depth_torch


class BackEnd(mp.Process):
    def __init__(self, config, save_dir=None):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False
        self.save_dir = save_dir

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        
        self.pcd_scale = 1
        self.theta = 0
        #self.matches_im0_init = None
        #self.matches_im1_init = None
        #self.matches_3d0_init = None
    # 从配置文件中读取和设置超参数
    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.global_BA_itr_num = self.config["Training"]["global_BA_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
    # 通过新关键帧的viewpoint和深度信息，得到一组3D Gaussian点并添加到现有的Gaussian序列中
    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )
    # 将dust3r点云加入到GS场景    
    def add_next_kf_dust3r(self, frame_idx, pts3d, imgs, T, mask=None, init=False, scale=1):
        fused_point_cloud, features, scales, rots, opacities = (
            self.gaussians.create_pcd_from_dust3r(pts3d, imgs, T, frame_idx, self.save_dir, scale, mask, init=init)
        )
        self.gaussians.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, frame_idx
        )
    # 重置后端状态。清空关键帧数据和其他后端数据、清空高斯点和后端sequence。
    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()
    # 初始化 SLAM 地图，通过多次迭代来优化Gaussians的参数，最终建立一个初始的三维地图，记录occ_aware_visibility，并返回最后一次迭代的render结果
    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, viewpoint, depth=depth,initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(  # 更新 max_radii2D，取当前和新计算半径的最大值？
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(                 # 添加稠密化统计信息
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:  # 每隔一定迭代次数，执行稠密化和剪枝操作
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )
                # 在指定次数迭代时重置不透明度
                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()                         # 更新参数
                self.gaussians.optimizer.zero_grad(set_to_none=True)    # 清零优化器梯度

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()   # 更新 occ_aware_visibility，记录当前帧的可见性信息
        Log("Initialized map")
        return render_pkg
    # 执行地图优化和高斯分布数据的稠密化和修剪。通过多次迭代渲染和损失计算，更新视角和高斯分布数据，并记录当前帧的可见性。
    # current_window 当前窗口，prune 是否修剪，iters 迭代次数
    def map(self, current_window, prune=False, iters=1, up_pose = True):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)            # set是什么意思？
        for cam_idx, viewpoint in self.viewpoints.items():  # 将不在当前窗口中的其他viewpoint添加到 random_viewpoint_stack
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)
            
        # 确保BA窗口内的viewpoint的pose都是groundtruth
        #for viewpoint in viewpoint_stack:
        #    viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        #for viewpoint in random_viewpoint_stack:
        #    viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
            
            
        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []                 # 累积所有关键帧的视点空间点
            visibility_filter_acm = []                      # 累积所有关键帧的可见性过滤器
            radii_acm = []                                  # 累积所有关键帧的半径
            n_touched_acm = []                              # 累积所有关键帧的触碰计数

            keyframes_opt = []          # 这个后续没用到！

            for cam_idx in range(len(current_window)):      # 对于当前窗口中的每个关键帧，执行渲染操作并计算损失；累积损失 loss_mapping，并将渲染结果存储到相应的累积列表中
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (                                           # 这些属性需要去看看render代码
                    image,
                    viewspace_point_tensor,                 # 视点空间中的点云数据
                    visibility_filter,                      # 可见性过滤器，标识哪些点是可见的
                    radii,                                  # 高斯点的半径
                    depth,                                  # 深度图
                    opacity,                                # 不透明度
                    n_touched,                              # 触碰计数，表示高斯点被视点触碰的次数
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                ##if up_pose:     # mono_depth表示需不需要深度监督
                ##    loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=False)
                ##else:
                ##    loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=True)
                loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=True)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)     # 用于累积所有关键帧的 n_touched 数据，以便在后续步骤中更新可见性信息。
                
            # 每次迭代都会随机抽取两个非窗口中的关键帧加入优化
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:     # 随机选择两个不在当前窗口中的视点，执行渲染操作并计算损失，和上面类似，但注意保存的东西有些不同！
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                ##if up_pose:     # mono_depth表示需不需要深度监督
                ##    loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=False)
                ##else:
                ##    loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=True)
                loss_mapping += get_loss_mapping(self.config, image, viewpoint, depth=depth, monodepth=True)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                
            # 计算各向同性损失并累加到总损失中
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}              # 更新当前窗口中每个关键帧的可见性信息
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:       # 如果启用了剪枝，并且当前窗口的大小等于配置的窗口大小，则执行剪枝操作；有不同模式；更新可见性信息并执行剪枝操作
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = self.config["Training"]["prune_num"]
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:         # 只有在单目情况下才执行剪枝操作，RGB-D情况下不进行剪枝。
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (                  # 布尔索引，只保留那些不需要被剪枝的高斯点的可见性
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False
                # 更新高斯分布的最大半径，并添加稠密化统计信息
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )
                # 每一定迭代次数后，执行高斯分布的稠密化和剪枝操作
                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## 每一定迭代次数后，执行Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian) :
                    # and (self.config["Dataset"]["type"]!='waymo')
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True
                # 更新高斯分布和关键帧优化器的参数
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                if up_pose:
                    for cam_idx in range(min(frames_to_optimize, len(current_window))):
                        viewpoint = viewpoint_stack[cam_idx]
                        if viewpoint.uid == 0:
                            continue
                        update_pose(viewpoint)
        return gaussian_split
    # Global BA，不需要用。
    def globa_BA(self, kf_indices):
        iters = self.config["Training"]["global_BA_itr_num"]
        
        for iteration in tqdm(range(1, iters + 1)):
            viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in kf_indices]
            
            loss_mapping = 0
            viewspace_point_tensor_acm = []                 # 累积所有关键帧的视点空间点
            visibility_filter_acm = []                      # 累积所有关键帧的可见性过滤器
            radii_acm = []                                  # 累积所有关键帧的半径
            n_touched_acm = []                              # 累积所有关键帧的触碰计数

            for cam_idx in range(len(kf_indices)):      # 对于当前窗口中的每个关键帧，执行渲染操作并计算损失；累积损失 loss_mapping，并将渲染结果存储到相应的累积列表中
                #print(cam_idx)
                #print(torch.cuda.memory_summary())
                viewpoint = viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (                                           # 这些属性需要去看看render代码
                    image,
                    viewspace_point_tensor,                 # 视点空间中的点云数据
                    visibility_filter,                      # 可见性过滤器，标识哪些点是可见的
                    radii,                                  # 高斯点的半径
                    #depth,                                  # 深度图
                    #opacity,                                # 不透明度
                    #n_touched,                              # 触碰计数，表示高斯点被视点触碰的次数
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    #render_pkg["depth"],
                    #render_pkg["opacity"],
                    #render_pkg["n_touched"],
                )
                
                torch.cuda.empty_cache()

                loss_mapping += get_loss_mapping(
                    self.config, image, viewpoint
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                #n_touched_acm.append(n_touched)     # 用于累积所有关键帧的 n_touched 数据，以便在后续步骤中更新可见性信息。
            del render_pkg, image, viewspace_point_tensor, visibility_filter, radii
            torch.cuda.empty_cache()
            
            loss_mapping.backward()
            with torch.no_grad():
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )
                    
                del viewspace_point_tensor_acm, visibility_filter_acm, radii_acm
                torch.cuda.empty_cache()
                
                # 每一定迭代次数后，执行高斯分布的稠密化和剪枝操作
                update_gaussian = (
                    iteration % 100
                    == 0
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    #gaussian_split = True

                ## 每一定迭代次数后，执行Opacity reset
                if (iteration % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True
                #  更新高斯分布和关键帧优化器的参数
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(len(kf_indices)):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
                
    # 执行颜色优化。通过多次迭代，使用 L1 损失和 SSIM 损失优化渲染结果与gt图像之间的差异。用于在SLAM最后优化地图。
    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())      # 获取视点索引列表
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]      # 随机选择一个视点
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():       # 在无梯度计算的上下文中，更新高斯点的最大半径；执行优化器的步进操作，更新高斯分布参数；清零优化器的梯度；更新学习率。
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(26000)
        Log("Map refinement done")
    # 将当前窗口的关键帧数据和高斯分布数据推送到前端
    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
            
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)
    # 主运行循环。处理后端队列中的消息，执行地图优化、颜色优化、初始化和关键帧管理等操作。同步数据并推送到前端。
    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:            # 这个在优化过程中是什么变化的？
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "global_BA":
                    kf_indices = data[1]
                    opt_params = []
                    for cam_idx in kf_indices:     # 为每个关键帧添加优化参数
                        viewpoint = self.viewpoints[cam_idx]       
                        opt_params.append(
                            {
                                "params": [viewpoint.cam_rot_delta],
                                "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                * 0.5,
                                "name": "rot_{}".format(viewpoint.uid),
                            }                            
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.cam_trans_delta],
                                "lr": self.config["Training"]["lr"][
                                    "cam_trans_delta"
                                ]
                                * 0.5,
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
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)
                    self.globa_BA(kf_indices)
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    pts3d = data[4]
                    imgs = data[5]
                    mask = data[6]
                    self.scale = 1
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    # T = torch.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T))
                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    #self.add_next_kf_dust3r(cur_frame_idx, pts3d, imgs, T, mask, init=True, scale=self.scale)
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    pts3d = data[5]
                    imgs = data[6]
                    mask = data[7]
                    self.scale = data[8]
                    self.theta = data[9]
                    theta_value = self.theta.item()
                    if (theta_value >= 2) and (self.config["Dataset"]["type"] == 'waymo1'):     ## 执行学习率调整，因为是'waymo1'所以都没执行
                        self.iteration_count = self.iteration_count * (1-np.sqrt(theta_value / 90))
                       # #self.iteration_count = self.iteration_count * (1-(theta_value / 90))
                        self.iteration_count = int(self.iteration_count)
                        self.gaussians.update_learning_rate(self.iteration_count)
                    #print("当前帧数为:", cur_frame_idx, "累积迭代次数为:", self.iteration_count)
                    print("current keyframe ",cur_frame_idx,'window is ',current_window)
                    
                    # 绘制渲染深度对比图
                    ##save_depth_comparison(depth_map, viewpoint.mono_depth, viewpoint.original_image, cur_frame_idx, os.path.join(self.save_dir, "插入新高斯depth"))

                    #T = torch.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T))
                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
                    #self.add_next_kf_dust3r(cur_frame_idx, pts3d, imgs, T, mask, scale=self.scale)
                    # 优化参数设置
                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_nosingle = self.config["Training"]["mapping_itr_nosingle"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else iter_nosingle
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        #    > self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):     # 为每个关键帧添加优化参数
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:        
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
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
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    #self.map(self.current_window, iters=50, up_pose=False)
                    self.map(self.current_window, iters=iter_per_kf, up_pose=True)
                    #self.map(self.current_window, prune=False)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
