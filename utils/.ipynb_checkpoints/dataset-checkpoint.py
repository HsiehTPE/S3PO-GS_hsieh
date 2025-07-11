import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
import json
from pathlib import Path

from gaussian_splatting.utils.graphics_utils import focal2fov
from scipy.spatial.transform import Rotation as R

try:
    import pyrealsense2 as rs
except Exception:
    pass

## ===============================================================数据解析器===================================================================================
class TartanAirParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"]
        
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.n_img = len(self.color_paths)
        
        # 读取位姿
        self.load_poses(f"{self.input_folder}/gt.txt")

    def load_poses(self, pose_file):
        """ 从 gt.txt 读取所有位姿，并转换为 4×4 矩阵 """
        self.poses = []
        self.frames = []

        # 读取所有行，每行是 7 元数格式：x, y, z, qx, qy, qz, qw
        all_poses = np.loadtxt(pose_file, delimiter=' ')

        # 选择需要的帧
        selected_poses = all_poses[self.begin:self.end]

        # 计算初始平移，用于对齐
        init_trans = selected_poses[0, :3]

        # 坐标系变换矩阵 (TartanAir → SLAM)
        R_w2slam = np.array([
            [0,  0,  1,  0],  # x → z
            [-1, 0,  0,  0],  # y → -x
            [0, -1,  0,  0],  # z → -y
            [0,  0,  0,  1]
        ])

        for i, pose in enumerate(selected_poses):
            x, y, z, qx, qy, qz, qw = pose

            # 转换为 3×3 旋转矩阵
            rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

            # 构造 4×4 变换矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = [x, y, z] - init_trans  # 归一化平移量
            
            # 转换到 SLAM 坐标系
            pose_slam = R_w2slam @ transform_matrix

            # 求逆矩阵 (SLAM 评估需要)
            inv_pose = np.linalg.inv(pose_slam)
            
            # 存储数据
            self.poses.append(inv_pose)  # 存储的是求逆后的位姿
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],  # 这里可能需要调整
                "mono_depth_path": self.color_paths[i],  # 这里可能需要调整
                "transform_matrix": transform_matrix.tolist(),  # 存储的是原始矩阵
            }
            self.frames.append(frame)

class re10kParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"]
        
        # 读取 RGB 图像路径
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.n_img = len(self.color_paths)
        
        # 读取相机位姿 (camera.json)
        self.load_poses(os.path.join(self.input_folder, "cameras.json"))

    def load_poses(self, pose_file):
        """ 从 camera.json 读取相机位姿，并转换为 4×4 矩阵 """
        self.poses = []
        self.frames = []

        # 读取 JSON 数据
        with open(pose_file, "r") as f:
            all_poses = json.load(f)

        # 选取 [begin:end] 范围内的帧
        selected_poses = all_poses[self.begin:self.end]

        # 计算初始平移，用于对齐
        init_trans = np.array(selected_poses[0]["cam_trans"])

        for i, pose in enumerate(selected_poses):
            # 提取四元数和位移
            qx, qy, qz, qw = pose["cam_quat"]
            tx, ty, tz = pose["cam_trans"]

            # 转换为 3×3 旋转矩阵
            rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

            # 构造 4×4 变换矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = [tx, ty, tz] - init_trans  # 归一化平移量
            
            # 求逆矩阵
            inv_pose = np.linalg.inv(transform_matrix)

            # 存储数据
            self.poses.append(inv_pose)  # 存储的是转换到 SLAM 坐标系的位姿（已求逆）
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": transform_matrix.tolist(),  # 存储原始位姿矩阵
            }
            self.frames.append(frame)

class dl3dvParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"]
        
        # 读取 RGB 图像路径
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.n_img = len(self.color_paths)
        
        # 读取相机位姿 (camera.json)
        self.load_poses(os.path.join(self.input_folder, "cameras.json"))

    def load_poses(self, pose_file):
        """ 从 camera.json 读取相机位姿，并转换为 4×4 矩阵 """
        self.poses = []
        self.frames = []

        # 读取 JSON 数据
        with open(pose_file, "r") as f:
            all_poses = json.load(f)

        # 选取 [begin:end] 范围内的帧
        selected_poses = all_poses[self.begin:self.end]

        # 计算初始平移，用于对齐
        init_trans = np.array(selected_poses[0]["cam_trans"])

        for i, pose in enumerate(selected_poses):
            # 提取四元数和位移
            qx, qy, qz, qw = pose["cam_quat"]
            tx, ty, tz = pose["cam_trans"]

            # 转换为 3×3 旋转矩阵
            rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

            # 构造 4×4 变换矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = [tx, ty, tz] - init_trans  # 归一化平移量
            
            # 求逆矩阵
            inv_pose = np.linalg.inv(transform_matrix)

            # 存储数据
            self.poses.append(inv_pose)  # 存储的是转换到 SLAM 坐标系的位姿（已求逆）
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": transform_matrix.tolist(),  # 存储原始位姿矩阵
            }
            self.frames.append(frame)

## VKITTI数据解析器
class VKITTIParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"] 
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.jpg"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.jpg"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.jpg"))[self.begin:self.end]
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/gt/*.txt")

    def load_poses(self, path):
        self.poses = []
        self.frames = []
        pose_files = sorted(glob.glob(path))[self.begin:self.end]
        init_trans = np.loadtxt(pose_files[0], delimiter=' ').reshape(4, 4)[:3,3]

        for i in range(self.n_img):
            # print(pose_files[i])
            pose = np.loadtxt(pose_files[i], delimiter=' ').reshape(4, 4)
            pose[:3,3] = (pose[:3,3] - init_trans) / 10
            #self.poses.append(pose)
            inv_pose = np.linalg.inv(pose)  # 这里进行矩阵的求逆操作
            self.poses.append(inv_pose)     # self.pose是求了逆的
            #self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": pose.tolist(),      # 这个pose是没求逆的
            }
            self.frames.append(frame)

## KITTI数据解析器
class KITTIParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"] 
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/gt/*.txt")

    def load_poses(self, path):
        self.poses = []
        self.frames = []
        pose_files = sorted(glob.glob(path))[self.begin:self.end]
        init_trans = np.loadtxt(pose_files[0], delimiter=' ').reshape(4, 4)[:3,3]

        for i in range(self.n_img):
            # print(pose_files[i])
            pose = np.loadtxt(pose_files[i], delimiter=' ').reshape(4, 4)
            pose[:3,3] = pose[:3,3] - init_trans
            #self.poses.append(pose)
            inv_pose = np.linalg.inv(pose)  # 这里进行矩阵的求逆操作
            self.poses.append(inv_pose)     # self.pose是求了逆的
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": pose.tolist(),      # 这个pose是没求逆的
            }
            self.frames.append(frame)

## Waymo数据解析器
class WaymoParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/mono_depth/*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/gt/*.txt")

    def load_poses(self, path):
        self.poses = []
        self.frames = []
        pose_files = sorted(glob.glob(path))

        for i in range(self.n_img):
            # print(pose_files[i])
            pose = np.loadtxt(pose_files[i], delimiter=' ').reshape(4, 4)
            #self.poses.append(pose)
            inv_pose = np.linalg.inv(pose)  # 这里进行矩阵的求逆操作
            self.poses.append(inv_pose)     # self.pose是求了逆的
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "mono_depth_path": self.mono_depth_paths[i],
                "transform_matrix": pose.tolist(),      # 这个pose是没求逆的
            }
            self.frames.append(frame)

class CampusParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"]
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/images/*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        self.camera_param_paths = sorted(glob.glob(f"{self.input_folder}/cameras/*.json"))
        self.n_img = len(self.color_paths)
        self.poses = []
        self.frames = []
        self.load_poses()

    def load_poses(self):
        ts_to_pose = {}
        for cam_file in self.camera_param_paths:
            with open(cam_file, 'r') as f:
                meta = json.load(f)

            # 构造 4x4 位姿矩阵
            pose = np.eye(4, dtype=np.float32)
            pose[0, 0] = meta['t_00']
            pose[0, 1] = meta['t_01']
            pose[0, 2] = meta['t_02']
            pose[0, 3] = meta['t_03']
            pose[1, 0] = meta['t_10']
            pose[1, 1] = meta['t_11']
            pose[1, 2] = meta['t_12']
            pose[1, 3] = meta['t_13']
            pose[2, 0] = meta['t_20']
            pose[2, 1] = meta['t_21']
            pose[2, 2] = meta['t_22']
            pose[2, 3] = meta['t_23']

            R_convert = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
            ], dtype=np.float32)
            pose = R_convert @ pose

            inv_pose = np.linalg.inv(pose)
            ts_to_pose[str(meta['timestamp'])] = {
                'pose': pose,
                'inv_pose': inv_pose
            }

        for color_path in self.color_paths:
            ts = Path(color_path).stem
            depth_path = f"{self.input_folder}/depth/{ts}.png"
            if not os.path.exists(depth_path) or ts not in ts_to_pose:
                continue

            raw_pose = ts_to_pose[ts]['pose']
            inv_pose = ts_to_pose[ts]['inv_pose']

            self.poses.append(inv_pose)
            self.frames.append({
                "file_path": color_path,
                "depth_path": depth_path,
                "transform_matrix": raw_pose.tolist()
            })


# 解析Replica数据集
class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/results/mono*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "mono_depth_path": self.mono_depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames


# 解析TUM数据集
class TUMParser:
    def __init__(self, input_folder):   # 初始化输入文件夹路径，调用load_poses方法加载位姿信息。
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    # 关联图像、深度图像和位姿信息
    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):        # 遍历每个图像时间戳，匹配rgb图、深度图和pose的时间戳（如果本来都是对齐的这里就不需要了）
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    # 加载图像、深度图像和位姿数据，关联这些数据并创建帧信息。
    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")
        mono_depth_list = os.path.join(datapath, "mono_depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        mono_depth_data = self.parse_list(mono_depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)
        print("标号:", tstamp_image[471])

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames, self.mono_depth_paths = [], [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]
            self.mono_depth_paths += [os.path.join(datapath, mono_depth_data[i, 1])]

            quat = pose_vecs[k][4:]     # 旋转四元数
            trans = pose_vecs[k][1:4]   # 平移向量
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))  # 使用 trimesh 库中的 quaternion_matrix 函数将四元数转换为4x4矩阵
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]    # 计算矩阵 T 的逆矩阵，这里到底是从世界到相机还是从相机到世界？

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
                "mono_depth_path": str(os.path.join(datapath, mono_depth_data[i, 1]))
            }

            self.frames.append(frame)


class EuRoCParser:
    def __init__(self, input_folder, start_idx=0):
        self.input_folder = input_folder
        self.start_idx = start_idx
        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png")
        )
        self.color_paths_r = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam1/data/*.png")
        )
        assert len(self.color_paths) == len(self.color_paths_r)
        self.color_paths = self.color_paths[start_idx:]
        self.color_paths_r = self.color_paths_r[start_idx:]
        self.n_img = len(self.color_paths)
        self.load_poses(
            f"{self.input_folder}/mav0/state_groundtruth_estimate0/data.csv"
        )

    def associate(self, ts_pose):
        pose_indices = []
        for i in range(self.n_img):
            color_ts = float((self.color_paths[i].split("/")[-1]).split(".")[0])
            k = np.argmin(np.abs(ts_pose - color_ts))
            pose_indices.append(k)

        return pose_indices

    def load_poses(self, path):
        self.poses = []
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [list(map(float, row)) for row in reader]
        data = np.array(data)
        T_i_c0 = np.array(
            [
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        pose_ts = data[:, 0]
        pose_indices = self.associate(pose_ts)

        frames = []
        for i in range(self.n_img):
            trans = data[pose_indices[i], 1:4]
            quat = data[pose_indices[i], 4:8]
            quat = quat[[1, 2, 3, 0]]
            
            
            T_w_i = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T_w_i[:3, 3] = trans
            T_w_c = np.dot(T_w_i, T_i_c0)

            self.poses += [np.linalg.inv(T_w_c)]

            frame = {
                "file_path": self.color_paths[i],
                "transform_matrix": (np.linalg.inv(T_w_c)).tolist(),
            }

            frames.append(frame)
        self.frames = frames

##=================================================================定义通用数据基类=============================================================================

# 基类，定义基本的数据集接口。
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass

# 单目数据集的基类。
class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):     # 初始化相机内参和畸变参数
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]    # 相机的焦距（像素单位）
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]    # 主点（光学中心）
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )                              # 内参矩阵    
        # distortion parameters
        self.disorted = calibration["distorted"]   # 是否有畸变校正（waymo有）
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(   # 生成畸变校正的映射矩阵
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale  初始化场景信息，包括NeRF（神经辐射场）的归一化半径和位移
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def load_image(self, image_path):
        image = Image.open(image_path)
        image_array = np.array(image)

        # 判断图像是否是 RGB（三通道），如果是，提取第一通道
        if len(image_array.shape) == 3:  # 三通道 RGB 图像
            return image_array[:, :, 0]  # 仅返回第一通道（红色通道）
        else:  # 单通道（深度图）
            return image_array  # 返回深度图，已经是二维数组

    def __getitem__(self, idx):  #  获取指定索引的图像和位姿，处理图像畸变并转换为张量
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)  # 校正畸变

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = self.load_image(depth_path) / self.depth_scale    # 根据深度比例因子对深度图进行缩放
            mono_depth_path = self.mono_depth_paths[idx]
            mono_depth = self.load_image(mono_depth_path) / (self.depth_scale*5)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose, mono_depth

#  双目数据集的基类
class StereoDataset(BaseDataset):  
    def __init__(self, args, path, config):  # 初始化双目相机的内参和畸变参数
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        self.width = calibration["width"]
        self.height = calibration["height"]

        cam0raw = calibration["cam0"]["raw"]
        cam0opt = calibration["cam0"]["opt"]
        cam1raw = calibration["cam1"]["raw"]
        cam1opt = calibration["cam1"]["opt"]
        # Camera prameters
        self.fx_raw = cam0raw["fx"]
        self.fy_raw = cam0raw["fy"]
        self.cx_raw = cam0raw["cx"]
        self.cy_raw = cam0raw["cy"]
        self.fx = cam0opt["fx"]
        self.fy = cam0opt["fy"]
        self.cx = cam0opt["cx"]
        self.cy = cam0opt["cy"]

        self.fx_raw_r = cam1raw["fx"]
        self.fy_raw_r = cam1raw["fy"]
        self.cx_raw_r = cam1raw["cx"]
        self.cy_raw_r = cam1raw["cy"]
        self.fx_r = cam1opt["fx"]
        self.fy_r = cam1opt["fy"]
        self.cx_r = cam1opt["cx"]
        self.cy_r = cam1opt["cy"]

        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K_raw = np.array(
            [
                [self.fx_raw, 0.0, self.cx_raw],
                [0.0, self.fy_raw, self.cy_raw],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.Rmat = np.array(calibration["cam0"]["R"]["data"]).reshape(3, 3)
        self.K_raw_r = np.array(
            [
                [self.fx_raw_r, 0.0, self.cx_raw_r],
                [0.0, self.fy_raw_r, self.cy_raw_r],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K_r = np.array(
            [[self.fx_r, 0.0, self.cx_r], [0.0, self.fy_r, self.cy_r], [0.0, 0.0, 1.0]]
        )
        self.Rmat_r = np.array(calibration["cam1"]["R"]["data"]).reshape(3, 3)

        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [cam0raw["k1"], cam0raw["k2"], cam0raw["p1"], cam0raw["p2"], cam0raw["k3"]]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.dist_coeffs,
            self.Rmat,
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        self.dist_coeffs_r = np.array(
            [cam1raw["k1"], cam1raw["k2"], cam1raw["p1"], cam1raw["p2"], cam1raw["k3"]]
        )
        self.map1x_r, self.map1y_r = cv2.initUndistortRectifyMap(
            self.K_raw_r,
            self.dist_coeffs_r,
            self.Rmat_r,
            self.K_r,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

    def __getitem__(self, idx):   # 获取指定索引的左、右图像和位姿，计算视差图和深度图
        color_path = self.color_paths[idx]
        color_path_r = self.color_paths_r[idx]

        pose = self.poses[idx]
        image = cv2.imread(color_path, 0)
        image_r = cv2.imread(color_path_r, 0)
        depth = None
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
        stereo.setUniquenessRatio(40)
        disparity = stereo.compute(image, image_r) / 16.0
        disparity[disparity == 0] = 1e10
        depth = 47.90639384423901 / (
            disparity
        )  ## Following ORB-SLAM2 config, baseline*fx
        depth[depth < 0] = 0
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        return image, depth, pose

##=======================================================定义具体数据集的数据集类===========================================================================================
class dl3dvDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        
        # 使用 re10kParser 解析数据
        parser = dl3dvParser(dataset_path, config)

        # 赋值数据
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.color_paths  # 这里可能需要调整
        self.mono_depth_paths = parser.color_paths  # 这里可能需要调整
        self.poses = parser.poses  # 这里存储的是求逆的位姿

class CampusDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"]
        # 使用 re10kParser 解析数据
        parser = CampusParser(dataset_path, config)

        # 赋值数据
        #self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths[self.begin:self.end]
        self.depth_paths = parser.depth_paths[self.begin:self.end] # 这里可能需要调整
        self.mono_depth_paths = parser.mono_depth_paths[self.begin:self.end]  # 这里可能需要调整
        self.poses = parser.poses[self.begin:self.end]  # 这里存储的是求逆的位姿
        self.num_imgs = len(self.poses)
        print(self.num_imgs)

class re10kDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        
        # 使用 re10kParser 解析数据
        parser = re10kParser(dataset_path, config)

        # 赋值数据
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.color_paths  # 这里可能需要调整
        self.mono_depth_paths = parser.color_paths  # 这里可能需要调整
        self.poses = parser.poses  # 这里存储的是求逆的位姿

#
class TartanAirDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        
        # 使用 TartanAirParser 解析数据
        parser = TartanAirParser(dataset_path, config)

        # 赋值数据
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.color_paths  # 这里可能需要调整
        self.mono_depth_paths = parser.color_paths  # 这里可能需要调整
        self.poses = parser.poses  # 这里存储的是求逆的位姿
        
# VKITTI数据集
class VKITTIDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = VKITTIParser(dataset_path,config)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses   

# KITTI数据集
class KITTIDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = KITTIParser(dataset_path,config)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses      


# Waymo数据集类
class WaymoDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = WaymoParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses       # 这是求了逆之后的pose


# TUM数据集类，使用TUMParser解析数据
class TUMDataset(MonocularDataset):   # 初始化路径和配置，使用TUMParser解析数据
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses
        self.mono_depth_paths = parser.mono_depth_paths


class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses


class EurocDataset(StereoDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = EuRoCParser(dataset_path, start_idx=config["Dataset"]["start_idx"])
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.color_paths_r = parser.color_paths_r
        self.poses = parser.poses


class RealsenseDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        self.pipeline = rs.pipeline()
        self.h, self.w = 720, 1280
        
        self.depth_scale = 0
        if self.config["Dataset"]["sensor_type"] == "depth":
            self.has_depth = True 
        else: 
            self.has_depth = False

        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)
        if self.has_depth:
            self.rs_config.enable_stream(rs.stream.depth)

        self.profile = self.pipeline.start(self.rs_config)

        if self.has_depth:
            self.align_to = rs.stream.color
            self.align = rs.align(self.align_to)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # rgb_sensor.set_option(rs.option.enable_auto_white_balance, True)
        self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.rgb_intrinsics = self.rgb_profile.get_intrinsics()
        
        self.fx = self.rgb_intrinsics.fx
        self.fy = self.rgb_intrinsics.fy
        self.cx = self.rgb_intrinsics.ppx
        self.cy = self.rgb_intrinsics.ppy
        self.width = self.rgb_intrinsics.width
        self.height = self.rgb_intrinsics.height
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.disorted = True
        self.dist_coeffs = np.asarray(self.rgb_intrinsics.coeffs)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, np.eye(3), self.K, (self.w, self.h), cv2.CV_32FC1
        )

        if self.has_depth:
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale  = self.depth_sensor.get_depth_scale()
            self.depth_profile = rs.video_stream_profile(
                self.profile.get_stream(rs.stream.depth)
            )
            self.depth_intrinsics = self.depth_profile.get_intrinsics()
        
        


    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)
        depth = None

        frameset = self.pipeline.wait_for_frames()

        if self.has_depth:
            aligned_frames = self.align.process(frameset)
            rgb_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth = np.array(aligned_depth_frame.get_data())*self.depth_scale
            depth[depth < 0] = 0
            np.nan_to_num(depth, nan=1000)
        else:
            rgb_frame = frameset.get_color_frame()

        image = np.asanyarray(rgb_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        return image, depth, pose


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "euroc":
        return EurocDataset(args, path, config)
    elif config["Dataset"]["type"] == "realsense":
        return RealsenseDataset(args, path, config)
    elif config["Dataset"]["type"] == "waymo":
        return WaymoDataset(args, path, config)
    elif config["Dataset"]["type"] == "KITTI":
        return KITTIDataset(args, path, config)
    elif config["Dataset"]["type"] == "TartanAir":
        return TartanAirDataset(args, path, config)
    elif config["Dataset"]["type"] == "VKITTI":
        return VKITTIDataset(args, path, config)
    elif config["Dataset"]["type"] == "re10k":
        return re10kDataset(args, path, config)
    elif config["Dataset"]["type"] == "dl3dv":
        return dl3dvDataset(args, path, config)
    elif config["Dataset"]["type"] == "campus":
        return CampusDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
