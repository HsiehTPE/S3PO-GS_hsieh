import os
import argparse
import glob
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl
from pathlib import Path
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from plyfile import PlyData, PlyElement

def save_colored_pointcloud(points, colors, filename):
    verts = np.array([
        (point[0], point[1], point[2], int(color[0]), int(color[1]), int(color[2]))
        for point, color in zip(points, colors)
    ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    ply = PlyData([PlyElement.describe(verts, 'vertex')], text=True)
    os.makedirs("test", exist_ok=True)
    ply.write(f"test/{filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='datasets/waymo/405841', help='Path to folder with images')
    parser.add_argument('--x', type=int, required=True, help='Index of first image')
    parser.add_argument('--y', type=int, required=True, help='Index of second image')
    args = parser.parse_args()

    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    # Load all images from the folder
    #image_files = sorted(glob.glob(os.path.join(args.folder, '*.jpg')))
    image_files = sorted(glob.glob(os.path.join(args.folder, '*.png')))
    img1_path, img2_path = image_files[args.x], image_files[args.y]

    images = load_images([img1_path, img2_path], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    x_min, x_max = 0, 1000
    y_min, y_max = 0, 1000

    mask_im0 = (matches_im0[:, 0] >= x_min) & (matches_im0[:, 0] <= x_max) & \
               (matches_im0[:, 1] >= y_min) & (matches_im0[:, 1] <= y_max)

    matches_im0, matches_im1 = matches_im0[mask_im0], matches_im1[mask_im0]

    # Save point cloud (from pred1['pts_3d'] and view1['img'])
    pts3d = pred1['pts3d'].squeeze(0).cpu().numpy()  # (H, W, 3)

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    rgb_tensor = view1['img'] * image_std + image_mean
    colors = rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255  # (H, W, 3)

    points = pts3d.reshape(-1, 3)
    colors = colors.reshape(-1, 3).astype(np.uint8)
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    colors = colors[mask]
    save_colored_pointcloud(points, colors, f"pts_{args.x}_{args.y}.ply")

    # visualization of matching pair
    n_viz = 20
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img_combined = np.concatenate((img0, img1), axis=1)

    pl.figure(figsize=(10, 5))
    pl.imshow(img_combined)
    pl.axis('off')

    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)

    os.makedirs("test", exist_ok=True)
    pl.savefig(f"test/match_{args.x}_{args.y}.png", bbox_inches='tight', pad_inches=0)
    pl.close()

