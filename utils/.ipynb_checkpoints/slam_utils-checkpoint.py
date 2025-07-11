import torch
import torch.nn.functional as F

# 使用 Scharr 滤波器计算图像梯度
def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]

# 计算图像的梯度mask，用于过滤掉无效区域(3×3区域内有=0的像素)
def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)

# 计算深度图的正则化损失，这里用了一些基于梯度的权重
def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err

# 计算tracking的L1损失，根据不同的输入类型；注意计算时会利用mask，用于过滤无效区域和加权重要区域，以提高跟踪的精度！！
def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    
    if config["Training"]["monocular"] and config["Dataset"]["depth_loss"]:
        #return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)
# 根据阈值 rgb_boundary_threshold 和梯度掩码 viewpoint.grad_mask 生成的像素掩码，用于过滤图像中的无效区域，只关注边缘和重要区域。
# 用opacity来给L1损失加权，重点关注更可靠的像素
def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    #opacity_mask = (opacity > 0.95).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    #l1 = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()

# 深度loss：过滤掉无效深度值（＞0.01），只保留不透明度大于0.95的像素，确保只对高置信度区域进行优化。
def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.mono_depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    #print("loss为: ", f"光度损失: {l1_rgb.mean()}", f"深度损失: {l1_depth.mean()}")
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()

# 计算mapping的L1损失，注意会用到mask，用于过滤无效区域和加权重要区域，以提高地图构建的精度！
def get_loss_mapping(config, image,  viewpoint, depth=None, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
        
    if config["Training"]["monocular"] and config["Dataset"]["depth_loss"]:
        return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)

# 根据阈值 rgb_boundary_threshold 生成mask，用于过滤无效区域，只关注重要区域。
def get_loss_mapping_rgb(config, image, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()

# 过滤掉无效的深度值，只保留有效的（＞0.01）；同上只保留重要rgb区域
def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.mono_depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
    #print("loss为: ", f"光度损失: {l1_rgb.mean()}", f"深度损失: {l1_depth.mean()}")
    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

# 计算深度图的有效深度（有mask）的中值，以及可选的标准差
# mask：深度值＞0，不透明度＞0.95，额外输入的mask
def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

##===============================================rgb-only的patch-wise正则化深度监督===========================================
# tracking loss
def get_loss_tracking_norm(config, image, depth, opacity, viewpoint):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    mono_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (mono_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth_global = patch_norm_l1_loss_global(depth[None,...], mono_depth[None,...], patch_size=16, margin=0.01)
    l1_depth_local = patch_norm_l1_loss(depth[None,...], mono_depth[None,...], patch_size=16, margin=0.01)
    #l1_depth_local = 0
    #print("loss为: ", f"光度损失: {l1_rgb}", f"深度损失: {l1_depth_local}")
    #return alpha * l1_rgb + (1 - alpha) * (0.1 * l1_depth_global + 1 * l1_depth_local)
    return alpha * l1_rgb + (1 - alpha) * (1 * l1_depth_local)

# mapping loss
def get_loss_mapping_norm(config, image, depth, viewpoint):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    
    gt_image = viewpoint.original_image.cuda()
    mono_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (mono_depth > 0.01).view(*depth.shape)
    
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask).mean()
    l1_depth_global = patch_norm_l1_loss_global(depth[None,...], mono_depth[None,...], patch_size=16, margin=0.01)
    l1_depth_local = patch_norm_l1_loss(depth[None,...], mono_depth[None,...], patch_size=16, margin=0.01)
    #l1_depth_local = 0
    
    #return alpha * l1_rgb + (1 - alpha) * (0.1 * l1_depth_global + 1 * l1_depth_local)
    return alpha * l1_rgb + (1 - alpha) * (1 * l1_depth_local)

# 为深度图做normalization
def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

# 计算L1 Loss
def margin_l1_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask].abs()).mean()
    else:
        return ((network_output - gt)[mask].abs()).mean(), mask

# 将输入图像张量按指定kernel_size划分为不重叠的patch    
def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

# Global normalization
def patch_norm_l1_loss_global(input, target, patch_size, margin=0.01, return_mask=False):
    input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
    return margin_l1_loss(input_patches, target_patches, margin, return_mask)

# Local normalization
def patch_norm_l1_loss(input, target, patch_size, margin=0.01, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l1_loss(input_patches, target_patches, margin, return_mask)