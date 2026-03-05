import torch


def get_rays(H, W, focal, c2w):
    """
    Generate one ray per pixel for a single image.

    The dataset uses the OpenGL / Blender camera convention:
        - camera looks along the  -Z axis
        - Y axis points  up
        - X axis points  right

    Args:
        H     (int):            image height in pixels
        W     (int):            image width  in pixels
        focal (float):          focal length in pixels
        c2w   (torch.Tensor):   [4, 4] camera-to-world transform

    Returns:
        rays_o (torch.Tensor): [H, W, 3]  ray origins    (world space)
        rays_d (torch.Tensor): [H, W, 3]  ray directions (world space, unit vectors)
    """
    # Pixel grid  (i = row index, j = column index)
    j, i = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=c2w.device),
        torch.arange(H, dtype=torch.float32, device=c2w.device),
        indexing="xy",
    )

    # Direction of each pixel ray in camera space
    #   x : right  →  (j - cx) / focal
    #   y : up     →  -(i - cy) / focal   (flip: image row grows downward)
    #   z : into scene along -Z convention →  -1
    dirs = torch.stack(
        [
            (j - W * 0.5) / focal,
            -(i - H * 0.5) / focal,
            -torch.ones_like(j),
        ],
        dim=-1,
    )  # [H, W, 3]

    # Rotate camera-space directions into world space using the 3×3 rotation block
    # rays_d[h,w] = c2w[:3,:3] @ dirs[h,w]
    rays_d = (dirs[..., None, :] @ c2w[:3, :3].T).squeeze(-2)  # [H, W, 3]

    # Normalise so each direction is a unit vector
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Ray origin: the camera centre in world space (translation column of c2w)
    rays_o = c2w[:3, 3].expand_as(rays_d)  # [H, W, 3]

    return rays_o, rays_d


def get_rays_batch(H, W, focal, poses):
    """
    Generate rays for a batch of camera poses.

    Args:
        H      (int):            image height
        W      (int):            image width
        focal  (float):          focal length in pixels
        poses  (torch.Tensor):   [N, 4, 4] camera-to-world matrices

    Returns:
        rays_o (torch.Tensor): [N, H, W, 3]
        rays_d (torch.Tensor): [N, H, W, 3]
    """
    all_o, all_d = [], []
    for i in range(poses.shape[0]):
        o, d = get_rays(H, W, focal, poses[i])
        all_o.append(o)
        all_d.append(d)
    return torch.stack(all_o), torch.stack(all_d)


def sample_points_along_rays(rays_o, rays_d, near, far, n_samples, perturb=True):
    """
    Stratified sampling of 3-D points along each ray.

    For each ray we partition [near, far] into n_samples equal bins and draw
    one sample uniformly inside each bin (when perturb=True), or use bin
    centres (when perturb=False / at inference time).

    Args:
        rays_o   (torch.Tensor): [..., 3]  ray origins
        rays_d   (torch.Tensor): [..., 3]  ray directions (unit)
        near     (float):        near bound of the sampling interval
        far      (float):        far  bound of the sampling interval
        n_samples (int):         number of samples per ray
        perturb  (bool):         add uniform noise inside each bin

    Returns:
        pts (torch.Tensor): [..., n_samples, 3]  3-D sample positions
        z   (torch.Tensor): [..., n_samples]     depth values (t values)
    """
    batch_shape = rays_o.shape[:-1]  # e.g. (H, W) or (N,)

    # Bin edges at evenly spaced t values
    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=rays_o.device)
    z = near * (1.0 - t_vals) + far * t_vals  # [n_samples]

    if perturb:
        # Width of each bin
        mid = 0.5 * (z[:-1] + z[1:])
        upper = torch.cat([mid, z[-1:]])
        lower = torch.cat([z[:1], mid])
        noise = torch.rand(*batch_shape, n_samples, device=rays_o.device)
        z = lower + (upper - lower) * noise  # [..., n_samples]
    else:
        z = z.expand(*batch_shape, n_samples)

    # pts = o + t * d
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z[..., None]
    # pts: [..., n_samples, 3]

    return pts, z
