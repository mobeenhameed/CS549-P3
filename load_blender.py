import os
import json
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset


def load_blender_data(basedir, split, half_res=False):
    """
    Load NeRF blender synthetic dataset for a given split.

    Args:
        basedir (str): Path to the scene directory (e.g. .../nerf_synthetic/lego)
        split (str): One of 'train', 'val', 'test'
        half_res (bool): If True, downsample images to 400x400

    Returns:
        imgs   (np.ndarray): [N, H, W, 4] float32 RGBA images in [0, 1]
        poses  (np.ndarray): [N, 4, 4] float32 camera-to-world matrices
        hwf    (list): [H, W, focal]  — image height, width, focal length in pixels
    """
    json_path = os.path.join(basedir, f"transforms_{split}.json")
    with open(json_path, "r") as f:
        meta = json.load(f)

    imgs, poses = [], []

    for frame in meta["frames"]:
        # File path stored without extension in the JSON
        img_path = os.path.join(basedir, frame["file_path"] + ".png")
        img = imageio.imread(img_path)                    # (H, W, 4)  uint8
        imgs.append(img)
        poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

    imgs = (np.stack(imgs, axis=0) / 255.0).astype(np.float32)   # [N, H, W, 4]
    poses = np.stack(poses, axis=0)                               # [N, 4, 4]

    H, W = imgs.shape[1], imgs.shape[2]
    camera_angle_x = float(meta["camera_angle_x"])
    # Focal length derived from horizontal FOV
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    if half_res:
        import cv2
        H_half, W_half = H // 2, W // 2
        imgs_half = np.stack(
            [cv2.resize(imgs[i], (W_half, H_half), interpolation=cv2.INTER_AREA)
             for i in range(imgs.shape[0])],
            axis=0
        )
        imgs = imgs_half
        H, W = H_half, W_half
        focal = focal / 2.0

    hwf = [int(H), int(W), float(focal)]
    return imgs, poses, hwf


class BlenderDataset(Dataset):
    """
    PyTorch Dataset that returns individual (image, pose) pairs.

    Each __getitem__ returns:
        img   (torch.Tensor): [H, W, 4]  float32 RGBA in [0, 1]
        pose  (torch.Tensor): [4, 4]     float32 camera-to-world matrix
    """

    def __init__(self, basedir, split, half_res=False):
        super().__init__()
        imgs, poses, hwf = load_blender_data(basedir, split, half_res=half_res)
        self.imgs = torch.from_numpy(imgs)    # [N, H, W, 4]
        self.poses = torch.from_numpy(poses)  # [N, 4, 4]
        self.H, self.W, self.focal = hwf
        self.n_images = self.imgs.shape[0]

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        return self.imgs[idx], self.poses[idx]
