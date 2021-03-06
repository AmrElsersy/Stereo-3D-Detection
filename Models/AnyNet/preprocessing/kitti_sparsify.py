import argparse
import os

import numpy as np
import torch
import tqdm


def pto_rec_map(velo_points, H=64, W=512, D=800):
    # depth, width, height
    valid_inds = (velo_points[:, 0] < 80) & \
                 (velo_points[:, 0] >= 0) & \
                 (velo_points[:, 1] < 50) & \
                 (velo_points[:, 1] >= -50) & \
                 (velo_points[:, 2] < 1) & \
                 (velo_points[:, 2] >= -2.5)
    velo_points = velo_points[valid_inds]

    x, y, z = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2]
    x_grid = (x * D / 80.).astype(int)
    x_grid[x_grid < 0] = 0
    x_grid[x_grid >= D] = D - 1

    y_grid = ((y + 50) * W / 100.).astype(int)
    y_grid[y_grid < 0] = 0
    y_grid[y_grid >= W] = W - 1

    z_grid = ((z + 2.5) * H / 3.5).astype(int)
    z_grid[z_grid < 0] = 0
    z_grid[z_grid >= H] = H - 1

    depth_map = - np.ones((D, W, H, 3))
    depth_map[x_grid, y_grid, z_grid, 0] = x
    depth_map[x_grid, y_grid, z_grid, 1] = y
    depth_map[x_grid, y_grid, z_grid, 2] = z
    depth_map = depth_map.reshape((-1, 3))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map

def pto_ang_map(pc_velo, H=64, W=512, slice=1):
        """
        :param H: the row num of depth map, could be 64(default), 32, 16
        :param W: the col num of depth map
        :param slice: output every slice lines
        """
        valid_inds =    torch.where((pc_velo[:, 0] < 120)    & \
                        (pc_velo[:, 0] >= 0)     & \
                        (pc_velo[:, 1] < 50)     & \
                        (pc_velo[:, 1] >= -50)   & \
                        (pc_velo[:, 2] < 1.5)    & \
                        (pc_velo[:, 2] >= -2.5))
        
        pc_velo = pc_velo[valid_inds]

        def radians(x):
            return x * 0.0174532925

        dtheta = radians(0.4 * 64.0 / H)
        dphi = radians(90.0 / W)

        x, y, z = pc_velo[:, 0], pc_velo[:, 1], pc_velo[:, 2]

        x_y = x**2 + y**2

        d = torch.sqrt(x_y + z ** 2)
        r = torch.sqrt(x_y) 
        d = d.clamp(0.000001)
        r = r.clamp(0.000001)
        phi = radians(45.) - torch.arcsin(y / r)
        phi_ = (phi / dphi).long()
        phi_ = phi_.clamp(min=0, max=W - 1) 

        theta = radians(2.) - torch.arcsin(z / d)
        theta_ = (theta / dtheta).long()
        theta_ = theta_.clamp(min=0, max= H - 1)

        depth_map = - torch.ones((H, W, 3), device= pc_velo.device)
        depth_map[theta_, phi_, 0] = x
        depth_map[theta_, phi_, 1] = y
        depth_map[theta_, phi_, 2] = z
        depth_map = depth_map[0::slice, :, :]
        depth_map = depth_map.reshape((-1, 3))
        depth_map = depth_map[depth_map[:, 0] != -1.0]
        return depth_map

def gen_sparse_points(pl_data_path, args):
    pc_velo = np.fromfile(pl_data_path, dtype=np.float32).reshape((-1, 4))
    pc_velo = torch.tensor(pc_velo)
    # depth, width, height
    valid_inds = (pc_velo[:, 0] < 120) & \
                 (pc_velo[:, 0] >= 0) & \
                 (pc_velo[:, 1] < 50) & \
                 (pc_velo[:, 1] >= -50) & \
                 (pc_velo[:, 2] < 1.5) & \
                 (pc_velo[:, 2] >= -2.5)
    pc_velo = pc_velo[valid_inds]

    return pto_ang_map(pc_velo, H=args.H, W=args.W, slice=args.slice)


def gen_sparse_points_all(args):
    outputfolder = args.sparse_pl_path
    os.makedirs(outputfolder, exist_ok=True)
    data_idx_list = sorted([x.strip() for x in os.listdir(args.pl_path) if x[-3:] == 'bin'])

    for data_idx in tqdm.tqdm(data_idx_list):
        sparse_points = gen_sparse_points(os.path.join(args.pl_path, data_idx), args).numpy()
        sparse_points = sparse_points.astype(np.float32)
        sparse_points.tofile(f'{outputfolder}/{data_idx}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate sparse pseudo-LiDAR points")
    parser.add_argument('--pl_path', default='/scratch/datasets', help='pseudo-lidar path')
    parser.add_argument('--sparse_pl_path', default='/scratch/datasets', help='sparsed pseudo lidar path')
    parser.add_argument('--slice', default=1, type=int)
    parser.add_argument('--H', default=64, type=int)
    parser.add_argument('--W', default=512, type=int)
    parser.add_argument('--D', default=700, type=int)
    args = parser.parse_args()

    gen_sparse_points_all(args)
