# Modified by 2024 Allen Ren, Princeton University
# Copyright (c) 2018 Andy Zeng
# Source: https://github.com/andyzeng/tsdf-fusion-python/blob/master/fusion.py
# BSD 2-Clause License
# Copyright (c) 2019, Princeton University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from numba import njit, prange
from typing import List

from .geom import points_in_circle, rigid_transform


class TSDFPlannerBase:
    """Volumetric TSDF Fusion of RGB-D Images. No GPU mode.

    Add frontier-based exploration and semantic map.
    """

    def __init__(
        self,
        vol_bnds,
        voxel_size,
        floor_height,
        floor_height_offset=0,
        pts_init=None,
        init_clearance=0,
        save_visualization=False,
    ):
        """Constructor.
        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."
        assert (vol_bnds[:, 0] < vol_bnds[:, 1]).all()

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = (
            np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size)
            .copy(order="C")
            .astype(int)
        )
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order="C").astype(np.float32)

        # Initialize pointers to voxel volume in CPU memory
        # Assume all unobserved regions are occupied
        self._tsdf_vol_cpu = -np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        self.save_visualization = save_visualization
        if save_visualization:
            # Initialize obstacle volume
            self._obstacle_vol_cpu = np.zeros(self._vol_dim).astype(bool)
        else:
            self._obstacle_vol_cpu = None

        # 3D体素数组, 已探索:非零, 未探索:0
        # Explored or not
        self._explore_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            range(self._vol_dim[0]),
            range(self._vol_dim[1]),
            range(self._vol_dim[2]),
            indexing="ij",
        )
        self.vox_coords = (
            np.concatenate(
                [xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0
            )
            .astype(int)
            .T
        )

        # pre-compute
        self.cam_pts_pre = TSDFPlannerBase.vox2world(
            self._vol_origin, self.vox_coords, self._voxel_size
        )

        self.floor_height = floor_height

        # 是0
        # Find the minimum height voxel
        self.min_height_voxel = int(floor_height_offset / self._voxel_size)

        # 初始位置及其邻域的一些点
        # For masking the area around initial pose to be unoccupied
        coords_init = self.normal2voxel(pts_init)
        self.init_points = points_in_circle(
            coords_init[0],
            coords_init[1],
            int(init_clearance / self._voxel_size),
            self._vol_dim[:2],
        )

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        """Convert voxel grid coordinates to world coordinates."""
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        """Convert camera coordinates to pixel coordinates."""
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    def voxel_size(self):
        return self._voxel_size
    
    def vol_origin(self):
        return self._vol_origin

    def pix2cam(self, pix, intr):
        """Convert pixel coordinates to camera coordinates."""
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        cam_pts = np.empty((pix.shape[0], 3), dtype=np.float32)
        for i in range(cam_pts.shape[0]):
            cam_pts[i, 2] = 1
            cam_pts[i, 0] = (pix[i, 0] - cx) / fx * cam_pts[i, 2]
            cam_pts[i, 1] = (pix[i, 1] - cy) / fy * cam_pts[i, 2]
        return cam_pts

    def normal2voxel(self, pts):
        '''
        给定一个点的坐标, normalize称voxel坐标系下的坐标
        即先算得相对voxel坐标系原点坐标, 再用voxel大小归一化坐标
        '''
        pts = pts - self._vol_origin
        coords = np.round(pts / self._voxel_size).astype(int)
        coords = np.clip(coords, 0, self._vol_dim - 1)
        return coords

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume."""
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate(
        self,
        color_im,
        depth_im,
        cam_intr,
        cam_pose,
        sem_im=None,
        w_new=None,
        obs_weight=1.0,
        margin_h=240,  # from top
        margin_w=120,  # each side
        explored_depth=1.5,
    ):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          sem_im (ndarray): An semantic image of shape (H, W).
          obs_weight (float): The weight to assign for the current observation. A higher
            value
          margin_h (int): The margin from the top of the image to exclude when integrating explored
          margin_w (int): The margin from the sides of the image to exclude when integrating explored
        """
        im_h, im_w = depth_im.shape
        max_dist = 3.0

        # Convert voxel grid coordinates to pixel coordinates
        cam_pts = rigid_transform(self.cam_pts_pre, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = TSDFPlannerBase.cam2pix(cam_pts, cam_intr)  
        pix_x, pix_y = pix[:, 0], pix[:, 1]

        cur_pose_xy = cam_pose[:2, 3]

        # Eliminate pixels outside view frustum
        valid_pix = (
            (pix_x >= 0)
            & (pix_x < im_w)
            & (pix_y >= 0)
            & (pix_y < im_h)
            & (pix_z > 0)
            & (pix_z < max_dist)
        )
        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        # narrow view
        depth_val_narrow = np.zeros(pix_x.shape)
        depth_val_narrow[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]
        # depth_val_narrow[depth_val_narrow >= explored_depth] = 0.0
        depth_val_narrow[
            np.linalg.norm(self.cam_pts_pre[:, :2] - cur_pose_xy, axis=1)
            >= explored_depth
        ] = 0.0

        # Integrate TSDF
        depth_diff = depth_val - pix_z
        depth_margin = 1 * self._voxel_size
        valid_pts = np.logical_and(depth_val > 0, depth_diff > depth_margin)
        dist = np.where(depth_diff > depth_margin, 1.0, -1.0)
        valid_vox_x = self.vox_coords[valid_pts, 0]
        valid_vox_y = self.vox_coords[valid_pts, 1]
        valid_vox_z = self.vox_coords[valid_pts, 2]
        w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]

        if self.save_visualization:
            # Mark obstacle
            obstacle_pts = (
                (depth_diff < 0)
                & (depth_diff >= -0.2)
                & (pix_x >= 0)
                & (pix_x < im_w)
                & (pix_y >= 0)
                & (pix_y < im_h)
                & (pix_z > 0)
                & (pix_z < max_dist)
            )
            obstacle_vox_x = self.vox_coords[obstacle_pts, 0]
            obstacle_vox_y = self.vox_coords[obstacle_pts, 1]
            obstacle_vox_z = self.vox_coords[obstacle_pts, 2]
            self._obstacle_vol_cpu[obstacle_vox_x, obstacle_vox_y, obstacle_vox_z] = (
                True
            )

        depth_diff_narrow = depth_val_narrow - pix_z
        valid_pts_narrow = np.logical_and(
            depth_val_narrow > 0, depth_diff_narrow > depth_margin
        )
        valid_vox_x_narrow = self.vox_coords[valid_pts_narrow, 0]
        valid_vox_y_narrow = self.vox_coords[valid_pts_narrow, 1]
        valid_vox_z_narrow = self.vox_coords[valid_pts_narrow, 2]
        if w_new is None:
            tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            tsdf_vol_new, w_new = TSDFPlannerBase.integrate_tsdf(
                tsdf_vals, valid_dist, w_old, obs_weight
            )
            self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

            # Mark explored
            self._explore_vol_cpu[
                valid_vox_x_narrow, valid_vox_y_narrow, valid_vox_z_narrow
            ] = 1

        return w_new

    def get_volume(self):
        return self._tsdf_vol_cpu, self._color_vol_cpu

    def check_within_bnds(self, pts, slack=0):
        '''
        检查是否在2D边界内, 在边界内则返回True, 否则False
        '''
        return not (
            pts[0] <= slack
            or pts[0] >= self._vol_dim[0] - slack
            or pts[1] <= slack
            or pts[1] >= self._vol_dim[1] - slack
        )

    def clip_2d_array(self, array):
        return array[
            (array[:, 0] >= 0)
            & (array[:, 0] < self._vol_dim[0])
            & (array[:, 1] >= 0)
            & (array[:, 1] < self._vol_dim[1])
        ]


    @staticmethod
    def update_path_points(path_points: List[np.ndarray], point: np.ndarray):
        '''
        根据一个新位置 point, 更新路径 path_points, 从离该点最近的路径段开始继续走
        即 把机器人"投影"到路径上, 然后返回剩余的路径
        '''
        # get the closest line segment
        dist = np.inf
        min_dist_idx = -1
        for i in range(len(path_points) - 1):
            p1, p2 = path_points[i], path_points[i + 1]
            seg = p2 - p1
            # if the point is between the two points
            if np.dot(point - p1, seg) * np.dot(point - p2, seg) <= 0:
                d = np.abs(np.cross(seg, point - p1) / np.linalg.norm(seg))
            # else, get the distance to the closest endpoint
            else:
                d = min(np.linalg.norm(point - p1), np.linalg.norm(point - p2))
            if d < dist + 1e-6:
                dist = d
                min_dist_idx = i

        updated_path_points = path_points.copy()
        updated_path_points = updated_path_points[min_dist_idx:]

        # cut the line if point is between the two endpoints of the nearest segment
        p1, p2 = updated_path_points[0], updated_path_points[1]
        seg = p2 - p1
        if np.dot(point - p1, seg) * np.dot(point - p2, seg) <= 0:
            # find the point on segment that is closest to the point
            t = np.dot(point - p1, seg) / np.dot(seg, seg)
            closest_point = p1 + t * seg
            updated_path_points[0] = closest_point

        return updated_path_points

    @staticmethod
    def rad2vector(angle):
        return np.array([-np.sin(angle), np.cos(angle)])

    def get_obstacle_map(self, height):
        assert self._obstacle_vol_cpu is not None
        height_voxel = int(height / self._voxel_size) + self.min_height_voxel
        return self._obstacle_vol_cpu[:, :, height_voxel]

    def voxel2normal(self, pts):
        assert len(pts) == 2, f"Expected 2D coordinate in voxel space, got {pts}"
        return pts * self._voxel_size + self._vol_origin[:2]