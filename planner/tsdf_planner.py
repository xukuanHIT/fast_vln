import os.path

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.cluster import DBSCAN, KMeans
from scipy import stats
import torch
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field
import scipy.ndimage as ndimage

from planner.geom import *
from .tsdf_base import TSDFPlannerBase
from map.map_elements import Frame, Keyframe, Object3D
from map.map import Map

@dataclass
class Frontier:
    """
    Frontier class for frontier-based exploration.
        position: 表示该frontier中心点, 会被调整为距离障碍物有一定距离
        orientation: frontier中心的方向, 没有被调整
        region:  根据frontier点坐标生成一个2D frontier地图
    """

    position: np.ndarray  # integer position in voxel grid
    orientation: np.ndarray  # directional vector of the frontier in float
    region: (
        np.ndarray
    )  # boolean array of the same shape as the voxel grid, indicating the region of the frontier
    frontier_id: (
        int  # unique id for the frontier to identify its region on the frontier map
    )
    image: str = None
    target_detected: bool = (
        False  # whether the target object is detected in the snapshot, only used when generating data
    )
    feature: torch.Tensor = (
        None  # the image feature of the snapshot, not used when generating data
    )

    frame: Frame = None
    view_angle_diff: float = np.pi

    def __eq__(self, other):
        if not isinstance(other, Frontier):
            raise TypeError("Cannot compare Frontier with non-Frontier object.")
        return np.array_equal(self.region, other.region)


class TSDFPlanner(TSDFPlannerBase):
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
        occupancy_height=0.4,
        vision_height=1.2,
        save_visualization=False,
    ):
        super().__init__(
            vol_bnds,
            voxel_size,
            floor_height,
            floor_height_offset,
            pts_init,
            init_clearance,
            save_visualization,
        )

        self.occupancy_height = (
            occupancy_height  # the occupied/navigable map is acquired at this height
        )
        self.vision_height = (
            vision_height  # the visibility map is acquired at this height
        )



        self.frontiers: List[Frontier] = []

        # about frontier allocation
        self.frontier_map = np.zeros(self._vol_dim[:2], dtype=int)
        self.frontier_counter = 1

        # about storing occupancy information on each step
        self.unexplored = None
        self.unoccupied = None
        self.occupied = None
        self.island = None
        self.unexplored_neighbors = None
        self.occupied_map_camera = None



    def plot_2d_map(self, occupied, frontier_areas, frontier_edge_areas, explored):
        h, w = occupied.shape
        # 初始化背景图 (BGR, OpenCV 格式)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (255, 255, 0)   # 青色 (cyan)

        # 颜色定义 (BGR)
        color_occupied = (50, 50, 50)      # 深灰
        color_explored = (255, 255, 255)   # 白色
        color_frontier = (180, 180, 180)   # 浅灰
        color_frontier_edge = (0, 0, 255)  # 红色

        # occupied 区域
        img[occupied > 0] = color_occupied

        # explored 区域
        img[explored > 0] = color_explored

        # frontier_areas (argwhere -> [y, x])
        for y, x in frontier_areas:
            img[y, x] = color_frontier

        # frontier_edge_areas
        for y, x in frontier_edge_areas:
            img[y, x] = color_frontier_edge

        return img



    def closest_angle(self, q, angle_list):
        diff = np.arctan2(np.sin(angle_list - q), np.cos(angle_list - q))
        diff_abs = np.abs(diff)
        idx = np.argmin(diff_abs)
        return diff_abs[idx], idx


    def set_explored_area(self, center, radisu):
        unoccupied_coords = np.argwhere(self.unoccupied)
        dists_unoccupied = np.linalg.norm(unoccupied_coords - center, axis=1)
        near_coords = unoccupied_coords[
            dists_unoccupied < radisu / self.voxel_size()
        ]
        self._explore_vol_cpu[near_coords[:, 0], near_coords[:, 1], :] = 1


    def update_frontier_map(
        self,
        pts,
        cfg,
        scene: Map,
        cnt_step: int,
        save_frontier_image: bool = False,
        eps_frontier_dir=None,
        prompt_img_size: Tuple[int, int] = (320, 320),
        visualization: bool = False
    ) -> bool:
        # 给定一个点的坐标, normalize称voxel坐标系下的坐标
        # 即先算得相对voxel坐标系原点坐标, 再用voxel大小归一化坐标
        cur_point = self.normal2voxel(pts)

        # 返回与当前位置最近的孤岛区域和2D占据地图, 在导航高度
        island, unoccupied = self.get_island_around_pts(
            pts, height=self.occupancy_height
        )

        # 被占据区域(也包含位置区域)
        occupied = np.logical_not(unoccupied).astype(int) 

        # 根据3D exploration地图获取2D unexplored区域, 通过将高度维度相加. 
        # 即高度上任一voxel被探索过了, 都表示2D的这个grid被探索过了
        unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)

        # 机器人初始位置及其邻域设置成已被探索
        for point in self.init_points:
            unexplored[point[0], point[1]] = 0

        # unexplored_neighbors存储的是某位置8邻域中有多少是未被探索的
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        unexplored_neighbors = ndimage.convolve(
            unexplored, kernel, mode="constant", cval=0.0
        )

        # vision_height高度, 先取和当前位置最近的孤岛, 再取反表示除孤岛外被占据的地方
        occupied_map_camera = np.logical_not(
            self.get_island_around_pts(pts, height=self.vision_height)[0]
        )
        self.unexplored = unexplored
        self.unoccupied = unoccupied
        self.occupied = occupied
        self.island = island
        self.unexplored_neighbors = unexplored_neighbors
        self.occupied_map_camera = occupied_map_camera

        # 检测frontiers: frontier_areas 是属于island区域(已被观测未被占据), 并且该位置邻域大于等于8处小于等于9处未探索区域(即完全未被探索)
        # frontier_edge_areas 是属于island区域, 并且该位置邻域大于等于4处小于等于6处未探索区域
        # detect and update frontiers
        frontier_areas = np.argwhere(
            island
            & (unexplored_neighbors >= cfg.frontier_area_min)
            & (unexplored_neighbors <= cfg.frontier_area_max)
        )
        frontier_edge_areas = np.argwhere(
            island
            & (unexplored_neighbors >= cfg.frontier_edge_area_min)
            & (unexplored_neighbors <= cfg.frontier_edge_area_max)
        )

        grid_map_vis = None
        if visualization:
            grid_map_vis = self.plot_2d_map(occupied, frontier_areas, frontier_edge_areas, np.logical_not(unexplored).astype(int))

        if len(frontier_areas) == 0:
            # this happens when there are stairs on the floor, and the planner cannot handle this situation
            # just skip this question
            logging.error(f"Error in update_frontier_map: frontier area size is 0")
            self.frontiers = []
            return False, grid_map_vis
        if len(frontier_edge_areas) == 0:
            # this happens rather rarely
            logging.error(f"Error in update_frontier_map: frontier edge area size is 0")
            self.frontiers = []
            return False, grid_map_vis

        # 对前面检测到的 frontier_areas 点做聚类，方便把一片相邻的边界点视为一个「frontier 区域」
        # cluster frontier regions
        db = DBSCAN(eps=cfg.eps, min_samples=2).fit(frontier_areas)
        labels = db.labels_
        # get one point from each cluster
        valid_ft_angles = []
        for label in np.unique(labels):   # 每个 frontier_area 单独处理
            if label == -1:
                continue

            # 单个frontier_area里面的点坐标
            cluster = frontier_areas[labels == label]

            # 过滤小的 frontiers
            # filter out small frontiers
            area = len(cluster)
            if area < cfg.min_frontier_area:
                continue

            # 根据坐标, 把frontier_area的点转为相对于当前位置的相对角度
            # convert the cluster from voxel coordinates to polar angle coordinates
            angle_cluster = np.asarray(
                [
                    np.arctan2(
                        cluster[i, 1] - cur_point[1], cluster[i, 0] - cur_point[0]
                    )
                    for i in range(len(cluster))
                ]
            )  # range from -pi to pi

            # 计算这些角度覆盖的最小角度范围
            # get the range of the angles
            angle_range = get_angle_span(angle_cluster)

            # 如果最后一个角度减去第一个角度大于 一个接近2*pi的阈值, 证明角度分布覆盖了几乎整个圆, 就需要"warp"——即找到一个缺口角度
            # 返回这个缺口的中的一个随机的角度
            warping_gap = get_warping_gap(
                angle_cluster
            )  # add 2pi to angles that smaller than this to avoid angles crossing -pi/pi line
            if warping_gap is not None:
                angle_cluster[angle_cluster < warping_gap] += 2 * np.pi

            # valid_ft_angles包含: angle: 范围在[-pi, pi]的角度中心. region: 根据frontier点坐标生成一个2D frontier地图
            if angle_range > cfg.max_frontier_angle_range_deg * np.pi / 180:
                # 如果覆盖的角度范围大于150度, 则分裂成多类, 类别数是: 覆盖角度/150 + 1
                # cluster again on the angle, ie, split the frontier
                num_clusters = (
                    int(angle_range / (cfg.max_frontier_angle_range_deg * np.pi / 180))
                    + 1
                )
                db_angle = KMeans(n_clusters=num_clusters).fit(angle_cluster[..., None])
                labels_angle = db_angle.labels_
                for label_angle in np.unique(labels_angle):
                    if label_angle == -1:
                        continue
                    ft_angle = np.mean(angle_cluster[labels_angle == label_angle])
                    valid_ft_angles.append(
                        {
                            "angle": (
                                ft_angle - 2 * np.pi if ft_angle > np.pi else ft_angle
                            ),
                            "region": self.get_frontier_region_map(
                                cluster[labels_angle == label_angle]
                            ),
                        }
                    )
            else:
                ft_angle = np.mean(angle_cluster)
                valid_ft_angles.append(
                    {
                        "angle": ft_angle - 2 * np.pi if ft_angle > np.pi else ft_angle,
                        "region": self.get_frontier_region_map(cluster),
                    }
                )

        frame_ids, view_angles = [], []
        for frame_id, frame in scene.frames.items():
            frame_ids.append(frame_id)
            Twc = frame.pose
            view_angles.append(np.arctan2(Twc[1, 0], Twc[0, 0]))
        view_angles = np.array(view_angles)

        # remove frontiers that have been changed
        filtered_frontiers = []
        kept_frontier_area = np.zeros_like(self.frontier_map, dtype=bool)
        scale_factor = int(
            (0.1 / self._voxel_size) ** 2
        )  # when counting the number of pixels in the frontier region, we use a default voxel length of 0.1m. Then other voxel lengths should be scaled by this factor
        for frontier in self.frontiers:
            if frontier in filtered_frontiers:
                continue

            IoU_values, pix_diff_values = [], []
            for new_ft in valid_ft_angles:
                intersection = np.sum(frontier.region & new_ft["region"])
                union = np.sum(frontier.region | new_ft["region"])
                IoU_values.append(intersection / union)
                pix_diff_values.append(union-intersection)

            IoU_values, pix_diff_values = np.asarray(IoU_values), np.asarray(pix_diff_values)


            frontier_appended = False
            # 当差值小于等于75个点且IOU大于0.95(对于大区域), 或者差值小于3个点(对于小区域)
            if np.any(
                (
                    (IoU_values > cfg.region_equal_threshold)
                    & (pix_diff_values < 75 * scale_factor)
                )  # ensure that a normal step on a very large region can cause the large region to be considered as changed
                | (
                    pix_diff_values <= 3 * scale_factor
                )  # ensure that a very small region can be considered as unchanged
            ):  # do not update frontier that is too far from the agent:
                # 这证明地图里的这个frontier和新加的某个frontier重合度很高, 
                # 则将此frontier保留, 并将和此frontier重合最多的新增的frontier去除
                # the frontier is not changed (almost)
                filtered_frontiers.append(frontier)
                kept_frontier_area = kept_frontier_area | frontier.region
                frontier_appended = True
                # then remove that new frontier
                ft_idx = np.argmax(IoU_values)
                valid_ft_angles.pop(ft_idx)
            # 至少有两个IoU > 0.02, 并且这些IoU>0.02的IOU值的总和大于0.95, 但不超过1
            # 证明虽然单个新增的frontier没有和这个地图中frontier有很大重合, 但这个frontier的绝大部分区域已经被新的frontier覆盖了, 有可能是这个老的frontier分裂了
            elif (
                np.sum(IoU_values > 0.02) >= 2
                and cfg.region_equal_threshold
                < np.sum(IoU_values[IoU_values > 0.02])
                <= 1
            ):
                # 这种情况的处理方式是: 保留这个老的frontier, 把分裂的新的frontier都移除
                # if one old frontier is split into two new frontiers, and their sizes are equal
                # then keep the old frontier
                logging.debug(
                    f"Frontier one split to many: {IoU_values[IoU_values > 0.02]}"
                )
                filtered_frontiers.append(frontier)
                kept_frontier_area = kept_frontier_area | frontier.region
                frontier_appended = True
                # then remove those new frontiers
                ft_ids = list(np.argwhere(IoU_values > 0.02).squeeze())
                ft_ids.sort(reverse=True)
                for ft_idx in ft_ids:
                    valid_ft_angles.pop(ft_idx)
            # 新的frontier只有1个和这个老的frontier有重合, 且重合度不高
            elif np.sum(IoU_values > 0.02) == 1:
                # 首先找到这个和老frontier有重叠的新frontier, 计算新frontier与地图中所有frontier的IOU
                # if some old frontiers are merged into one new frontier
                ft_idx = np.argmax(IoU_values)
                IoU_with_old_ft = np.asarray([IoU(valid_ft_angles[ft_idx]["region"], ft.region) for ft in self.frontiers])

                # 至少有两个老的frontier和这个新的frontier的IoU > 0.02, 并且这些IoU>0.02的IOU值的总和大于0.95, 但不超过1
                # 证明这个新的frontier的绝大部分区域已经被老的frontier覆盖了, 有可能是这个老的frontier在探索途中合并了
                if (np.sum(IoU_with_old_ft > 0.02) >= 2 and cfg.region_equal_threshold < np.sum(IoU_with_old_ft[IoU_with_old_ft > 0.02]) <= 1):
                    # 如果这个新的frontier的面积基本覆盖了和该frontiers IOU>0.02的所有老frontiers的面积 (面积比值大于0.95)
                    # 则证明确实是老的frontier合并成了一个新的frontier
                    if (
                        np.sum(valid_ft_angles[ft_idx]["region"]) / np.sum(
                            [
                                np.sum(self.frontiers[old_ft_id].region)
                                for old_ft_id in np.argwhere(
                                    IoU_with_old_ft > 0.02
                                ).squeeze()
                            ]
                        )
                        > cfg.region_equal_threshold
                    ):
                        # if the new frontier is merged from two or more old frontiers, and their sizes are equal
                        # then add all the old frontiers
                        logging.debug(
                            f"Frontier many merged to one: {IoU_with_old_ft[IoU_with_old_ft > 0.02]}"
                        )
                        # 把所有这些老的frontiers都保留, 新的去除
                        for i in list(np.argwhere(IoU_with_old_ft > 0.02).squeeze()):
                            if self.frontiers[i] not in filtered_frontiers:
                                filtered_frontiers.append(self.frontiers[i])
                                kept_frontier_area = kept_frontier_area | self.frontiers[i].region
                        valid_ft_angles.pop(ft_idx)
                        frontier_appended = True

            # 对于地图中的某个老的frontier, 如果不是上面三种情况: 
            #   1. 和某个新的frontier重合很多
            #   2. 分裂成了多个新的frontier, 且重合很多
            #   3. 和其他老的frontier合并成老一个新的frontier, 且重合很多
            # 证明这个老的frontier在当前步骤被探索了, 已经不是一个frontier了, 或者已经被更新了很多(增加或减少)
            if not frontier_appended:
                # 从frontier地图标记为非frontier
                self.free_frontier(frontier)

                # 如果这个老的frontier和某个新的frontier重合比较大(大于0.8), 但又没有特别大(没有超过0.95)
                # 证明新的一个frontier是基于这个老的更新的, 且更新的也不是非常多(IOU仍大于0.8)
                if np.any(IoU_values > 0.8):
                    # the frontier is slightly updated
                    # choose the new frontier that updates the current frontier
                    update_ft_idx = np.argmax(IoU_values)
                    ang = valid_ft_angles[update_ft_idx]["angle"]

                    # 如果这个新的frontier方向1米之内有障碍物
                    # get_collision_distance: 给定起点和方向, 在占据地图里沿着该方向行进, 直到碰到障碍物或走到 max_step 为止, 返回行进的距离(voxel空间的). 
                    # if the new frontier has no valid observations
                    if 1 > self._voxel_size * get_collision_distance(
                        occupied_map=occupied_map_camera,
                        pos=cur_point,
                        direction=np.array([np.cos(ang), np.sin(ang)]),
                    ):
                        # 新建一个frontier添加到地图里, 这个加到地图里的frontier包含:
                        #   1. 老frontier的图片, feature, target_detected
                        #   2. 新frontier的角度, frontier_edge_areas, 位置, region
                        # create a new frontier with the old image
                        new_frontier = self.create_frontier(
                                valid_ft_angles[update_ft_idx],
                                frontier_edge_areas=frontier_edge_areas,
                                cur_point=cur_point,
                            )
                        
                        new_frontier_angle = valid_ft_angles[update_ft_idx]["angle"]
                        new_view_angle_diff, closest_idx = self.closest_angle(new_frontier_angle, view_angles)

                        old_frontier_frame_pose = frontier.frame.pose
                        old_view_angle = np.arctan2(old_frontier_frame_pose[1, 0], old_frontier_frame_pose[0, 0])
                        old_view_angle_diff = np.arctan2(np.sin(old_view_angle - new_frontier_angle), np.cos(old_view_angle - new_frontier_angle))
                        old_view_angle_diff = np.abs(old_view_angle_diff)

                        if new_view_angle_diff < old_view_angle_diff:
                            new_frontier.frame = scene.frames[frame_ids[closest_idx]]
                            new_frontier.image = new_frontier.frame.image_path
                            new_frontier.feature = new_frontier.frame.image
                            new_frontier.view_angle_diff = new_view_angle_diff
                        else:
                            new_frontier.frame = frontier.frame
                            new_frontier.image = frontier.image 
                            new_frontier.feature = frontier.feature
                            new_frontier.view_angle_diff = old_view_angle_diff

                        new_frontier.target_detected = frontier.target_detected
                        
                        filtered_frontiers.append(new_frontier)

                        # 从新frontier的list中去除这个 frontier
                        valid_ft_angles.pop(update_ft_idx)

                        # 在 kept_frontier_area 中把这个新的frontier的region加进去
                        kept_frontier_area = (
                            kept_frontier_area | filtered_frontiers[-1].region
                        )

        self.frontiers = filtered_frontiers

        # 上面的操作是要在新增的frontiers里面先找到和地图中老的frontiers重合度高的, 并先去掉这些
        # 这步操作是 把剩下的新的frontiers 都新建一个Frontier对象, 加到地图里(self.frontiers)
        # create new frontiers and add to frontier list
        for i, ft_data in enumerate(valid_ft_angles):
            # exclude the new frontier's region that is already covered by the existing frontiers
            ft_data["region"] = ft_data["region"] & np.logical_not(kept_frontier_area)
            if np.sum(ft_data["region"]) > 0:
                new_frontier = self.create_frontier(
                    ft_data,
                    frontier_edge_areas=frontier_edge_areas,
                    cur_point=cur_point,
                )
                        
                new_frontier_angle = ft_data["angle"]
                new_view_angle_diff, closest_idx = self.closest_angle(new_frontier_angle, view_angles)

                new_frontier.frame = scene.frames[frame_ids[closest_idx]]
                new_frontier.image = new_frontier.frame.image_path
                new_frontier.feature = new_frontier.frame.image
                new_frontier.view_angle_diff = new_view_angle_diff

                self.frontiers.append(new_frontier)

                if save_frontier_image:
                    assert os.path.exists(
                        eps_frontier_dir
                    ), f"Error in update_frontier_map: {eps_frontier_dir} does not exist"
                    plt.imsave(
                        os.path.join(eps_frontier_dir, f"{cnt_step}_{i}.png"),
                        new_frontier.frame.image,
                    )

        # clear temporary frames
        scene.clear_frames()
        return True, grid_map_vis

    def get_island_around_pts(self, pts, fill_dim=0.4, height=0.4):
        """
        函数做了以下操作:
            1. 根据TSDF地图生成2D占据地图
            2. 将2D占据地图连通性, 分成各个孤岛, 找到距离当前位置最近的孤岛
            3. 如果给的height是生成2D占据地图导航所用的height, 则把包含frontier最多的孤岛区域也加到上面被选中的孤岛区域中
            4. 返回被选中的孤岛区域和2D占据地图, 两个返回值都是2D bool数组
        Find the empty space around the point (x,y,z) in the world frame
        """
        # Convert to voxel coordinates
        cur_point = self.normal2voxel(pts)

        # 根据3D voxel情况, 得到一个2D占据地图: 只用当上层空闲(height处)且下层有地面时，才标记为 True. 未知区域为False
        # TSDF > 0 --> 点在表面外部，即 空闲空间 (free space)
        # TSDF < 0 --> 点在表面内部，即 被物体或地面占据 (occupied space)
        # TSDF = 0 --> 点在表面上。
        # Check if the height voxel is occupied
        height_voxel = int(height / self._voxel_size) + self.min_height_voxel
        unoccupied = np.logical_and(
            self._tsdf_vol_cpu[:, :, height_voxel] > 0, self._tsdf_vol_cpu[:, :, 0] < 0
        )  # check there is ground below

        # 把初始位置及其邻域的一些点设置为free区域(已被观测到未被占用)
        # Set initial pose to be free
        for point in self.init_points:
            unoccupied[point[0], point[1]] = 1

        # filter small islands smaller than size 2x2 and fill in gap of size 2
        # fill_size = int(fill_dim / self._voxel_size)
        # structuring_element_close = np.ones((fill_size, fill_size)).astype(bool)
        # unoccupied = close_operation(unoccupied, structuring_element_close)

        # 先把free区域分成各个连同的孤岛, 再找到与当前位置最近的孤岛
        # 一般整个区域只有一个, 但由于相机视角原因, 当前位置的邻域可能未被观测到, 而当前区域的更小邻域被设置成了free, 导致不止一个孤岛
        # Find the connected component closest to the current location is, if the current location is not free
        # this is a heuristic to determine reachable space, although not perfect
        islands = measure.label(unoccupied, connectivity=1) # 把空闲空间里(为True)连续的 True 区域分成独立的区域并编号。
        if unoccupied[cur_point[0], cur_point[1]] == 1:
            islands_ind = islands[cur_point[0], cur_point[1]]  # use current one
        else:
            # find the closest one - tbh, this should not happen, but it happens when the robot cannot see the space immediately in front of it because of camera height and fov
            y, x = np.ogrid[: unoccupied.shape[0], : unoccupied.shape[1]]
            dist_all = np.sqrt((x - cur_point[1]) ** 2 + (y - cur_point[0]) ** 2)
            dist_all[islands == islands[cur_point[0], cur_point[1]]] = np.inf
            island_coords = np.unravel_index(np.argmin(dist_all), dist_all.shape)
            islands_ind = islands[island_coords[0], island_coords[1]]
        island = islands == islands_ind

        # 断言: 孤岛数组里的被占据元素数量和原本unoccupied里占据元素数量相等
        assert (islands == 0).sum() == (
            unoccupied == 0
        ).sum(), f"{(islands == 0).sum()} != {(unoccupied == 0).sum()}"

        # 如果给的height是获取占据地图的那个height, 则找到frontier最多的那个孤岛, 加到上面被选中的与当前位置最近的孤岛里
        # also we need to include the island of all existing frontiers when calculating island at the same height as frontier
        if abs(height - self.occupancy_height) < 1e-3: 
            for frontier in self.frontiers:
                # frontier区域对应的岛屿编号
                frontier_inds = islands[frontier.region] 

                # 众数统计，找到出现次数最多的岛屿编号
                # get the most common index
                mode_result = stats.mode(frontier_inds, axis=None)
                frontier_ind = mode_result.mode

                # 如果众数不是0(0表示背景/非空区域), 说明 frontier 属于某个连通岛屿。
                if frontier_ind != 0:
                    island = island | (islands == frontier_ind)

        # 返回被选中的孤岛, 和占据地图
        return island, unoccupied

    def get_frontier_region_map(self, frontier_coordinates):
        '''
        根据frontier点坐标生成一个2D frontier地图
        '''
        # frontier_coordinates: [N, 2] ndarray of the coordinates covered by the frontier in voxel space
        region_map = np.zeros_like(self.frontier_map, dtype=bool)
        for coord in frontier_coordinates:
            region_map[coord[0], coord[1]] = True
        return region_map

    def create_frontier(
        self, ft_data: dict, frontier_edge_areas, cur_point
    ) -> Frontier:
        '''
        # ft_data包含frontier的: angle: 范围在[-pi, pi]的角度中心. region: 根据frontier点坐标生成一个2D frontier地图
        '''
        ft_direction = np.array([np.cos(ft_data["angle"]), np.sin(ft_data["angle"])])

        # 在这个frontier张成的2D地图中, 计算一个位置周围5*5区域(包含这个位置)frontier点的数量
        kernel = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        frontier_edge = ndimage.convolve(
            ft_data["region"].astype(int), kernel, mode="constant", cval=0
        )

        # 对于周围包含的frontier点在2-12个之间的frontier_edge_areas, 则保留
        frontier_edge_areas_filtered = np.asarray(
            [p for p in frontier_edge_areas if 2 <= frontier_edge[p[0], p[1]] <= 12]
        )

        # 如果有上面的点, 就把保留的点作为 新的 frontier_edge_areas
        if len(frontier_edge_areas_filtered) > 0:
            frontier_edge_areas = frontier_edge_areas_filtered

        # 机器人为中心的各个frontier_edge_areas的方向, 且对距离做了归一化
        all_directions = frontier_edge_areas - cur_point[:2]
        all_direction_norm = np.linalg.norm(all_directions, axis=1, keepdims=True)
        all_direction_norm = np.where(
            all_direction_norm == 0, np.inf, all_direction_norm
        )
        all_directions = all_directions / all_direction_norm

        # 选择和frontier中心方向夹角最小的5个frontier_edge_areas里的点作为候选, 并在候选点里面找到与当前位置最近的点作为中心
        # ft_direction为这个frontier中心的方向
        # the center is the closest point in the edge areas from current point that have close cosine angles
        cos_sim_rank = np.argsort(-np.dot(all_directions, ft_direction))
        center_candidates = np.asarray(
            [frontier_edge_areas[idx] for idx in cos_sim_rank[:5]]
        )
        center = center_candidates[
            np.argmin(np.linalg.norm(center_candidates - cur_point[:2], axis=1))
        ]

        # 对一个点 center 进行调整, 使它远离障碍物, 但又不会离当前位置太远
        center = adjust_navigation_point(
            center,
            self.occupied,
            max_dist=0.5,
            max_adjust_distance=0.3,
            voxel_size=self._voxel_size,
        )

        region = ft_data["region"]

        # 新增 frontier id, 在frontier map里面把该frontier对于区域set为该frontier的id
        # allocate an id for the frontier
        # assert np.all(self.frontier_map[region] == 0)
        frontier_id = self.frontier_counter
        self.frontier_map[region] = frontier_id
        self.frontier_counter += 1

        # 新建一个Frontier类并返回
        #   position: 表示该frontier中心点, 会被调整为距离障碍物有一定距离
        #   orientation: frontier中心的方向, 没有被调整
        #   region:  根据frontier点坐标生成一个2D frontier地图
        return Frontier(
            position=center,
            orientation=ft_direction,
            region=region,
            frontier_id=frontier_id,
        )

    def free_frontier(self, frontier: Frontier):
        self.frontier_map[self.frontier_map == frontier.frontier_id] = 0