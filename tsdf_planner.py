import os.path

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.cluster import DBSCAN, KMeans
from scipy import stats
import torch
import habitat_sim
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field
import supervision as sv
from matplotlib.patches import Wedge
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

from geom import *
from habitat_data import pos_normal_to_habitat, pos_habitat_to_normal
from tsdf_base import TSDFPlannerBase
from utils import resize_image
from map_elements import Frame, Keyframe, Object3D
from map import Map

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


@dataclass
class SceneGraphItem:
    object_id: int
    bbox_center: np.ndarray
    confidence: float
    image: str


@dataclass
class SnapShot:
    image: str
    color: Tuple[float, float, float]
    obs_point: np.ndarray  # integer position in voxel grid
    full_obj_list: Dict[int, float] = field(
        default_factory=dict
    )  # object id to confidence
    cluster: List[int] = field(default_factory=list)
    position: np.ndarray = None

    def __eq__(self, other):
        raise NotImplementedError("Cannot compare SnapShot objects.")


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

        # about navigation
        self.max_point: [Frontier, SnapShot] = (
            None  # the frontier/snapshot the agent chooses
        )
        self.target_point: [np.ndarray] = (
            None  # the corresponding navigable location of max_point. The agent goes to this point to observe the max_point
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
        pts_habitat = pts.copy()

        # 点转为位姿, 旋转补为单位阵
        pts = pos_habitat_to_normal(pts)

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

            # # 地图中frontier和新frontier 的IOU
            # IoU_values = np.asarray(
            #     [IoU(frontier.region, new_ft["region"]) for new_ft in valid_ft_angles]
            # )

            # # 地图中frontier和新frontier 不一致的面积大小
            # pix_diff_values = np.asarray(
            #     [
            #         pix_diff(frontier.region, new_ft["region"])
            #         for new_ft in valid_ft_angles
            #     ]
            # )

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


    def set_next_navigation_point(
        self,
        choice: Union[Keyframe, Frontier, Object3D], # 目标点
        pts,    # 当前位置
        scene: Map,
        cfg,
        pathfinder,
        random_position=False,
        observe_snapshot=True,
    ) -> bool:
        '''
        这个函数是要根据ChatGPT挑选的目标, 来选择可以导航的目标点, 设置self.max_point(被选择的snaphhot或者frontier) 和 self.target_point(在habitat下具体的目标点)
        如果被选择的目标是个snapshot, 则先会对这个snapshot包含的物体的物体中心求一个均值, 再在这个物体中心均值附近搜索合适的导航点(未被占据, 与观测点连通, 不靠墙)
        如果被选择的目标是个frontier, 则会在frontier的位置搜索合适的导航点(在2D边界内, 不是被占据的, 在被选中的岛屿内)
        '''
        if self.max_point is not None or self.target_point is not None:
            # if the next point is already set
            logging.error(
                f"Error in set_next_navigation_point: the next point is already set: {self.max_point}, {self.target_point}"
            )
            return False
        pts = pos_habitat_to_normal(pts)
        cur_point = self.normal2voxel(pts)
        self.max_point = choice


        if type(choice) == Object3D:
            object_center = choice.bbox.center
            object_center = self.normal2voxel(object_center)[:2]
            object_center = np.asarray(object_center)

            target_point = object_center
            # # set the object center as the navigation target
            # target_navigable_point = get_nearest_true_point(target_point, unoccupied)  # get the nearest unoccupied point for the nav target
            # since it's not proper to directly go to the target point,
            # we'd better find a navigable point that is certain distance from it to better observe the target
            if not random_position:
                # 在给定点 (target_point) 周围找到一个合适的观察点(target_navigable_point), 要求该点位于空闲区域, 不靠墙, 并且距离大约为 dist(0.75m)
                # the target navigation point is deterministic
                target_navigable_point = get_proper_observe_point(
                    target_point,
                    self.unoccupied,
                    cur_point=cur_point,
                    dist=cfg.final_observe_distance / self._voxel_size,
                )
            else:
                # 给定一个目标点, 找一个 随机的合法合适的观测点, 并且确保目标点和观察点之间没有障碍物
                # [min_dist, max_dist]: 限制观测点和目标点的距离范围 (0.75~1.25)
                target_navigable_point = get_random_observe_point(
                    target_point,
                    self.unoccupied,
                    min_dist=cfg.final_observe_distance / self._voxel_size,
                    max_dist=(cfg.final_observe_distance + 1.5) / self._voxel_size,
                )

            # 如果没找到合适的位置, 由于目标物体太远, 位于未探索区域, 所以占据地图里不是未占据状态
            if target_navigable_point is None:
                # 基于 Habitat 的导航器 pathfinder, 在目标点附近(半径1m内)找到一个可行走的, 且高度与机器人当前位置高度相差0.1m以内的点
                # this is usually because the target object is too far, so its surroundings are not detected as unoccupied
                # so we just temporarily use pathfinder to find a navigable point around it
                target_point_normal = (
                    target_point * self._voxel_size + self._vol_origin[:2]
                )
                target_point_normal = np.append(target_point_normal, pts[-1])
                target_point_habitat = pos_normal_to_habitat(target_point_normal)

                target_navigable_point_habitat = (
                    get_proper_observe_point_with_pathfinder(
                        target_point_habitat, pathfinder, height=pts[-1]
                    )
                )
                if target_navigable_point_habitat is None:
                    logging.error(
                        f"Error in set_next_navigation_point: cannot find a proper navigable point around the target object"
                    )
                    return False

                target_navigable_point = self.habitat2voxel(
                    target_navigable_point_habitat
                )[:2]
            self.target_point = target_navigable_point
            return True
        # 如果选择的目标是 snapshot
        elif type(choice) == Keyframe:
            # 把snapshot包含的物体的中心坐标转为voxel空间下坐标, 并去重
            obj_centers = [scene.objects_3d[obj_id].bbox.center for obj_id in choice.objects_3d]
            obj_centers = [self.normal2voxel(center)[:2] for center in obj_centers]
            obj_centers = list(set([tuple(center) for center in obj_centers]))  # remove duplicates
            obj_centers = np.asarray(obj_centers)

            # snapshot_center 为物体中心的均值
            snapshot_center = np.mean(obj_centers, axis=0)
            choice.position = snapshot_center

            # 
            if not observe_snapshot:
                # if the agent does not need to observe the snapshot, then the target point is the snapshot center
                target_point = snapshot_center
                # 找到距离 target_point 最近的没有被占据的点
                self.target_point = get_nearest_true_point(
                    target_point, self.unoccupied
                )  # get the nearest unoccupied point for the nav target
                return True

            # 如果 snapshot 里只有1个物体
            if len(obj_centers) == 1:
                # 那这个物体中心就是目标点
                # if there is only one object in the snapshot, then the target point is the object center
                target_point = snapshot_center
                # # set the object center as the navigation target
                # target_navigable_point = get_nearest_true_point(target_point, unoccupied)  # get the nearest unoccupied point for the nav target
                # since it's not proper to directly go to the target point,
                # we'd better find a navigable point that is certain distance from it to better observe the target
                if not random_position:
                    # 在给定点 (target_point) 周围找到一个合适的观察点(target_navigable_point), 要求该点位于空闲区域, 不靠墙, 并且距离大约为 dist(0.75m)
                    # the target navigation point is deterministic
                    target_navigable_point = get_proper_observe_point(
                        target_point,
                        self.unoccupied,
                        cur_point=cur_point,
                        dist=cfg.final_observe_distance / self._voxel_size,
                    )
                else:
                    # 给定一个目标点, 找一个 随机的合法合适的观测点, 并且确保目标点和观察点之间没有障碍物
                    # [min_dist, max_dist]: 限制观测点和目标点的距离范围 (0.75~1.25)
                    target_navigable_point = get_random_observe_point(
                        target_point,
                        self.unoccupied,
                        min_dist=cfg.final_observe_distance / self._voxel_size,
                        max_dist=(cfg.final_observe_distance + 1.5) / self._voxel_size,
                    )

                # 如果没找到合适的位置, 由于目标物体太远, 位于未探索区域, 所以占据地图里不是未占据状态
                if target_navigable_point is None:
                    # 基于 Habitat 的导航器 pathfinder, 在目标点附近(半径1m内)找到一个可行走的, 且高度与机器人当前位置高度相差0.1m以内的点
                    # this is usually because the target object is too far, so its surroundings are not detected as unoccupied
                    # so we just temporarily use pathfinder to find a navigable point around it
                    target_point_normal = (
                        target_point * self._voxel_size + self._vol_origin[:2]
                    )
                    target_point_normal = np.append(target_point_normal, pts[-1])
                    target_point_habitat = pos_normal_to_habitat(target_point_normal)

                    target_navigable_point_habitat = (
                        get_proper_observe_point_with_pathfinder(
                            target_point_habitat, pathfinder, height=pts[-1]
                        )
                    )
                    if target_navigable_point_habitat is None:
                        logging.error(
                            f"Error in set_next_navigation_point: cannot find a proper navigable point around the target object"
                        )
                        return False

                    target_navigable_point = self.habitat2voxel(
                        target_navigable_point_habitat
                    )[:2]
                self.target_point = target_navigable_point
                return True
            else:
                if not random_position:
                    target_point = get_proper_snapshot_observation_point(
                        obj_centers=obj_centers,
                        snapshot_observation_point=self.normal2voxel(choice.pose[:3, 3]),
                        unoccupied_map=self.unoccupied,
                        min_obs_dist=cfg.final_observe_distance / self._voxel_size - 1,
                        max_obs_dist=cfg.final_observe_distance / self._voxel_size + 1,
                    )
                else:
                    target_point = get_random_snapshot_observation_point(
                        obj_centers=obj_centers,
                        snapshot_observation_point=self.normal2voxel(choice.pose[:3, 3]),
                        unoccupied_map=self.unoccupied,
                        min_obs_dist=cfg.final_observe_distance / self._voxel_size - 1,
                        max_obs_dist=cfg.final_observe_distance / self._voxel_size + 1,
                    )
                if target_point is None:
                    logging.error(
                        f"Error in set_next_navigation_point: cannot find a proper observation point for the snapshot"
                    )
                    return False

                self.target_point = target_point
                return True
        # 如果选择的目标是 frontier
        elif type(choice) == Frontier:
            # find the direction into unexplored
            ft_direction = self.max_point.orientation

            # find an unoccupied point between the agent and the frontier
            next_point = np.array(self.max_point.position, dtype=float)
            try_count = 0
            # 沿着目标点方向回朔搜索, 目标点需要, 在2D边界内, 不是被占据的, 在被选中的岛屿内
            # not优先级大于or, 所以当上面三个条件有一个不满足, 则继续循环
            while (
                not self.check_within_bnds(next_point.astype(int))
                or self.occupied[int(next_point[0]), int(next_point[1])]
                or not self.island[int(next_point[0]), int(next_point[1])]
            ):
                next_point -= ft_direction
                try_count += 1
                if try_count > 1000:
                    logging.error(
                        f"Error in set_next_navigation_point: cannot find a proper next point"
                    )
                    return False

            self.target_point = next_point.astype(int)
            return True
        else:
            logging.error(
                f"Error in find_next_pose_with_path: wrong choice type: {type(choice)}"
            )
            return False

    def agent_step(
        self,
        pts,        # 机器人当前位置
        angle,      # 机器人当前旋转
        pathfinder,     
        cfg,
        path_points=None,
    ):
        '''
        根据所选的目标位置, 前进一步, 前进的距离不超过 max_dist_from_cur (配置文件为1m)
        '''
        if self.max_point is None or self.target_point is None:
            logging.error(
                f"Error in agent_step: max_point or next_point is None: {self.max_point}, {self.target_point}"
            )
            return (None,)
        
        # check the distance to next navigation point
        # if the target navigation point is too far
        # then just go to a point between the current point and the target point
        # 当前位置
        pts = pos_habitat_to_normal(pts)
        cur_point = self.normal2voxel(pts)

        # 当前位置和目标点能容忍的最大距离(默认都是1m)
        max_dist_from_cur = (
            cfg.max_dist_from_cur_phase_1
            if type(self.max_point) == Frontier
            else cfg.max_dist_from_cur_phase_2
        )  # in phase 2, the step size should be smaller

        # 计算当前位置和目标位置之间的距离, 优先使用 Habitat 的导航路径距离, 如果路径不可达则退化为欧式距离
        # 返回值是距离和路径点
        dist, path_to_target = self.get_distance(
            cur_point[:2], self.target_point, height=pts[2], pathfinder=pathfinder
        )

        # 如果当前位置和目标点距离大于最大容忍距离
        if dist > max_dist_from_cur:
            print("objective position is too far from current position")
            target_arrived = False
            # 如果当前点和目标点之间有可达路径
            if path_to_target is not None:
                print("there is a path to get there")
                # 去掉路径点的高度值
                # drop the y value of the path to avoid errors when calculating seg_length
                path_to_target = [np.asarray([p[0], 0.0, p[2]]) for p in path_to_target]

                # 在这条路径上, 找到从当前位置点出发, 距离为 max_dist_from_cur 的下一个点
                # if the pathfinder find a path, then just walk along the path for max_dist_from_cur distance
                dist_to_travel = max_dist_from_cur
                next_point = None
                for i in range(len(path_to_target) - 1):
                    seg_length = np.linalg.norm(
                        path_to_target[i + 1] - path_to_target[i]
                    )
                    if seg_length < dist_to_travel:
                        dist_to_travel -= seg_length
                    else:
                        # 找到了使距离大于 max_dist_from_cur 点, 为了让距离等于 max_dist_from_cur, 
                        # 则在最后两个路径点之间 找到那个使距离正好等于max_dist_from_cur的点, 再把这个点设为 next_point (单个点)
                        # find the point on the segment according to the length ratio
                        next_point_habitat = (
                            path_to_target[i]
                            + (path_to_target[i + 1] - path_to_target[i])
                            * dist_to_travel
                            / seg_length
                        )
                        next_point = self.normal2voxel(
                            pos_habitat_to_normal(next_point_habitat)
                        )[:2]
                        break
                
                # 如果next_point是None, 说明总距离小于 max_dist_from_cur, 有可能是去掉高度值导致了距离变短
                # 这时可以直接把 目标点 作为 next_point
                if next_point is None:
                    print("however, next_point is None, so target_arrived is true")
                    # this is a very rare case that, the sum of the segment lengths is smaller than the dist returned by the pathfinder
                    # and meanwhile the max_dist_from_cur larger than the sum of the segment lengths
                    # resulting that the previous code cannot find a proper point in the middle of the path
                    # in this case, just go to the target point
                    next_point = self.target_point.copy()
                    target_arrived = True
            # 如果当前点和目标点之间没有可达路径, (在当前位置和目标点距离大于最大容忍距离前提下)
            else:
                print("there is no path to get there")
                # if the pathfinder cannot find a path, then just go to a point between the current point and the target point
                logging.info(
                    f"pathfinder cannot find a path from {cur_point[:2]} to {self.target_point}, just go to a point between them"
                )
                # 设置初始 next_point, 即沿着目标点方向 (walk_dir方向), 距离当前位置为 max_dist_from_cur 的点
                walk_dir = self.target_point - cur_point[:2]
                walk_dir = walk_dir / np.linalg.norm(walk_dir)
                next_point = (
                    cur_point[:2] + walk_dir * max_dist_from_cur / self._voxel_size
                )

                # 根据上面初始的 next_point, 找到合适的 next_point 需同时满足三个条件: 在2D边界内, 在当前岛屿内, 未被占据
                # 如不满足, 则沿着 walk_dir 反方向回退, 每次回退1个voxel, 最多回退1000次
                # 如果找到了, 则则返回这个修正的 next_point, 如果未找到, 返回None
                # ensure next point is valid, otherwise go backward a bit
                try_count = 0
                while (
                    not self.check_within_bnds(next_point)
                    or not self.island[
                        int(np.round(next_point[0])), int(np.round(next_point[1]))
                    ]
                    or self.occupied[
                        int(np.round(next_point[0])), int(np.round(next_point[1]))
                    ]
                ):
                    next_point -= walk_dir
                    try_count += 1
                    if try_count > 1000:
                        logging.error(
                            f"Error in agent_step: cannot find a proper next point"
                        )
                        return (None,)
                next_point = np.round(next_point).astype(int)
        # 如果当前位置和目标点距离不大于最大容忍距离, 则直接把 目标点设置为 next_point
        else:
            print("objective position is near current position, success")
            target_arrived = True
            next_point = self.target_point.copy()

        # 对一个点 next_point 进行调整, 使它远离障碍物(大于max_dist, 默认0.5), 但又不会离当前位置太远(小于max_adjust_distance)
        next_point_old = next_point.copy()
        next_point = adjust_navigation_point(
            next_point,
            self.occupied,
            voxel_size=self._voxel_size,
            max_adjust_distance=0.1,
        )

        # target_arrived 为 True 的 条件:
        #       1. 当前位置和目标点距离(3D)不大于最大容忍距离 max_dist_from_cur
        #       2. 当前位置和目标点距离大于最大容忍距离, 但当前点和目标点之间有可达路径(2D), 且路径距离(2D)小于 最大容忍距离
        # 如果为 True, 方向 direction 设置为目标点和当前点组成的向量
        # determine the direction
        if target_arrived:  # if the next arriving position is the target point
            if type(self.max_point) == Frontier:
                # direction = self.rad2vector(angle)  # if the target is a frontier, then the agent's orientation does not change
                direction = (
                    self.target_point - cur_point[:2]
                )  # if the target is a frontier, then the agent should face the target point
            elif type(self.max_point) == Object3D:
                object_center = self.max_point.bbox.center
                object_center = self.normal2voxel(object_center)[:2]
                object_center = np.asarray(object_center)
                direction = (object_center - cur_point[:2])
            else:
                direction = (
                    self.max_point.position - cur_point[:2]
                )  # if the target is an object, then the agent should face the object
        # 如果 target_arrived 为False, 方向 direction 设置为 next_point 和当前点组成的向量
        else:  # the agent is still on the way to the target point
            direction = next_point - cur_point[:2]

        # 如果方向向量(非单位)太短, 即目标点或者next_point与当前位置重合, 则把 机器人当前旋转 设为 direction
        if (
            np.linalg.norm(direction) < 1e-6
        ):  # this is a rare case that next point is the same as the current point
            # usually this is a problem in the pathfinder
            logging.warning(
                f"Warning in agent_step: next point is the same as the current point when determining the direction"
            )
            direction = self.rad2vector(angle)

        # direction 向量正则化
        direction = direction / np.linalg.norm(direction)

        # 将 next_point 由 voxel坐标系转为真实坐标系
        # Convert back to world coordinates
        next_point_normal = next_point * self._voxel_size + self._vol_origin[:2]

        # 目标点和当前点 张成向量的角度
        # Find the yaw angle again
        next_yaw = np.arctan2(direction[1], direction[0]) - np.pi / 2

        # path_points默认是 None
        # update the path points
        if path_points is not None:
            # 根据一个新位置 next_point_normal, 更新路径 path_points, 从离该点(next_point_normal)最近的路径段开始继续走
            updated_path_points = self.update_path_points(
                path_points, next_point_normal
            )
        else:
            updated_path_points = None

        # 设置 next_point 附近 0.7m 的邻域中未被占据的区域为已探索
        # set the surrounding points of the next point as explored
        unoccupied_coords = np.argwhere(self.unoccupied)
        dists_unoccupied = np.linalg.norm(unoccupied_coords - next_point, axis=1)
        near_coords = unoccupied_coords[
            dists_unoccupied < cfg.surrounding_explored_radius / self._voxel_size
        ]
        self._explore_vol_cpu[near_coords[:, 0], near_coords[:, 1], :] = 1

        # 如果到达目标点了, 更新 self.max_point 和 self.target_point 都为 None. 
        if target_arrived:
            self.max_point = None
            self.target_point = None

        print("target_arrived === {}".format(target_arrived))

        return (
            self.normal2habitat(next_point_normal),  # next_point 的 habitat 坐标
            next_yaw,       # 目标点和当前点 张成向量的角度
            next_point,     # next_point 的 voxel 坐标
            updated_path_points,    # 更新后的路标点, 这里是None
            target_arrived,         # 是否到达了目标点 (即目标点距离和当前位置距离小于 max_dist_from_cur)
        )

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
