
import numpy as np
from scipy import stats
import torch
from typing import List, Tuple, Optional, Dict, Union

import habitat_sim

from planner.geom import *
from habitat.habitat_data import pos_normal_to_habitat, pos_habitat_to_normal
from map.map_elements import Frame, Keyframe, Object3D
from map.map import Map
from planner.tsdf_planner import Frontier, TSDFPlanner   

class Policy:
    def __init__(self,):
            # about navigation

        # the Frontier/Object3D the agent chooses
        self.max_point: [Frontier, Keyframe, Object3D] = None

        # the corresponding navigable location of max_point. The agent goes to this point to observe the max_point
        self.target_point: [np.ndarray] = None


    def get_distance(self, p1, p2, height, planner, pathfinder, input_voxel=True):
        '''
        计算两个点之间的距离, 优先使用 Habitat 的导航路径距离(geodesic distance), 如果路径不可达则退化为欧式距离
        返回值是距离和路径点
        '''
        # p1, p2 are in voxel space or habitat space
        # convert p1, p2 to habitat space if input_voxel is True
        if input_voxel:
            p1_world = p1 * planner.voxel_size() + planner.vol_origin()[:2]
            p2_world = p2 * planner.voxel_size() + planner.vol_origin()[:2]
        else:
            p1_world = p1
            p2_world = p2

        p1_world = np.append(p1_world, height)
        p1_habitat = pos_normal_to_habitat(p1_world)

        p2_world = np.append(p2_world, height)
        p2_habitat = pos_normal_to_habitat(p2_world)

        path = habitat_sim.ShortestPath()
        path.requested_start = p1_habitat
        path.requested_end = p2_habitat
        found_path = pathfinder.find_path(path)

        if found_path:
            return path.geodesic_distance, path.points

        # 如果无可用路径, 则在点附近邻域找可导航的点
        # if path not found, then try to find a path to a near point of p1 and p2
        p1_habitat_near = get_near_navigable_point(p1_habitat, pathfinder, radius=0.2)
        p2_habitat_near = get_near_navigable_point(p2_habitat, pathfinder, radius=0.4)

        if p1_habitat_near is not None and p2_habitat_near is not None:
            path.requested_start = p1_habitat_near
            path.requested_end = p2_habitat_near
            found_path = pathfinder.find_path(path)
            if found_path:
                return path.geodesic_distance, path.points

        # if still not found, then return the euclidean distance
        if input_voxel:
            return np.linalg.norm(p1 - p2) * planner.voxel_size(), None
        else:
            return np.linalg.norm(p1 - p2), None


    def set_next_navigation_point(
        self,
        choice: Union[Keyframe, Frontier, Object3D], # 目标点
        pts,    # 当前位置
        scene: Map,
        planner: TSDFPlanner,
        cfg,
        pathfinder,
        random_position=False,
        observe_snapshot=True,
    ) -> bool:
        '''
        这个函数是要根据ChatGPT挑选的目标, 来选择可以导航的目标点, 设置self.max_point(被选择的snaphhot或者frontier) 和 self.target_point(在habitat下具体的目标点)
        如果被选择的目标是个keyframe, 则先会对这个snapshot包含的物体的物体中心求一个均值, 再在这个物体中心均值附近搜索合适的导航点(未被占据, 与观测点连通, 不靠墙)
        如果被选择的目标是个frontier, 则会在frontier的位置搜索合适的导航点(在2D边界内, 不是被占据的, 在被选中的岛屿内)
        '''
        if self.max_point is not None or self.target_point is not None:
            # if the next point is already set
            logging.error(
                f"Error in set_next_navigation_point: the next point is already set: {self.max_point}, {self.target_point}"
            )
            return False
        pts = pos_habitat_to_normal(pts)
        cur_point = planner.normal2voxel(pts)
        self.max_point = choice


        if type(choice) == Object3D:
            object_center = choice.bbox.center
            object_center = planner.normal2voxel(object_center)[:2]
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
                    planner.unoccupied,
                    cur_point=cur_point,
                    dist=cfg.final_observe_distance / planner.voxel_size(),
                )
            else:
                # 给定一个目标点, 找一个 随机的合法合适的观测点, 并且确保目标点和观察点之间没有障碍物
                # [min_dist, max_dist]: 限制观测点和目标点的距离范围 (0.75~1.25)
                target_navigable_point = get_random_observe_point(
                    target_point,
                    planner.unoccupied,
                    min_dist=cfg.final_observe_distance / planner.voxel_size(),
                    max_dist=(cfg.final_observe_distance + 1.5) / planner.voxel_size(),
                )

            # 如果没找到合适的位置, 由于目标物体太远, 位于未探索区域, 所以占据地图里不是未占据状态
            if target_navigable_point is None:
                # 基于 Habitat 的导航器 pathfinder, 在目标点附近(半径1m内)找到一个可行走的, 且高度与机器人当前位置高度相差0.1m以内的点
                # this is usually because the target object is too far, so its surroundings are not detected as unoccupied
                # so we just temporarily use pathfinder to find a navigable point around it
                target_point_normal = (
                    target_point * planner.voxel_size() + planner.vol_origin()[:2]
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

                target_navigable_point = planner.normal2voxel(pos_habitat_to_normal(target_navigable_point_habitat))[:2]
            self.target_point = target_navigable_point
            return True
        # 如果选择的目标是 snapshot
        elif type(choice) == Keyframe:
            # 把snapshot包含的物体的中心坐标转为voxel空间下坐标, 并去重
            obj_centers = [scene.objects_3d[obj_id].bbox.center for obj_id in choice.objects_3d]
            obj_centers = [planner.normal2voxel(center)[:2] for center in obj_centers]
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
                    target_point, planner.unoccupied
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
                        planner.unoccupied,
                        cur_point=cur_point,
                        dist=cfg.final_observe_distance / planner.voxel_size(),
                    )
                else:
                    # 给定一个目标点, 找一个 随机的合法合适的观测点, 并且确保目标点和观察点之间没有障碍物
                    # [min_dist, max_dist]: 限制观测点和目标点的距离范围 (0.75~1.25)
                    target_navigable_point = get_random_observe_point(
                        target_point,
                        planner.unoccupied,
                        min_dist=cfg.final_observe_distance / planner.voxel_size(),
                        max_dist=(cfg.final_observe_distance + 1.5) / planner.voxel_size(),
                    )

                # 如果没找到合适的位置, 由于目标物体太远, 位于未探索区域, 所以占据地图里不是未占据状态
                if target_navigable_point is None:
                    # 基于 Habitat 的导航器 pathfinder, 在目标点附近(半径1m内)找到一个可行走的, 且高度与机器人当前位置高度相差0.1m以内的点
                    # this is usually because the target object is too far, so its surroundings are not detected as unoccupied
                    # so we just temporarily use pathfinder to find a navigable point around it
                    target_point_normal = (
                        target_point * planner.voxel_size() + planner.vol_origin()[:2]
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

                    target_navigable_point = planner.normal2voxel(pos_habitat_to_normal(target_navigable_point_habitat))[:2]
                self.target_point = target_navigable_point
                return True
            else:
                if not random_position:
                    target_point = get_proper_snapshot_observation_point(
                        obj_centers=obj_centers,
                        snapshot_observation_point=planner.normal2voxel(choice.pose[:3, 3]),
                        unoccupied_map=planner.unoccupied,
                        min_obs_dist=cfg.final_observe_distance / planner.voxel_size() - 1,
                        max_obs_dist=cfg.final_observe_distance / planner.voxel_size() + 1,
                    )
                else:
                    target_point = get_random_snapshot_observation_point(
                        obj_centers=obj_centers,
                        snapshot_observation_point=planner.normal2voxel(choice.pose[:3, 3]),
                        unoccupied_map=planner.unoccupied,
                        min_obs_dist=cfg.final_observe_distance / planner.voxel_size() - 1,
                        max_obs_dist=cfg.final_observe_distance / planner.voxel_size() + 1,
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
                not planner.check_within_bnds(next_point.astype(int))
                or planner.occupied[int(next_point[0]), int(next_point[1])]
                or not planner.island[int(next_point[0]), int(next_point[1])]
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
        planner: TSDFPlanner,
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
        cur_point = planner.normal2voxel(pts)

        # 当前位置和目标点能容忍的最大距离(默认都是1m)
        max_dist_from_cur = (
            cfg.max_dist_from_cur_phase_1
            if type(self.max_point) == Frontier
            else cfg.max_dist_from_cur_phase_2
        )  # in phase 2, the step size should be smaller

        # 计算当前位置和目标位置之间的距离, 优先使用 Habitat 的导航路径距离, 如果路径不可达则退化为欧式距离
        # 返回值是距离和路径点
        dist, path_to_target = self.get_distance(
            cur_point[:2], self.target_point, height=pts[2], planner=planner, pathfinder=pathfinder
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
                        next_point = planner.normal2voxel(
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
                    cur_point[:2] + walk_dir * max_dist_from_cur / planner.voxel_size()
                )

                # 根据上面初始的 next_point, 找到合适的 next_point 需同时满足三个条件: 在2D边界内, 在当前岛屿内, 未被占据
                # 如不满足, 则沿着 walk_dir 反方向回退, 每次回退1个voxel, 最多回退1000次
                # 如果找到了, 则则返回这个修正的 next_point, 如果未找到, 返回None
                # ensure next point is valid, otherwise go backward a bit
                try_count = 0
                while (
                    not planner.check_within_bnds(next_point)
                    or not planner.island[
                        int(np.round(next_point[0])), int(np.round(next_point[1]))
                    ]
                    or planner.occupied[
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
            planner.occupied,
            voxel_size=planner.voxel_size(),
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
                object_center = planner.normal2voxel(object_center)[:2]
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
            direction = planner.rad2vector(angle)

        # direction 向量正则化
        direction = direction / np.linalg.norm(direction)

        # 将 next_point 由 voxel坐标系转为真实坐标系
        # Convert back to world coordinates
        next_point_normal = next_point * planner.voxel_size() + planner.vol_origin()[:2]

        # 目标点和当前点 张成向量的角度
        # Find the yaw angle again
        next_yaw = np.arctan2(direction[1], direction[0]) - np.pi / 2

        # path_points默认是 None
        # update the path points
        if path_points is not None:
            # 根据一个新位置 next_point_normal, 更新路径 path_points, 从离该点(next_point_normal)最近的路径段开始继续走
            updated_path_points = planner.update_path_points(
                path_points, next_point_normal
            )
        else:
            updated_path_points = None

        # 设置 next_point 附近 0.7m 的邻域中未被占据的区域为已探索
        # set the surrounding points of the next point as explored
        planner.set_explored_area(next_point, cfg.surrounding_explored_radius)

        # 如果到达目标点了, 更新 self.max_point 和 self.target_point 都为 None. 
        if target_arrived:
            self.max_point = None
            self.target_point = None

        print("target_arrived === {}".format(target_arrived))

        return (
            pos_normal_to_habitat(np.append(next_point_normal, planner.floor_height)),  # next_point 的 habitat 坐标
            next_yaw,       # 目标点和当前点 张成向量的角度
            next_point,     # next_point 的 voxel 坐标
            updated_path_points,    # 更新后的路标点, 这里是None
            target_arrived,         # 是否到达了目标点 (即目标点距离和当前位置距离小于 max_dist_from_cur)
        )