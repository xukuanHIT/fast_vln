import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

import time
import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import time
import json
import logging
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R

from geom import get_cam_intr, get_scene_bnds
from utils import get_pts_angle_aeqa
from habitat_data import HabitatData, pose_habitat_to_tsdf, pos_habitat_to_normal
from map import Map
from map_elements import Keyframe, Object3D
from tsdf_planner import TSDFPlanner, Frontier, SnapShot
from query_vlm_aeqa import query_vlm_for_response
from logger import Logger
from visualization import Visualization, concat_images_opencv
from vlm import VLM


def main(cfg, start_ratio=0.0, end_ratio=1.0):
    # load the default concept graph config
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    questions_list = json.load(open(cfg.questions_list_path, "r"))
    total_questions = len(questions_list)
    # sort the data according to the question id
    questions_list = sorted(questions_list, key=lambda x: x["question_id"])
    logging.info(f"Total number of questions: {total_questions}")
    # only process a subset of the questions
    questions_list = questions_list[
        int(start_ratio * total_questions) : int(end_ratio * total_questions)
    ]
    logging.info(f"number of questions after splitting: {len(questions_list)}")
    logging.info(f"question path: {cfg.questions_list_path}")

    # Initialize the logger
    logger = Logger(
        cfg.output_dir,
        start_ratio,
        end_ratio,
        len(questions_list),
        voxel_size=cfg.tsdf_grid_size,
    )

    scene_map = Map(cfg)
    vlm_model = VLM(cfg)
    vis = Visualization(cfg=cfg)

    questions_list = questions_list[6:7]

    # Run all questions
    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data["question_id"]
        scene_id = question_data["episode_history"]

        question = question_data["question"]
        answer = question_data["answer"]

        print("Task: {}".format(question))
        qa_text_lines = ["Question: {}".format(question)]
        vis.update_chat(qa_text_lines)

        scene_map.update_task(question)

        target_vl_response = vlm_model.parse_target_objects(question, scene_map.get_class_list())
        class_to_add = []
        if target_vl_response is not None:
            class_to_add = scene_map.add_target_class(*target_vl_response)
        else:
            scene_map.set_target_object_with_clip(question)
        parsing_target_lines = scene_map.target_manager.print_information()
        if len(class_to_add) > 0:
            parsing_target_lines += f"Add new classes to detector: {class_to_add}"

        qa_text_lines.append(parsing_target_lines)
        vis.update_chat(qa_text_lines)


        pts, angle = get_pts_angle_aeqa(
            question_data["position"], question_data["rotation"]
        )

        data_generator = HabitatData(cfg)
        data_generator.load_data(scene_id)

        # initialize the TSDF
        tsdf_planner = TSDFPlanner(
                vol_bnds=get_scene_bnds(data_generator.pathfinder, floor_height=pts[1])[0],
                voxel_size=cfg.tsdf_grid_size,
                floor_height=pts[1],
                floor_height_offset=0,
                pts_init=pts,  # 机器人初始位置
                init_clearance=cfg.init_clearance * 2,
                save_visualization=cfg.save_visualization,
            )


        episode_dir, eps_chosen_snapshot_dir, eps_frontier_dir, eps_snapshot_dir = (
            logger.init_episode(
                question_id=question_id,
                init_pts_voxel=tsdf_planner.habitat2voxel(pts)[:2],  # 3D点转为voxel坐标系下位姿
            )
        )
        vlm_model.update_save_dir(eps_chosen_snapshot_dir)

        # run steps
        cnt_step = -1
        frame_index = 1
        found_target_objects = []
        task_success = False
        while cnt_step < cfg.num_step - 1:
            step_start_time = time.perf_counter()
            cnt_step += 1
            logging.info(f"\n== step: {cnt_step}")

            # (1) Observe the surroundings, update the scene graph and occupancy map
            # Determine the viewing angles for the current step
            if cnt_step == 0:
                angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180 # 40 degree
                total_views = 1 + cfg.extra_view_phase_2 # = 7
            else:
                angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180 # 60 degree
                total_views = 1 + cfg.extra_view_phase_1 # = 3
            all_angles = [
                angle + angle_increment * (i - total_views // 2)
                for i in range(total_views)
            ] # [-120, -80, -40, 0, 40, 80, 120], [-60, 0, 60]
            # Let the main viewing angle be the last one to avoid potential overwriting problems
            main_angle = all_angles.pop(total_views // 2) # 取出第中间的那个角度并从list移除此角度
            all_angles.append(main_angle) # 把main_angle放在首位


            vis_images = []
            for view_idx, ang in enumerate(all_angles):
                # For each view
                obs, cam_pose = data_generator.get_observation(pts, ang)

                rgb = obs["color_sensor"]
                depth = obs["depth_sensor"]

                obs_file_name = f"{cnt_step}-view_{view_idx}.png"
                with torch.no_grad():
                    # Concept graph pipeline update
                    annotated_image, _, update_pcds = scene_map.update_scene_graph(
                        image_rgb=rgb, 
                        depth=depth, 
                        intrinsics=cam_intr, 
                        cam_pos=pose_habitat_to_tsdf(cam_pose), 
                        img_path=obs_file_name, 
                        frame_idx=frame_index, 
                        visualization=True)
                    
                    frame_index += 1
                    vis_images.append(cv2.resize(annotated_image, (cfg.img_width // 2, cfg.img_height // 2), interpolation=cv2.INTER_AREA))

                # 更新TSDF volume地图
                # Update depth map, occupancy map
                tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=cam_intr,
                    cam_pose=pose_habitat_to_tsdf(cam_pose),
                    obs_weight=1.0,
                    margin_h=int(cfg.margin_h_ratio * img_height),
                    margin_w=int(cfg.margin_w_ratio * img_width),
                    explored_depth=cfg.explored_depth,
                )

                if cfg.save_visualization:
                    plt.imsave(os.path.join(eps_snapshot_dir, obs_file_name), rgb)


            scene_map.print_map_object_labels()

            # 更新 frontier
            # (3) Update the Frontier Snapshots
            update_success, grid_map_vis = tsdf_planner.update_frontier_map(
                pts=pts,
                cfg=cfg.planner,
                scene=scene_map,
                cnt_step=cnt_step,  # step的index
                save_frontier_image=cfg.save_visualization,
                eps_frontier_dir=eps_frontier_dir,
                prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
                visualization=True,
            )
            if not update_success:
                logging.info("Warning! Update frontier map failed!")
                if cnt_step == 0:  # if the first step fails, we should stop
                    logging.info(
                        f"Question id {question_id} invalid: update_frontier_map failed!"
                    )
                    break
            
            if grid_map_vis is not None:
                grid_map_vis_resized = cv2.resize(
                    grid_map_vis, 
                    (vis_images[0].shape[1], int(grid_map_vis.shape[0] * vis_images[0].shape[1] / grid_map_vis.shape[1]))
                )
                vis_images.append(grid_map_vis_resized)



            cam_pos_normal = pos_habitat_to_normal(pts)
            vis.add_trajectory_point(cam_pos_normal.tolist())
            vis.update_image(annotated_image)
            for object_id in update_pcds[0]:
                vis.update_object(object_id, scene_map.objects_3d[object_id].pcd)
            for object_id in update_pcds[1]:
                vis.delete_object(object_id)

            vis_image_show = concat_images_opencv(vis_images)
            vis.update_image(vis_image_show)


            need_new_target_point = tsdf_planner.max_point is None and tsdf_planner.target_point is None
            detect_new_target_class = len(scene_map.target_manager.valid_target_object_ids()) > 0 and not isinstance(tsdf_planner.max_point, Object3D)
            if need_new_target_point or detect_new_target_class:
                print("start reasoning..........")


                vlm_model.update_step(cnt_step)
                task_success, max_point_choice, choice_image, reason = vlm_model.query_vlm_for_response(
                    question=question,
                    scene=scene_map,
                    tsdf_planner=tsdf_planner,
                    cfg=cfg,
                    verbose=False,
                )


                vis_text_a = "Step {}: {}".format(cnt_step, reason)
                qa_text_lines.append(vis_text_a)
                vis.update_chat(qa_text_lines)


                if need_new_target_point or isinstance(max_point_choice, Object3D):
                    tsdf_planner.max_point = None
                    tsdf_planner.target_point = None

                    # 根据ChatGPT挑选的目标, 来选择可以导航的目标点, 设置self.max_point(被选择的snaphhot或者frontier) 和 self.target_point(在habitat下具体的目标点)
                    # 如果被选择的目标是个snapshot, 则先会对这个snapshot包含的物体的物体中心求一个均值, 再在这个物体中心均值附近搜索合适的导航点(未被占据, 与观测点连通, 不靠墙)
                    # 如果被选择的目标是个frontier, 则会在frontier的位置搜索合适的导航点(在2D边界内, 不是被占据的, 在被选中的岛屿内)
                    # set the vlm choice as the navigation target
                    update_success = tsdf_planner.set_next_navigation_point(
                        choice=max_point_choice, # 目标点
                        pts=pts,  # 当前位置
                        scene=scene_map, # 地图中的object
                        cfg=cfg.planner,
                        pathfinder=data_generator.pathfinder,
                        random_position=False,
                    )
                    if not update_success:
                        logging.info(
                            f"Question id {question_id} invalid: set_next_navigation_point failed!"
                        )
                        break

            # 根据所选的目标位置, 前进一步, 前进的距离不超过 max_dist_from_cur (配置文件为1m)
            # (5) Agent navigate to the target point for one step
            return_values = tsdf_planner.agent_step(
                pts=pts,        # 机器人当前位置
                angle=angle,    # 机器人当前旋转
                pathfinder=data_generator.pathfinder,
                cfg=cfg.planner,
                path_points=None,
            )
            if return_values[0] is None:
                logging.info(f"Question id {question_id} invalid: agent_step failed!")
                break

            # pts和angle为更新后的位置和角度, 即下次获取观测数据的位置和角度
            # pts 为  next_point 的 habitat 坐标, angle 为 目标点和当前点 张成向量的角度
            # update agent's position and rotation
            pts, angle, pts_voxel, _, target_arrived = return_values
            logger.log_step(pts_voxel=pts_voxel)
            logging.info(f"Current position: {pts}, {logger.explore_dist:.3f}")


            # (6) Check if the agent has arrived at the target to finish the question
            if task_success and target_arrived:
                # when the target is a snapshot, and the agent arrives at the target
                # we consider the question is finished and save the chosen target snapshot
                logging.info(
                    f"Question id {question_id} finished after arriving at target!"
                )
                break

            step_end_time = time.perf_counter()
            print(f"step runtime: {step_end_time - step_start_time:.4f} s")

        scene_map.reset()


if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(
        str(cfg.output_dir), f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}.log"
    )

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # Set up the logging format
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg, args.start_ratio, args.end_ratio)