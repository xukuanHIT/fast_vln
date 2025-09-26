import os
import numpy as np
import random
import torch

import supervision as sv
import logging
from collections import Counter
from typing import List, Optional, Tuple, Dict, Union
import copy
from scipy.spatial.transform import Rotation as R
from collections import Counter
import time
import math

import open_clip
from ultralytics import SAM, YOLOWorld
from utils import resize_image

from hierarchy_clustering import SceneHierarchicalClustering

from map_elements import (
    Frame,
    Keyframe,
    Object3D,
    ObjectClasses,
)

from map_utils import (
    compute_clip_features_batched, 
    filter_detections, 
    filter_masks, 
    mask_subtract_contained,
    compute_visual_similarities,
    match_detections_to_objects, 
)

from pointcloud import (
    detections_to_obj_pcd_and_bbox,
    init_process_pcd,
    get_bounding_box,
    compute_overlap_matrix_general,
    denoise_objects,
)


class Map:
    def __init__(
        self,
        cfg,
        graph_cfg,
    ):
        self.cfg = cfg
        # concept graph configuration
        self.cfg_cg = graph_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load object classes
        # maintain a list of object classes
        self.obj_classes = ObjectClasses(
            classes_file_path="",
            bg_classes=self.cfg_cg["bg_classes"],
            skip_bg=self.cfg_cg["skip_bg"],
            class_set=self.cfg["class_set"],
        )

        self.clustering = SceneHierarchicalClustering(
            min_sample_split=0,
            random_state=66,
        )

        # Keyframe, Object
        self.object_id_counter = 1
        self.frames: Dict[int, Frame] = {}
        self.keyframes: Dict[int, Keyframe] = {}
        self.objects_3d: Dict[int, Object3D] = {}


        # load detection and segmentation models
        self.detection_model = YOLOWorld(cfg.yolo_model_name)
        self.detection_model.set_classes(self.obj_classes.get_classes_arr())
        logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

        self.sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
        logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

        clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", "laion2b_s34b_b79k"  # "ViT-H-14", "laion2b_s32b_b79k"
        )
        self.clip_model = clip_model.to(self.device)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        logging.info(f"Load CLIP model successful!")

        self.last_update_observation = None # pose, rgb, depth, detection results
        self.update_num = 0


    def reset(self):
        self.object_id_counter = 1
        self.keyframes.clear()
        self.objects_3d.clear()

    def clear_frames(self):
        self.frames.clear()

    def print_keyframe_objects(self):
        print("================== keyframes ====================")
        for kf_id, kf in self.keyframes.items():
            print("kf id = {}, objects = {}".format(kf_id, kf.objects_3d))
        print("================== end ====================")

    def print_objects_keyframe(self):
        print("================== objects ====================")
        for obj_id, obj in self.objects_3d.items():
            print("obj_id = {}, observers = {}".format(obj_id, obj.observers))
        print("================== end ====================")


    def delete_keyframe(self, frame_id):
        if frame_id not in self.keyframes:
            return 

        object_ids = self.keyframes[frame_id].objects_3d
        for object_id in object_ids:
            self.objects_3d[object_id].observers.discard(frame_id)

        self.keyframes.pop(frame_id)
        return


    def get_all_object_pointcloud(self):
        pcds = [obj.pcd for obj in self.objects_3d.values()]
        return pcds


    def filter_gobs_with_distance(self, pts, gobs):
        '''
        找出距离机器人平面(2D距离)距离比较远的物体, 并去除
        '''
        idx_to_keep = []
        for idx in range(len(gobs["bbox"])):
            if gobs["bbox"][idx] is None:  # point cloud was discarded
                continue

            # habitat中X, Y, Z表示右，上，前
            # get the distance between the object and the current observation point
            if (
                np.linalg.norm(gobs["bbox"][idx].center - pts)
                > self.cfg.scene_graph.obj_include_dist
            ):
                logging.debug(
                    f"Object {gobs['detection_class_labels'][idx]} is too far away, skipping"
                )
                continue
            idx_to_keep.append(idx)

        for attribute in gobs.keys():
            if isinstance(gobs[attribute], str) or attribute == "classes":  # Captions
                continue
            if attribute in ["labels", "edges", "text_feats", "captions"]:
                # Note: this statement was used to also exempt 'detection_class_labels' but that causes a bug. It causes the edges to be misalgined with the objects.
                continue
            elif isinstance(gobs[attribute], list):
                gobs[attribute] = [gobs[attribute][i] for i in idx_to_keep]
            elif isinstance(gobs[attribute], np.ndarray):
                gobs[attribute] = gobs[attribute][idx_to_keep]
            else:
                raise NotImplementedError(f"Unhandled type {type(gobs[attribute])}")

        return gobs


    def merge_obj_matches(
        self,
        new_objects: Object3D,
        match_indices: List[Tuple[int, Optional[int]]],
        obj_classes: ObjectClasses,
        visualization: bool = False
    ):
        added_obj_ids = set()
        update_pcds = [[], []]
        for idx, (detected_obj_id, existing_obj_match_id) in enumerate(match_indices):
            if existing_obj_match_id is None: # 没有在地图中找到匹配的物体，新加一个object到map
                self.objects_3d[detected_obj_id] = new_objects[detected_obj_id]
                added_obj_ids.add(detected_obj_id)

                if visualization:
                    update_pcds[0].append(detected_obj_id)
            else:  # 在地图中找到了向匹配的物体
                if visualization:
                    update_pcds[0].append(existing_obj_match_id)

                # merge detected object into existing object
                self.objects_3d[existing_obj_match_id].merge(
                    other=new_objects[detected_obj_id],
                    downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg_cg["dbscan_eps"],
                    dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                    run_dbscan=False,
                    spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                    obj_classes=obj_classes
                )

                added_obj_ids.add(existing_obj_match_id)

        return added_obj_ids, update_pcds


    def is_map_undate_needed(self, cur_pose, cur_detection_ids):
        if len(cur_detection_ids) < 1:
            logging.info("No detections in this frame, no update")
            return False
        
        if self.last_update_observation is None:
            logging.info("The first frame, update")
            return True

        last_p, last_R, last_detection_ids = self.last_update_observation

        cur_p, cur_R = cur_pose[:3, 3], R.from_matrix(cur_pose[:3, :3])
        if np.linalg.norm((cur_p - last_p)) > 0.5:
            logging.info("Large distance, update")
            return True
        

        delta_R = cur_R.inv() * last_R
        if np.linalg.norm(delta_R.as_rotvec()) > 0.25 * np.pi:
            logging.info("Large angle, update")
            return True

        if len(cur_detection_ids) > len(last_detection_ids):
            logging.info("More object, update")
            return True
        
        cc, cl = Counter(cur_detection_ids), Counter(last_detection_ids)
        is_contained = all(cl[x] >= cc[x] for x in cc)
        if not is_contained:
            logging.info("New object, update")
            return True

        logging.info("Without object, no update")
        return False


    def update_scene_graph(
        self,
        image_rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics,
        cam_pos,
        img_path,
        frame_idx,
        visualization=False,
    ) -> Tuple[np.ndarray, List[int], Optional[int]]:
        # return annotated image; the detected object ids in current frame; the object id of the target object (if detected)

        # set up object_classes first
        obj_classes = self.obj_classes

        time0 = time.perf_counter()

        # 物体检测 yolo-world
        # Detect objects,
        results = self.detection_model.predict(image_rgb, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detection_class_labels = [
            f"{obj_classes.get_classes_arr()[class_id]}"
            for class_idx, class_id in enumerate(detection_class_ids)
        ]
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        time1 = time.perf_counter()


        print("detection_class_labels = {}".format(detection_class_labels))

        filtered_xyxy, filtered_confidences, filtered_class_ids, filtered_class_labels = filter_detections(
            image=image_rgb,
            xyxy=xyxy_np, 
            class_ids=detection_class_ids, 
            confidences=confidences, 
            class_labels=detection_class_labels,
            confidence_threshold=self.cfg.object_detection_confidence_threshold,
            min_area_ratio=self.cfg.object_detection_min_area_ratio,
            skip_bg=self.cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
        )

        print("filtered_class_labels = {}".format(filtered_class_labels))


        time1_5 = time.perf_counter()

        processed_rgb = None
        if len(filtered_xyxy) > 0:
            cam_hov = 2 * math.atan(float(image_rgb.shape[1])/intrinsics[0, 0])
            processed_rgb = resize_image(image_rgb, self.cfg.prompt_h, self.cfg.prompt_w)
            frame = Frame(frame_id=frame_idx, hov=cam_hov, image=processed_rgb, image_path=img_path, pose=cam_pos, class_ids=set(filtered_class_ids))
            self.frames[frame_idx] = frame


        # 可视化
        # create a Detection object for visualization
        annotated_image = image_rgb.copy()
        if len(filtered_class_ids) > 0:
            visualize_captions = []
            for i in range(len(filtered_class_ids)):
                visualize_captions.append(
                    f"{filtered_class_ids[i]} {filtered_class_labels[i]} {filtered_confidences[i]:.3f}"
                )

            det_visualize = sv.Detections(xyxy=filtered_xyxy, confidence=filtered_confidences, class_id=filtered_class_ids)
            det_visualize.data["class_name"] = visualize_captions
            BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
            LABEL_ANNOTATOR = sv.LabelAnnotator(
                text_thickness=1, text_scale=0.25, text_color=sv.Color.BLACK
            )
            annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(
                annotated_image, det_visualize
            )
            annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, det_visualize)


        if not self.is_map_undate_needed(cam_pos, filtered_class_ids):
            return annotated_image, [], [[], []]

        save_filtered_class_ids = filtered_class_ids.copy()

        time2 = time.perf_counter()


        # 基于检测结果分割，SAM
        # if there are detections,
        # Get Masks Using SAM or MobileSAM
        # UltraLytics SAM
        if len(filtered_xyxy) != 0:
            sam_out = self.sam_predictor.predict(
                image_rgb, bboxes=torch.from_numpy(filtered_xyxy), verbose=False
            )
            masks_tensor = sam_out[0].masks.data

            masks_np = masks_tensor.cpu().numpy()
        else:
            masks_np = np.empty((0, *image_rgb.shape[:2]), dtype=np.float64)


        time3 = time.perf_counter()


        keep_flag = filter_masks(
            image=image_rgb,
            masks=masks_np,
            mask_area_threshold=self.cfg.mask_area_threshold,
            mask_area_ratio=self.cfg.mask_area_ratio,
            mask_iou_threshold=self.cfg.mask_iou_threshold,
        )


        time4 = time.perf_counter()

        filtered_class_labels = [class_label for class_label, keep in zip(filtered_class_labels, keep_flag) if keep]
        filtered_xyxy, filtered_confidences, filtered_class_ids, filtered_masks = \
            filtered_xyxy[keep_flag], filtered_confidences[keep_flag], filtered_class_ids[keep_flag], masks_np[keep_flag]

        if len(filtered_xyxy) == 0:  # no detections, skip
            logging.debug("No detections in this frame")
            return annotated_image, [], [[], []]

        # extract CLIP features 
        image_crops, image_feats, text_feats = compute_clip_features_batched(
            image_rgb,
            filtered_xyxy,
            filtered_class_ids,
            self.clip_model,
            self.clip_preprocess,
            self.clip_tokenizer,
            obj_classes.get_classes_arr(),
            self.device,
        )

        time5 = time.perf_counter()


        gobs = {
            # add new uuid for each detection
            "xyxy": filtered_xyxy,
            "confidence": filtered_confidences,
            "class_id": filtered_class_ids,
            "mask": filtered_masks,
            "classes": obj_classes.get_classes_arr(),
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "detection_class_labels": detection_class_labels,
        }


        if len(gobs["mask"]) == 0:  # no detections in this frame
            logging.debug("No detections left after filter_gobs")
            return annotated_image, [], [[], []]

        # 找出大bbox包含小bbox的情况, 对于这些, 去除大bbox对应mask中小bbox对应的mask
        # this helps make sure things like pillows on couches are separate objects
        gobs["mask"] = mask_subtract_contained(gobs["xyxy"], gobs["mask"])

        time6 = time.perf_counter()

        # 2D -> 3D
        # obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
        obj_pcds_and_bboxes = detections_to_obj_pcd_and_bbox(
            depth_array=depth,
            masks=gobs["mask"],
            cam_K=intrinsics[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=cam_pos,
            min_points_threshold=self.cfg.min_points_threshold,
            spatial_sim_type=self.cfg.spatial_sim_type,
            obj_pcd_max_points=self.cfg.obj_pcd_max_points,
            device=self.device,
        )

        time7 = time.perf_counter()

        for obj in obj_pcds_and_bboxes:
            if obj:
                # 降采样和去噪
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg_cg["dbscan_eps"],
                    dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                )
                # 更新bbox,是3D的紧致包围框
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                    pcd=obj["pcd"],
                )
        # all()是有1个false就返回false
        # if the list is all None, then skip
        if all([obj is None for obj in obj_pcds_and_bboxes]):
            logging.debug("All objects are None in obj_pcds_and_bboxes")
            return annotated_image, [], [[], []]

        # add pcds and bboxes to gobs
        gobs["bbox"] = [
            obj["bbox"] if obj is not None else None for obj in obj_pcds_and_bboxes
        ]
        gobs["pcd"] = [
            obj["pcd"] if obj is not None else None for obj in obj_pcds_and_bboxes
        ]

        time8 = time.perf_counter()


        # TODO 需要先根据距离筛选, 增加label, 表明哪些未被三角化, 并把mask和观测位置加进去

        # 找出距离机器人平面(2D距离)距离比较远的物体, 并去除
        # pts为机器人的3d位置。
        # filter out objects that are far away
        gobs = self.filter_gobs_with_distance(cam_pos[:3, 3], gobs)
        if len(gobs["mask"]) == 0:
            logging.debug(
                "No detections left after filter_gobs_with_distance"
            )
            return annotated_image, [], [[], []]


        time9 = time.perf_counter()


        ### add objects to the map
        # construct new object
        new_objects = {}
        for mask_idx in range(len(gobs["mask"])):
            if gobs["pcd"][mask_idx] is None:  # point cloud was discarded
                continue
            
            new_obj_id = self.object_id_counter
            curr_class_idx = gobs["class_id"][mask_idx]
            curr_class_name = gobs["classes"][curr_class_idx]
            new_obj = Object3D(
                object_id=new_obj_id,
                class_name=curr_class_name,
                class_id=[curr_class_idx],
                confidence=[gobs["confidence"][mask_idx]],
                pcd=gobs["pcd"][mask_idx],
                bbox=gobs["bbox"][mask_idx],
                clip_ft=torch.from_numpy(gobs["image_feats"][mask_idx]),
                observers={frame_idx},
            )
            new_objects[new_obj_id] = new_obj
            self.object_id_counter += 1

        update_pcds = [[], []]
        frame_object_ids = set()
        if len(self.objects_3d) == 0:
            self.objects_3d = new_objects
            frame_object_ids = set(new_objects.keys())

            if visualization:
                for obj in new_objects.values():
                    update_pcds[0].append(obj.object_id)
        else:
            ### compute similarities and then merge
            # spatial similarity
            points_new = [np.asarray(obj_pcd.points, dtype=np.float32) for obj_pcd in gobs["pcd"]] 
            points_map = [np.asarray(obj.pcd.points, dtype=np.float32) for obj in self.objects_3d.values()]  
            bbox_new = [np.asarray(new_bbox.get_box_points()) for new_bbox in gobs["bbox"]] 
            bbox_new = np.stack(bbox_new, axis=0)
            bbox_map = [np.asarray(obj.bbox.get_box_points()) for obj in self.objects_3d.values()] 
            bbox_map = np.stack(bbox_map, axis=0)
            spatial_sim = compute_overlap_matrix_general(
                points_a=points_map,
                points_b=points_new,
                bbox_a=torch.from_numpy(bbox_map),
                bbox_b=torch.from_numpy(bbox_new),
                downsample_voxel_size=self.cfg["downsample_voxel_size"],
            )
            spatial_sim = spatial_sim.T

            # visual similarity
            features_new = torch.from_numpy(gobs["image_feats"])
            features_map = [obj.clip_ft for obj in self.objects_3d.values()]
            features_map = torch.stack(features_map, dim=0)
            visual_sim = compute_visual_similarities(features_new, features_map)

            agg_sim = (1 + self.cfg["phys_bias"]) * spatial_sim + (1 - self.cfg["phys_bias"]) * visual_sim


            # 对于每个新detect的object, 找到地图中与其相似度最高的object, 如果相似度分数超过阈值，则匹配
            # 返回值是一个二维list，第一列是新检测的物体id,第二列是与之匹配的地图中的物体id或者None
            # Perform matching of detections to existing objects
            match_indices = match_detections_to_objects(
                agg_sim=agg_sim,
                detection_threshold=self.cfg_cg["sim_threshold"],
                existing_obj_ids=list(self.objects_3d.keys()),
                detected_obj_ids=list(new_objects.keys()),
            )


            # 将新检测的物体和地图中的物体合并
            # 返回值 visualize_captions 包含各个物体的: object id, 类别名字, confidence, 被检测到的次数
            # 返回值 target_obj_id 为更新后的target id如果target在本帧被检测到，且也和地图中的物体匹配上
            # 返回值 added_obj_ids 为本帧新检测到的，地推中没有的object的id列表
            # Now merge the detected objects into the existing objects based on the match indices
            added_obj_ids, update_pcds = self.merge_obj_matches(
                new_objects=new_objects,
                match_indices=match_indices,
                obj_classes=obj_classes,
                visualization=visualization,
            )

            frame_object_ids = added_obj_ids

        time10 = time.perf_counter()

        # construct keyframe
        cam_hov = 2 * math.atan(float(image_rgb.shape[1])/intrinsics[0, 0])
        curr_keyframe = Keyframe(frame_id=frame_idx, hov=cam_hov, image=processed_rgb, image_path=img_path, pose=cam_pos, objects_3d=frame_object_ids)
        self.keyframes[frame_idx] = curr_keyframe

        self.last_update_observation = [cam_pos[:3, 3], R.from_matrix(cam_pos[:3, :3]), save_filtered_class_ids]
        self.update_num += 1
        if self.update_num > 10:
            self.periodic_cleanup_objects()
            self.update_num = 1

        time11 = time.perf_counter()

        # print("==================== time  ===========================")
        # print("object_detetcion = {}".format(time1-time0))
        # print("visualization = {}".format(time1_5-time1))
        # print("filter_detections = {}".format(time2-time1_5))
        # print("segmentation = {}".format(time3-time2))
        # print("filter_masks = {}".format(time4-time3))
        # print("compute_clip_features_batched = {}".format(time5-time4))
        # print("mask_subtract_contained = {}".format(time6-time5))
        # print("detections_to_obj_pcd_and_bbox = {}".format(time7-time6))
        # print("init_process_pcd = {}".format(time8-time7))
        # print("filter_gobs_with_distance = {}".format(time9-time8))
        # print("add object to map = {}".format(time10-time9))
        # print("periodic_cleanup_objects======= = {}".format(time11-time10))
        # print("sum time =========== {}".format(time11-time0))
        # print("===============================================")

        # print("Update this frame")

        return annotated_image, frame_object_ids, update_pcds
    

    def periodic_cleanup_objects(self):
        '''
        frame_idx: frame 全局 id
        pts: 机器人位置
        该函数的作用是每隔一些帧, 就周期性的为地图中物体点云去噪, 去除一些冗余的关键帧
        '''
        ### Perform post-processing periodically if told so

        # 每隔固定帧，使用dbscan对地图中的物体点云降噪, 降噪后会更新3d bbox
        # Denoising
        self.objects_3d = denoise_objects(
            downsample_voxel_size=self.cfg["downsample_voxel_size"],
            dbscan_remove_noise=self.cfg["dbscan_remove_noise"],
            dbscan_eps=self.cfg["dbscan_eps"],
            dbscan_min_points=self.cfg["dbscan_min_points"],
            spatial_sim_type=self.cfg["spatial_sim_type"],
            device=self.device,
            objects=self.objects_3d,
        )

        # Remove redundant keyframes
        frame_id_to_object_num = {}
        for frame_id, keyframe in self.keyframes.items():
            frame_id_to_object_num[frame_id] = len(keyframe.objects_3d)

        sorted_frame_ids = sorted(frame_id_to_object_num, key=lambda k : frame_id_to_object_num[k], reverse=True)
        object_set = set()
        for frame_id in sorted_frame_ids:
            keyframe = self.keyframes[frame_id]
            if keyframe.objects_3d.issubset(object_set):
                # delete keyframe and observers of objects
                self.delete_keyframe(frame_id)
            else:
                object_set.update(keyframe.objects_3d)
