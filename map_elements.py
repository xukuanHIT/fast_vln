import os
import numpy as np

import open3d as o3d
from typing import List, Tuple, Set, Dict, Union, Optional
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from collections import Counter

from pointcloud import process_pcd, get_bounding_box


@dataclass
class FrameDetections:
    confidences: Optional[np.ndarray] = field(default=None)
    bbox: Optional[np.ndarray] = field(default=None)
    class_labels: List[str] = field(default_factory=list)
    class_label_set: Set[str] = field(default_factory=set)

    def __post_init__(self):
        # 确保 numpy array 类型正确
        if self.confidences is not None:
            self.confidences = np.array(self.confidences)
        if self.bbox is not None:
            self.bbox = np.array(self.bbox)



@dataclass
class Frame:
    frame_id: int
    hov: float
    image_path: str
    image: Optional[np.ndarray] = field(default=None)
    pose: Optional[np.ndarray] = field(default=None)

    # detection results
    detections: FrameDetections = None

    clip_ft: np.ndarray = None

    def __eq__(self, other):
        raise NotImplementedError("Cannot compare Keyframe objects.")


@dataclass
class Keyframe:
    frame_id: int
    hov: float
    image_path: str
    image: Optional[np.ndarray] = field(default=None)
    pose: Optional[np.ndarray] = field(default=None)

    # detection results
    detections: FrameDetections = None

    objects_3d: Set = field(default_factory=set)
    position: Optional[np.ndarray] = field(default=None) # average center of object 3d bbox

    def __eq__(self, other):
        raise NotImplementedError("Cannot compare Keyframe objects.")
    

@dataclass
class Object3D:
    object_id: int
    # class_name: str
    # class_id: List = field(default_factory=list)
    class_labels: List[str] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)

    position: Optional[o3d.geometry.PointCloud] = None

    pcd: Optional[o3d.geometry.PointCloud] = None
    bbox: Optional[o3d.geometry.OrientedBoundingBox] = None
    clip_ft: Optional[np.ndarray] = field(default=None)


    observers: Set = field(default_factory=set)

    def __eq__(self, other):
        raise NotImplementedError("Cannot compare Object3D objects.")
    

    # def get_class_id(self):
    #     class_id_counter = Counter(self.class_id)
    #     return class_id_counter.most_common(1)[0][0]

    def get_class_label(self):
        class_label_counter = Counter(self.class_labels)
        return class_label_counter.most_common(1)[0][0]
    

    def get_confidence(self):
        class_labels = np.array(self.class_labels)
        confidences = np.array(self.confidence)
        most_common_class_label = self.get_class_label()
        return confidences[class_labels==most_common_class_label].max()


    def merge(self, other, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, 
                dbscan_min_points, run_dbscan, spatial_sim_type, obj_classes):
        """
        合并另一个 Object3D 到当前对象
        """
        if self.pcd is not None and other.pcd is not None:
            self.pcd += other.pcd
            self.pcd = process_pcd(
                self.pcd,
                downsample_voxel_size,
                dbscan_remove_noise,
                dbscan_eps,
                dbscan_min_points,
                run_dbscan,
            )
            self.bbox = get_bounding_box(spatial_sim_type, self.pcd)
        elif self.pcd is None:
            self.pcd = other.pcd
            self.bbox = other.bbox

        
        if self.clip_ft is not None and other.clip_ft is not None:
            n_det, n_det_other = len(self.class_labels), len(other.class_labels)
            self.clip_ft = (self.clip_ft * n_det + other.clip_ft * n_det_other) / (n_det + n_det_other)
        elif self.clip_ft is None:
            self.clip_ft = other.clip_ft

        # self.class_id.extend(other.class_id)
        self.class_labels.extend(other.class_labels)
        self.confidence.extend(other.confidence)

        self.observers |= other.observers

        # # fix the class name by adopting the most popular class name
        # most_common_class_id = self.get_class_id()
        # most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
        # self.class_name = most_common_class_name

        return self
    


@dataclass
class ClassToIds:
    # key: target class, value: set of keyframe/object ids
    class_to_ids: Dict[str, Set[int]] = field(default_factory=dict)

    def add_class_set_for_id(self, class_set: Set[str], id: int):
        for class_label in class_set:
            if class_label not in self.class_to_ids:
                self.class_to_ids[class_label] = set()
            self.class_to_ids[class_label].add(id)

    def delete_class_set_for_id(self, class_set: Set[str], id: int):
        for class_label in class_set:
            if class_label in self.class_to_ids:
                self.class_to_ids[class_label].discard(id)

    def add_class_for_id(self, class_label, id):
        if class_label not in self.class_to_ids:
            self.class_to_ids[class_label] = set()
        self.class_to_ids[class_label].add(id)

    def delete_class_for_id(self, class_label, id):
        if class_label in self.class_to_ids:
            self.class_to_ids[class_label].discard(id)


@dataclass
class TargetManager:
    # Task/Question clip feature
    task_clip_ft: Optional[np.ndarray] = field(default=None)

    # target and relevant objects
    target_class_set: set = field(default_factory=set)
    relevant_class_set: set = field(default_factory=set)

    # Reason of selecting target/relevant objects
    reason_of_selecting_target: str = None
    reason_of_selecting_relevant: str = None

    # key: target class, value: set of keyframe id
    target_in_keyframe: ClassToIds = field(default_factory=ClassToIds)
    relevant_in_keyframe: ClassToIds = field(default_factory=ClassToIds)

    # key: target class, value: set of object id
    target_in_object: ClassToIds = field(default_factory=ClassToIds)
    relevant_in_object: ClassToIds = field(default_factory=ClassToIds)

    # blacklist
    keyframe_blacklist: Set[int] = field(default_factory=set)
    object_blacklist: Set[int] = field(default_factory=set)


    def print_information(self):
        info = f"Target classes: {self.target_class_set}. "
        if self.reason_of_selecting_target is not None:
            info += self.reason_of_selecting_target
        info += "\n"

        info += f"Relevant classes: {self.relevant_class_set}. "
        if self.reason_of_selecting_relevant is not None:
            info += self.reason_of_selecting_relevant
        info += "\n"
        return info
    

    def add_keyframes_to_blacklist(self, keyframe_ids):
        self.keyframe_blacklist.update(keyframe_ids)

    def add_objects_to_blacklist(self, object_ids):
        self.object_blacklist.update(object_ids)


    def get_all_target_objects(self):
        target_object_ids = set()
        for _, object_ids in self.target_in_object.class_to_ids.items():
            target_object_ids.update(object_ids)
        print("self.target_in_object.class_to_ids.keys() = {}".format(self.target_in_object.class_to_ids.keys()))
        print("self.object_blacklist = {}".format(self.object_blacklist))
        return target_object_ids - self.object_blacklist
    

    def get_all_relevant_object(self):
        relevant_object_ids = set()
        for _, object_ids in self.relevant_in_object.class_to_ids.items():
            relevant_object_ids.update(object_ids)

        return relevant_object_ids - self.object_blacklist



    def add_keyframe(self, keyframe: Keyframe):
        kf_id = keyframe.frame_id

        target_detected = self.target_class_set & keyframe.detections.class_label_set
        self.target_in_keyframe.add_class_set_for_id(target_detected, kf_id)
                
        relevant_detected = self.relevant_class_set & keyframe.detections.class_label_set
        self.relevant_in_keyframe.add_class_set_for_id(relevant_detected, kf_id)



    def delete_keyframe(self, keyframe: Keyframe):
        kf_id = keyframe.frame_id

        self.target_in_keyframe.delete_class_set_for_id(keyframe.detections.class_label_set, kf_id)
        self.relevant_in_keyframe.delete_class_set_for_id(keyframe.detections.class_label_set, kf_id)


    def add_object(self, object: Object3D):
        object_label, object_id = object.get_class_label(), object.object_id
        if object_label in self.target_class_set:
            self.target_in_object.add_class_for_id(object_label, object_id)

        if object_label in self.relevant_class_set:
            self.relevant_in_object.add_class_for_id(object_label, object_id)


    def delete_object(self, object: Object3D):
        object_label, object_id = object.get_class_label(), object.object_id
        self.target_in_object.delete_class_for_id(object_label, object_id)
        self.relevant_in_object.delete_class_for_id(object_label, object_id)


    def valid_target_object_ids(self):
        object_id_set = set()
        for _, object_ids in self.target_in_object.class_to_ids.items():
            object_id_set.update(set(object_ids))

        object_id_set -= self.object_blacklist
        return object_id_set


class ObjectClasses:
    """
    Manages object classes and their associated colors, allowing for exclusion of background classes.

    This class facilitates the creation or loading of a color map from a specified file containing
    class names. It also manages background classes based on configuration, allowing for their
    inclusion or exclusion. Background classes are ["wall", "floor", "ceiling"] by default.

    Attributes:
        classes_file_path (str): Path to the file containing class names, one per line.

    Usage:
        obj_classes = ObjectClasses(classes_file_path, skip_bg=True)
        model.set_classes(obj_classes.get_classes_arr())
        some_class_color = obj_classes.get_class_color(index or class_name)
    """

    def __init__(self, classes_file_path, bg_classes, skip_bg, class_set):
        self.classes_file_path = Path(classes_file_path)
        self.bg_classes = bg_classes
        self.skip_bg = skip_bg

        assert class_set in [
            "hm3d",
            "scannet200",
            "yolo_finetune",
        ], f"Invalid class set: {class_set}"
        self.class_set = class_set

        self.base_classes = self._load_or_create_colors()
        self.extra_classes = []

    def _load_or_create_colors(self):
        if self.class_set == "hm3d":
            # normally load hm3d class
            with open(self.classes_file_path, "r") as f:
                all_lines = [cls.strip() for cls in f.readlines()][1:]
                all_classes = [
                    line.split(",")[2].replace('"', "") for line in all_lines
                ]
                all_classes = list(set(all_classes))
                all_classes = [cls for cls in all_classes if cls != "unknown"]
                logging.info(
                    f"Loaded {len(all_classes)} classes from hm3d: {self.classes_file_path}!!!"
                )

            # Filter classes based on the skip_bg parameter
            if self.skip_bg:
                all_classes = [cls for cls in all_classes if cls not in self.bg_classes]

        elif self.class_set == "scannet200":
            # load scannet 200 class
            self.classes_file_path = Path("data/scannet200_classes.txt")
            with open(self.classes_file_path, "r") as f:
                all_lines = [cls.strip() for cls in f.readlines()]
                all_classes = list(set(all_lines))
                all_classes = [cls for cls in all_classes if cls != "unknown"]
                logging.info(
                    f"Loaded {len(all_classes)} classes from scannet 200: {self.classes_file_path}!!!"
                )

        else:
            # load finetune yolo class
            self.classes_file_path = "yolo_finetune/class_id_to_class_name.json"
            class_id_to_class_name = json.load(open(self.classes_file_path, "r"))
            all_classes = list(class_id_to_class_name.values())
            logging.info(
                f"Loaded {len(all_classes)} classes from yolo finetune class: {self.classes_file_path}!!!"
            )

        return all_classes

    def update_extra_classes(self, extra_classes):
        self.extra_classes = extra_classes


    def add_extra_classes(self, extra_classes):
        extra_classes = set(extra_classes) | set(self.extra_classes)
        self.extra_classes = list(extra_classes)


    def get_classes_arr(self):
        """
        Returns the list of class names, excluding background classes if configured to do so.
        """
        return self.base_classes + self.extra_classes

    def get_bg_classes_arr(self):
        """
        Returns the list of background class names, if configured to do so.
        """
        return self.bg_classes