import os
import numpy as np

import open3d as o3d
from typing import List, Tuple, Set, Dict, Union
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from collections import Counter

from pointcloud import process_pcd, get_bounding_box

@dataclass
class Frame:
    frame_id: int
    hov: float
    image_path: str
    image: np.ndarray = None
    pose: np.ndarray = None

    # detection results
    class_ids: Set = field(default_factory=set)

    def __eq__(self, other):
        raise NotImplementedError("Cannot compare Keyframe objects.")


@dataclass
class Keyframe:
    frame_id: int
    hov: float
    image_path: str
    image: np.ndarray = None
    pose: np.ndarray = None

    objects_3d: Set = field(default_factory=set)
    position: np.ndarray = None # average center of object 3d bbox

    def __eq__(self, other):
        raise NotImplementedError("Cannot compare Keyframe objects.")
    

@dataclass
class Object3D:
    object_id: int
    class_name: str
    class_id: List = field(default_factory=list)
    confidence: List = field(default_factory=list)
    pcd: o3d.geometry.PointCloud = None      
    bbox: o3d.geometry.OrientedBoundingBox = None
    clip_ft: np.ndarray = None
    
    observers: Set = field(default_factory=set)

    def __eq__(self, other):
        raise NotImplementedError("Cannot compare Object3D objects.")
    

    def get_class_id(self):
        class_id_counter = Counter(self.class_id)
        return class_id_counter.most_common(1)[0][0]
    

    def get_confidence(self):
        class_ids = np.array(self.class_id)
        confidences = np.array(self.confidence)
        most_common_class_id = self.get_class_id()
        return confidences[class_ids==most_common_class_id].max()


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
            n_det, n_det_other = len(self.class_id), len(other.class_id)
            self.clip_ft = (self.clip_ft * n_det + other.clip_ft * n_det_other) / (n_det + n_det_other)
        elif self.clip_ft is None:
            self.clip_ft = other.clip_ft

        self.class_id.extend(other.class_id)
        self.confidence.extend(other.confidence)

        self.observers |= other.observers

        # fix the class name by adopting the most popular class name
        most_common_class_id = self.get_class_id()
        most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
        self.class_name = most_common_class_name

        return self
    

    



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

        self.classes = self._load_or_create_colors()

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

    def get_classes_arr(self):
        """
        Returns the list of class names, excluding background classes if configured to do so.
        """
        return self.classes

    def get_bg_classes_arr(self):
        """
        Returns the list of background class names, if configured to do so.
        """
        return self.bg_classes