import logging
from typing import Tuple, Optional, Union

from eval_utils_gpt_aeqa import explore_step
from tsdf_planner import TSDFPlanner, SnapShot, Frontier
from map import Map

def query_vlm_for_response(
    question: str,
    scene: Map,
    tsdf_planner: TSDFPlanner,
    cfg,
    verbose: bool = False,
) -> Optional[Tuple[Union[SnapShot, Frontier], str, int]]:
    # prepare input for vlm
    step_dict = {}

    # prepare question
    step_dict["question"] = question

    # prepare keyframes
    keyframe_objects, keyframe_images = {}, {}
    for _, obj in scene.objects_3d.items():
        if obj.class_name not in keyframe_objects:
            keyframe_objects[obj.class_name] = set()

        keyframe_objects[obj.class_name].update(obj.observers)

        # if obj.class_name in keyframe_objects:
        #     keyframe_objects[obj.class_name].update(obj.observers)
        # else:
        #     keyframe_objects[obj.class_name] = obj.observers

    for kf_id, kf in scene.keyframes.items():
        keyframe_images[kf_id] = kf.image

    # prepare frontiers
    frontier_objects, frontier_images, frontier_id_to_index = {}, {}, {}
    for i, frontier in enumerate(tsdf_planner.frontiers):
        object_ids = frontier.frame.class_ids
        class_labels = [scene.obj_classes.get_classes_arr()[obj_id] for obj_id in object_ids]
        for class_label in class_labels:
            if class_label in frontier_objects:
                frontier_objects[class_label].add(frontier.frontier_id)
            else:
                frontier_objects[class_label] = set([frontier.frontier_id])

        frontier_images[frontier.frontier_id] = frontier.frame.image
        frontier_id_to_index[frontier.frontier_id] = i


    step_dict["keyframe_objects"] = keyframe_objects
    step_dict["keyframe_images"] = keyframe_images
    step_dict["frontier_objects"] = frontier_objects
    step_dict["frontier_images"] = frontier_images

    outputs, frontier_ids, keyframe_ids, reason = explore_step(
        step_dict, cfg, verbose=verbose
    )


    if outputs is None:
        logging.error(f"explore_step failed and returned None")
        return None
    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")

    # 解析 vlm 给出的结果, 即目标类型和目标id
    # parse returned results
    try:
        target_type, target_index = outputs.split(" ")[0], outputs.split(" ")[1]
        logging.info(f"Prediction: {target_type}, {target_index}")
    except:
        logging.info(f"Wrong output format, failed!")
        return None

    # 目标类型为 snapshot 或者 frontier
    if target_type not in ["snapshot", "frontier"]:
        logging.info(f"Wrong target type: {target_type}, failed!")
        return None

    # 如果目标是snapshot, 则返回snapshot(是SnapShot类实例), ChatGPT给出的reason, 和prefiltering后剩余的snapshot的数量
    if target_type == "snapshot":
        if int(target_index) < 0 or int(target_index) >= len(keyframe_ids):
            logging.info(
                f"Target index can not match real objects: {target_index}, failed!"
            )
            return None
        target_keyframe_id = keyframe_ids[int(target_index)]
        logging.info(f"The index of target snapshot {target_keyframe_id}")

        # get the target snapshot
        if target_keyframe_id not in scene.keyframes:
            logging.info(
                f"Predicted snapshot target index is not in the map: {target_keyframe_id}, failed!"
            )
            return None

        pred_target_snapshot = scene.keyframes[target_keyframe_id]

        return pred_target_snapshot, reason
    # 如果目标是frontier, 则返回frontier(是Frontier类实例), ChatGPT给出的reason, 和prefiltering后剩余的snapshot的数量
    else:  # target_type == "frontier"
        target_frontier_id = frontier_ids[int(target_index)]
        target_frontier_index = frontier_id_to_index[target_frontier_id]
        if target_frontier_index < 0 or target_frontier_index >= len(tsdf_planner.frontiers):
            logging.info(
                f"Predicted frontier target index out of range: {target_index}, failed!"
            )
            return None
        target_point = tsdf_planner.frontiers[target_frontier_index].position
        logging.info(f"Next choice: Frontier at {target_point}")
        pred_target_frontier = tsdf_planner.frontiers[target_frontier_index]

        return pred_target_frontier, reason
