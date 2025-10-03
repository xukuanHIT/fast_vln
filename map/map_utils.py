import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.distance import cosine
from typing import List, Optional, Tuple, Dict


def resize_image(image, target_h, target_w):
    # image: np.array, h, w, c
    image = Image.fromarray(image)
    image = image.resize((target_w, target_h))
    return np.array(image)


def object_id_to_color(object_id):
    np.random.seed(object_id)            # 保证同 id 颜色固定
    color = np.random.rand(3)            # RGB 随机颜色
    return color


def resize_gobs(gobs, image):

    # If the shapes are the same, no resizing is necessary
    if gobs["mask"].shape[1:] == image.shape[:2]:
        return gobs

    new_masks = []

    for mask_idx in range(len(gobs["xyxy"])):
        # TODO: rewrite using interpolation/resize in numpy or torch rather than cv2
        mask = gobs["mask"][mask_idx]
        # Rescale the xyxy coordinates to the image shape
        x1, y1, x2, y2 = gobs["xyxy"][mask_idx]
        x1 = round(x1 * image.shape[1] / mask.shape[1])
        y1 = round(y1 * image.shape[0] / mask.shape[0])
        x2 = round(x2 * image.shape[1] / mask.shape[1])
        y2 = round(y2 * image.shape[0] / mask.shape[0])
        gobs["xyxy"][mask_idx] = [x1, y1, x2, y2]

        # Reshape the mask to the image shape
        mask = cv2.resize(
            mask.astype(np.uint8),
            image.shape[:2][::-1],
            interpolation=cv2.INTER_NEAREST,
        )
        mask = mask.astype(bool)
        new_masks.append(mask)

    if len(new_masks) > 0:
        gobs["mask"] = np.asarray(new_masks)

    return gobs



# @profile
def compute_clip_features_batched(
    image,
    xyxy,
    class_ids,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    classes,
    device,
    prompt_h=None,
    prompt_w=None,
):
    '''
    为detection出的每个bbox生成clip feature, clip的input是bbox外扩20的像素图像patch。
    返回的image_feats是各个bbox的clip features, text_feats是空list
    '''

    image = Image.fromarray(image)
    padding = 20  # Adjust the padding amount as needed

    image_crops = []
    preprocessed_images = []
    text_tokens = []

    original_width, original_height = image.size
    if prompt_h is not None and prompt_w is not None:
        scale_w, scale_h = prompt_w / original_width, prompt_h / original_height
    else:
        scale_w, scale_h = 1, 1

    # Prepare data for batch processing
    for idx in range(len(xyxy)):
        # 根据detection结果，四周外扩20个像素
        x_min, y_min, x_max, y_max = xyxy[idx]
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # get the image crop without padding
        image_crops.append(image.crop((x_min, y_min, x_max, y_max)))

        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        # clip预处理，打包数据
        preprocessed_image = clip_preprocess(
            image.crop((x_min, y_min, x_max, y_max))
        ).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)

        class_id = class_ids[idx]
        text_tokens.append(classes[class_id])

    # Convert lists to batches
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
    text_tokens_batch = clip_tokenizer(text_tokens).to(device)

    # 使用clip，为每个detection的bbox生成feature
    # Batch inference
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # text_features = clip_model.encode_text(text_tokens_batch)
        # text_features /= text_features.norm(dim=-1, keepdim=True)

    # Convert to numpy
    image_feats = image_features.cpu().numpy()
    # text_feats = text_features.cpu().numpy()
    # image_feats = []
    text_feats = []

    # resize image crops
    if prompt_h is not None and prompt_w is not None:
        for i in range(len(image_crops)):
            crop_w, crop_h = image_crops[i].size
            image_crops[i] = image_crops[i].resize(
                (int(crop_w * scale_w), int(crop_h * scale_h))
            )

    return image_crops, image_feats, text_feats



def filter_detections(
    image,
    xyxy, 
    class_ids,
    confidences,
    class_labels,
    top_x_detections=None,
    confidence_threshold: float = 0.0,
    min_area_ratio: float = 0.0001,  
    target_claasses: set = set(),
    skip_bg: bool = True,  # Explicitly passing skip_bg
    BG_CLASSES: list = [],  # Explicitly passing BG_CLASSES
):
    """
    Filter detections based on confidence, top X detections, and proximity of bounding boxes.
    Args:
        proximity_threshold (float): The minimum distance between centers of bounding boxes to consider them non-overlapping.
        keep_larger (bool): If True, keeps the larger bounding box when overlaps occur; otherwise keeps the smaller.
    Returns:
        tuple[sv.Detections, list[str]]: Filtered detections and labels.
    """
    # Sort by confidence initially
    detections_combined = sorted(
        zip(confidences,
            xyxy,
            class_ids,
            class_labels,),
        key=lambda x: x[0],
        reverse=True,
    )

    if top_x_detections is not None:
        detections_combined = detections_combined[:top_x_detections]

    min_area_threshold = min_area_ratio * image.shape[0] * image.shape[1]
    keep_idx_list, keep_lables = [], []
    for idx, current_det in enumerate(detections_combined):
        # remove background objects
        conf, curr_xyxy, curr_class_id, curr_label = current_det
        if skip_bg and curr_label in BG_CLASSES:
            # print("filter {}, as it is bg class".format(curr_label))
            continue

        # remove small objects
        curr_area = (curr_xyxy[2] - curr_xyxy[0]) * (curr_xyxy[3] - curr_xyxy[1])
        if curr_area < min_area_threshold:
            # print("filter {}, as it is too small".format(curr_label))
            continue

        # Check confidence threshold
        if conf < confidence_threshold and curr_label not in target_claasses:
            # print("filter {}, as its low confidence. Its confidence: {}".format(curr_label, conf))
            continue

        keep_idx_list.append(idx)
        keep_lables.append(curr_label)

    return xyxy[keep_idx_list], confidences[keep_idx_list], class_ids[keep_idx_list], keep_lables


def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


# @profile
def filter_masks(
    image: np.ndarray,
    masks: np.ndarray,
    mask_area_threshold: float = 10,  # Default value as fallback
    mask_area_ratio: float = None,  # Explicitly passing max_bbox_area_ratio
    mask_iou_threshold: float = 0.9,
):
    # If no detection at all
    if len(masks) == 0: 
        return None

    # Filter out the objects based on various criteria
    image_area = image.shape[0] * image.shape[1]
    min_area_threshold = min(mask_area_ratio*image_area, mask_area_threshold)

    keep_flag = np.zeros(len(masks), dtype=bool)
    mask_to_keep = []
    for mask_idx in range(len(masks)):

        curr_mask = masks[mask_idx]
        mask_area = curr_mask.sum()
        if mask_area < min_area_threshold:
            continue

        keep = True
        for other_mask in mask_to_keep:
            if mask_iou(curr_mask, other_mask) > mask_iou_threshold:
                keep = False
                break

        if keep:
            keep_flag[mask_idx] = True
            mask_to_keep.append(curr_mask)


    return keep_flag


def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    """
    找出大bbox包含小bbox的情况, 对于这些, 去除大bbox对应mask中小bbox对应的mask
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.

    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2

    Returns:
        mask_sub: (N, H, W), binary mask
    """
    N = xyxy.shape[0]  # number of boxes

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])  # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(
        xyxy[:, None, 2:], xyxy[None, :, 2:]
    )  # right-bottom points (N, N, 2)

    # N * N * 2, 有N个bbox, 得到的矩阵是关于主对角线对称的对称矩阵，第一行/列表示其他bbox与第一个bbox的重叠，如无重叠则为两个坐标至少有一个是负值
    inter = (rb - lt).clip(
        min=0
    )  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # 重叠区域面积
    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

    # 按行看，每行是各个bbox与ni的重叠区域占ni大小的比例
    # 按列看，每列是各个bbox与ni的重叠区域占各个bbox面积的比例
    inter_over_box1 = inter_areas / areas[:, None]  # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T  # (N, N)

    # 判断是否被包含：如果重叠区域占一个bbox很少，而占另一个很多，则为被包含. 
    # 如果第一个bbox大，包含第二个，则在右上角。如果第二个大，包含第一个，则在左下角
    # if the intersection area is smaller than th2 of the area of box1,
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)  # (N, N)

    # （行的idx, 列的idx），表示：行的idx对应的bbox包含列的idx对应的bbox
    contained_idx = contained.nonzero()  # (num_contained, 2)

    # 在大的bbox的mask里面去掉小的bbox的mask
    mask_sub = mask.copy()  # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (
            ~mask_sub[contained_idx[1][i]]
        )

    return mask_sub


def compute_visual_similarities(
    det_fts, obj_fts
) -> torch.Tensor:
    """
    Compute the visual similarities between the detections and the objects

    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of visual similarities
    """
    det_fts = det_fts.unsqueeze(-1)  # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0)  # (1, D, N)

    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1)  # (M, N)

    return visual_sim


def match_detections_to_objects(
    agg_sim: torch.Tensor,
    existing_obj_ids: List[int],
    detected_obj_ids: List[int],
    detection_threshold: float = float("-inf"),
) -> List[Tuple[int, Optional[int]]]:
    """
    Matches detections to objects based on similarity, returning match indices or None for unmatched.

    Args:
        agg_sim: Similarity matrix (detections vs. objects).
        detection_threshold: Threshold for a valid match (default: -inf).

    Returns:
        List of matching object indices (or None if unmatched) for each detection.
    """
    match_indices = []
    for detected_obj_idx in range(agg_sim.shape[0]):
        max_sim_value = agg_sim[detected_obj_idx].max()
        if max_sim_value <= detection_threshold:
            match_indices.append((detected_obj_ids[detected_obj_idx], None))
        else:
            # match_indices.append(agg_sim[detected_obj_idx].argmax().item())
            match_indices.append(
                (
                    detected_obj_ids[detected_obj_idx],
                    existing_obj_ids[agg_sim[detected_obj_idx].argmax().item()],
                )
            )

    return match_indices
