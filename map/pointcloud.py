from collections import Counter
import logging
import numpy as np
import open3d as o3d
import torch
import typing as tp
import torch.nn.functional as F
import faiss


def batch_mask_depth_to_points_colors(
    depth_tensor: torch.Tensor,
    masks_tensor: torch.Tensor,
    cam_K: torch.Tensor,
    image_rgb_tensor: torch.Tensor = None,  # Parameter for RGB image tensor
    device: str = "cuda",
) -> tuple:
    """
    Converts a batch of masked depth images to 3D points and corresponding colors.

    Args:
        depth_tensor (torch.Tensor): A tensor of shape (N, H, W) representing the depth images.
        masks_tensor (torch.Tensor): A tensor of shape (N, H, W) representing the masks for each depth image.
        cam_K (torch.Tensor): A tensor of shape (3, 3) representing the camera intrinsic matrix.
        image_rgb_tensor (torch.Tensor, optional): A tensor of shape (N, H, W, 3) representing the RGB images. Defaults to None.
        device (str, optional): The device to perform the computation on. Defaults to 'cuda'.

    Returns:
        tuple: A tuple containing the 3D points tensor of shape (N, H, W, 3) and the colors tensor of shape (N, H, W, 3).
    """
    N, H, W = masks_tensor.shape
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]

    # Generate grid of pixel coordinates
    y, x = torch.meshgrid(torch.arange(0, H, device=device), torch.arange(0, W, device=device), indexing='ij')
    z = depth_tensor.repeat(N, 1, 1) * masks_tensor  # Apply masks to depth
    
    valid = (z > 0).float()  # Mask out zeros
    
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    points = torch.stack((x, y, z), dim=-1) * valid.unsqueeze(-1)  # Shape: (N, H, W, 3)

    # _x, _z = torch.meshgrid(
    #     torch.arange(0, W, device=device),
    #     torch.arange(H - 1, -1, -1, device=device),
    #     indexing="xy",
    # )
    # y = depth_tensor.repeat(N, 1, 1) * masks_tensor  # Apply masks to depth

    # valid = (y > 0).float()  # Mask out zeros

    # x = (_x - cx) * y / fx
    # z = (_z - cy) * y / fy
    # points = torch.stack((x, z, -y), dim=-1) * valid.unsqueeze(-1)  # Shape: (N, H, W, 3)

    if image_rgb_tensor is not None:
        # Repeat RGB image for each mask and apply masks
        repeated_rgb = image_rgb_tensor.repeat(N, 1, 1, 1) * masks_tensor.unsqueeze(-1)
        colors = repeated_rgb * valid.unsqueeze(
            -1
        )  # Apply valid mask to filter out background
    else:
        print("No RGB image provided, assigning random colors to objects")
        # log it as well
        logging.warning("No RGB image provided, assigning random colors to objects")
        # Generate a random color for each mask
        random_colors = (
            torch.randint(0, 256, (N, 3), device=device, dtype=torch.float32) / 255.0
        )  # RGB colors in [0, 1]
        # Expand dims to match (N, H, W, 3) and apply to valid points
        colors = random_colors.unsqueeze(1).unsqueeze(1).expand(
            -1, H, W, -1
        ) * valid.unsqueeze(-1)

    return points, colors



# @profile
def dynamic_downsample(points, colors=None, target=5000):
    """
    Simplified and configurable downsampling function that dynamically adjusts the
    downsampling rate based on the number of input points. If a target of -1 is provided,
    downsampling is bypassed, returning the original points and colors.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) for N points.
        target (int): Target number of points to aim for in the downsampled output,
                      or -1 to bypass downsampling.
        colors (torch.Tensor, optional): Corresponding colors tensor of shape (N, 3).
                                         Defaults to None.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: Downsampled points and optionally
                                                     downsampled colors, or the original
                                                     points and colors if target is -1.
    """
    # Check if downsampling is bypassed
    if target == -1:
        return points, colors

    num_points = points.size(0)

    # If the number of points is less than or equal to the target, return the original points and colors
    if num_points <= target:
        return points, colors

    # Calculate downsampling factor to aim for the target number of points
    downsample_factor = max(1, num_points // target)

    # Select points based on the calculated downsampling factor
    downsampled_points = points[::downsample_factor]

    # If colors are provided, downsample them with the same factor
    downsampled_colors = colors[::downsample_factor] if colors is not None else None

    return downsampled_points, downsampled_colors


# @profile
def get_bounding_box(spatial_sim_type, pcd):
    if ("accurate" in spatial_sim_type or "overlap" in spatial_sim_type) and len(
        pcd.points
    ) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()
    


def detections_to_obj_pcd_and_bbox(
    depth_array,
    masks,
    cam_K,
    image_rgb=None,
    trans_pose=None,
    min_points_threshold=5,
    spatial_sim_type="axis_aligned",
    obj_pcd_max_points=None,
    downsample_voxel_size=None,
    dbscan_remove_noise=None,
    dbscan_eps=None,
    dbscan_min_points=None,
    run_dbscan=None,
    device="cuda",
):
    """
    This function processes a batch of objects to create colored point clouds, apply transformations, and compute bounding boxes.

    Args:
        depth_array (numpy.ndarray): Array containing depth values.
        masks (numpy.ndarray): Array containing binary masks for each object.
        cam_K (numpy.ndarray): Camera intrinsic matrix.
        image_rgb (numpy.ndarray, optional): RGB image. Defaults to None.
        trans_pose (numpy.ndarray, optional): Transformation matrix. Defaults to None.
        min_points_threshold (int, optional): Minimum number of points required for an object. Defaults to 5.
        spatial_sim_type (str, optional): Type of spatial similarity. Defaults to 'axis_aligned'.
        device (str, optional): Device to use. Defaults to 'cuda'.

    Returns:
        list: List of dictionaries containing processed objects. Each dictionary contains a point cloud and a bounding box.
    """
    N, H, W = masks.shape
    if trans_pose is not None:
        trans_pose = torch.tensor(trans_pose, device=device, dtype=torch.float)

    # Convert inputs to tensors and move to the specified device
    depth_tensor = torch.from_numpy(depth_array).to(device).float()
    masks_tensor = torch.from_numpy(masks).to(device).float()
    cam_K_tensor = torch.from_numpy(cam_K).to(device).float()

    if image_rgb is not None:
        image_rgb_tensor = (
            torch.from_numpy(image_rgb).to(device).float() / 255.0
        )  # Normalize RGB values
    else:
        image_rgb_tensor = None

    points_tensor, colors_tensor = batch_mask_depth_to_points_colors(
        depth_tensor, masks_tensor, cam_K_tensor, image_rgb_tensor, device
    )  # points_tensor: [N, H, W, 3], colors_tensor: [N, H, W, 3]

    processed_objects = [None] * N  # Initialize with placeholders
    for i in range(N):
        mask_points = points_tensor[i]
        mask_colors = colors_tensor[i] if colors_tensor is not None else None

        # valid_points_mask = mask_points[:, :, 2] < 0
        valid_points_mask = mask_points[:, :, 2] > 0
        if torch.sum(valid_points_mask) < min_points_threshold:
            continue

        valid_points = mask_points[valid_points_mask]
        valid_colors = (
            mask_colors[valid_points_mask] if mask_colors is not None else None
        )

        downsampled_points, downsampled_colors = dynamic_downsample(
            valid_points, colors=valid_colors, target=obj_pcd_max_points
        )

        if trans_pose is not None:
            downsampled_points = torch.cat(
                (downsampled_points, torch.ones_like(downsampled_points[:, :1])), dim=1
            ).transpose(
                0, 1
            )  # Add ones for homogeneous coordinates, shape: (4, N)
            downsampled_points = trans_pose @ downsampled_points  # Apply transformation
            downsampled_points = downsampled_points[:3].transpose(
                0, 1
            )  # Remove ones and transpose back to (N, 3)


        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(downsampled_points.cpu().numpy())
        if downsampled_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(downsampled_colors.cpu().numpy())

        # if trans_pose is not None:
        #     pcd.transform(trans_pose)  # Apply transformation directly to the point cloud
        #     pass

        bbox = get_bounding_box(spatial_sim_type, pcd)
        if bbox.volume() < 1e-6:
            continue

        processed_objects[i] = {"pcd": pcd, "bbox": bbox}

    return processed_objects



def init_pcd_denoise_dbscan(
    pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10
) -> o3d.geometry.PointCloud:
    ## Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(  # inint
        eps=eps,
        min_points=min_points,
    )

    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]

        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)

        pcd = largest_cluster_pcd

    return pcd



def init_process_pcd(
    pcd,
    downsample_voxel_size,
    dbscan_remove_noise,
    dbscan_eps,
    dbscan_min_points,
    run_dbscan=True,
):
    '''
    降采样和去噪
    '''
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)

    if dbscan_remove_noise and run_dbscan:
        pcd = init_pcd_denoise_dbscan(pcd, eps=dbscan_eps, min_points=dbscan_min_points)

    return pcd
 


def expand_3d_box(bbox: torch.Tensor, eps=0.02) -> torch.Tensor:
    """
    Expand the side of 3D boxes such that each side has at least eps length.
    Assumes the bbox cornder order in open3d convention.

    bbox: (N, 8, D)

    returns: (N, 8, D)
    """
    center = bbox.mean(dim=1)  # shape: (N, D)

    va = bbox[:, 1, :] - bbox[:, 0, :]  # shape: (N, D)
    vb = bbox[:, 2, :] - bbox[:, 0, :]  # shape: (N, D)
    vc = bbox[:, 3, :] - bbox[:, 0, :]  # shape: (N, D)

    a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)  # shape: (N, 1)

    va = torch.where(a < eps, va / a * eps, va)  # shape: (N, D)
    vb = torch.where(b < eps, vb / b * eps, vb)  # shape: (N, D)
    vc = torch.where(c < eps, vc / c * eps, vc)  # shape: (N, D)

    new_bbox = torch.stack(
        [
            center - va / 2.0 - vb / 2.0 - vc / 2.0,
            center + va / 2.0 - vb / 2.0 - vc / 2.0,
            center - va / 2.0 + vb / 2.0 - vc / 2.0,
            center - va / 2.0 - vb / 2.0 + vc / 2.0,
            center + va / 2.0 + vb / 2.0 + vc / 2.0,
            center - va / 2.0 + vb / 2.0 + vc / 2.0,
            center + va / 2.0 - vb / 2.0 + vc / 2.0,
            center + va / 2.0 + vb / 2.0 - vc / 2.0,
        ],
        dim=1,
    )  # shape: (N, 8, D)

    new_bbox = new_bbox.to(bbox.device)
    new_bbox = new_bbox.type(bbox.dtype)

    return new_bbox


def compute_3d_iou_accurate_batch(bbox1, bbox2):
    """
    Compute IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.

    bbox1: (M, 8, D), e.g. (M, 8, 3)
    bbox2: (N, 8, D), e.g. (N, 8, 3)

    returns: (M, N)
    """
    # Must expend the box beforehand, otherwise it may results overestimated results
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)

    import pytorch3d.ops as ops

    bbox1 = bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    bbox2 = bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]]

    inter_vol, iou = ops.box3d_overlap(bbox1.float(), bbox2.float())

    return iou



# @profile
def compute_overlap_matrix_general(
    points_a,
    bbox_a,
    points_b=None,
    bbox_b=None,
    downsample_voxel_size=None
):
    """
    Compute the overlap matrix between two sets of objects represented by their point clouds. This function can also perform self-comparison when `objects_b` is not provided. The overlap is quantified based on the proximity of points from one object to the nearest points of another, within a threshold specified by `downsample_voxel_size`.

    Parameters
    ----------
    objects_a : MapObjectList
        A list of object representations where each object contains a point cloud ('pcd') and bounding box ('bbox').
        This is the primary set of objects for comparison.

    objects_b : Optional[MapObjectList]
        A second list of object representations similar to `objects_a`. If None, `objects_a` will be compared with itself to calculate self-overlap. Defaults to None.

    downsample_voxel_size : Optional[float]
        The threshold for determining whether points are close enough to be considered overlapping. Specifically, it's the square of the maximum distance allowed between points from two objects to consider those points as overlapping.
        Must be provided; if None, a ValueError is raised.

    Returns
    -------
    torch.Tensor
        A 2D numpy array of shape (len(objects_a), len(objects_b)) containing the overlap ratios between objects.
        The overlap ratio is defined as the fraction of points in the second object's point cloud that are within `downsample_voxel_size` distance to any point in the first object's point cloud.

    Raises
    ------
    ValueError
        If `downsample_voxel_size` is not provided.

    Notes
    -----
    The function uses the FAISS library for efficient nearest neighbor searches to compute the overlap.
    Additionally, it employs a 3D IoU (Intersection over Union) computation for bounding boxes to quickly filter out pairs of objects without spatial overlap, improving performance.
    - The overlap matrix helps identify potential duplicates or matches between new and existing objects based on spatial overlap.
    - High values (e.g., >0.8) in the matrix suggest a significant overlap, potentially indicating duplicates or very close matches.
    - Moderate values (e.g., 0.5-0.8) may indicate similar objects with partial overlap.
    - Low values (<0.5) generally suggest distinct objects with minimal overlap.
    - The choice of a "match" threshold depends on the application's requirements and may require adjusting based on observed outcomes.

    Examples
    --------
    >>> objects_a = [{'pcd': pcd1, 'bbox': bbox1}, {'pcd': pcd2, 'bbox': bbox2}]
    >>> objects_b = [{'pcd': pcd3, 'bbox': bbox3}, {'pcd': pcd4, 'bbox': bbox4}]
    >>> downsample_voxel_size = 0.05
    >>> overlap_matrix = compute_overlap_matrix_general(objects_a, objects_b, downsample_voxel_size)
    >>> print(overlap_matrix)
    """
    # if downsample_voxel_size is None, raise an error
    if downsample_voxel_size is None:
        raise ValueError("downsample_voxel_size is not provided")

    # hardcoding for now because its this value is actually not supposed to be the downsample voxel size
    downsample_voxel_size = 0.025

    # same_objects = points_2 is None
    # points_2, bbox_2 = points_1, bbox_1 if same_objects else points_2, bbox_2

    # to_swap = len(bbox_1) < len(bbox_2)
    # points_a, bbox_a, points_b, bbox_b = points_2, bbox_2, points_1, bbox_1 if to_swap else points_1, bbox_1, points_2, bbox_2

    same_objects = points_b is None
    if same_objects:
        points_b, bbox_b = points_a, bbox_a 

    len_a = len(bbox_a)
    len_b = len(bbox_b)
    overlap_matrix = np.zeros((len_a, len_b))

    indices_a = [faiss.IndexFlatL2(points_a_arr.shape[1]) for points_a_arr in points_a]  # m indices

    # Add the points from the numpy arrays to the corresponding FAISS indices
    for idx_a, points_a_arr in zip(indices_a, points_a):
        idx_a.add(points_a_arr)

    ious = compute_3d_iou_accurate_batch(bbox_a, bbox_b)  # (m, n)

    # Compute the pairwise overlaps
    for idx_a in range(len_a):
        for idx_b in range(len_b):

            # skip same object comparison if same_objects is True
            if same_objects and idx_a == idx_b:
                continue

            # skip if the boxes do not overlap at all
            if ious[idx_a, idx_b] < 1e-6:
                continue

            # get the distance of the nearest neighbor of
            # each point in points_b[idx_b] to the points_a[idx_a]
            D, I = indices_a[idx_a].search(points_b[idx_b], 1)
            overlap = (D < downsample_voxel_size**2).sum()  # D is the squared distance

            # Calculate the ratio of points within the threshold
            overlap_matrix[idx_a, idx_b] = overlap / len(points_b[idx_b])

    return torch.from_numpy(overlap_matrix)



# @profile
def pcd_denoise_dbscan(
    pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10
) -> o3d.geometry.PointCloud:
    # Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )

    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]

        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)

        pcd = largest_cluster_pcd

    return pcd



# @profile
def process_pcd(
    pcd,
    downsample_voxel_size,
    dbscan_remove_noise,
    dbscan_eps,
    dbscan_min_points,
    run_dbscan=True,
):
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)

    if dbscan_remove_noise and run_dbscan:
        pcd = pcd_denoise_dbscan(pcd, eps=dbscan_eps, min_points=dbscan_min_points)

    return pcd



# @profile
def denoise_objects(
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
    objects,
):
    '''
    使用dbscan对物体点云降噪, 降噪后会更新3d bbox
    '''
    logging.debug(f"Starting denoising with {len(objects)} objects")
    for obj_id in objects.keys():
        og_object_pcd = objects[obj_id].pcd

        if len(og_object_pcd.points) > 1:  # no need to denoise
            # Adjust the call to process_pcd with explicit parameters
            objects[obj_id].pcd = process_pcd(
                objects[obj_id].pcd,
                downsample_voxel_size,
                dbscan_remove_noise,
                dbscan_eps,
                dbscan_min_points,
                run_dbscan=True,
            )
            if len(objects[obj_id].pcd.points) < 4:
                objects[obj_id].pcd = og_object_pcd

        # Adjust the call to get_bounding_box with explicit parameters
        objects[obj_id].bbox = get_bounding_box(spatial_sim_type, objects[obj_id].pcd)
        # logging.debug(f"Finished denoising object {obj_id} out of {len(objects)}")
        # logging.debug(
        #     f"before denoising: {len(og_object_pcd.points)}, after denoising: {len(objects[obj_id]['pcd'].points)}"
        # )

    logging.debug(f"Finished denoising with {len(objects)} objects")
    return objects
