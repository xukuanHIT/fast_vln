import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import quat_to_angle_axis, quat_from_coeffs
import quaternion
import time
import logging

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        # print(f"Starting {func.__name__}...")
        result = func(
            *args, **kwargs
        )  # Call the function with any arguments it was called with
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(
            f"Done! Execution time of {func.__name__} function: {elapsed_time:.2f} seconds"
        )
        return result  # Return the result of the function call

    return wrapper


def resize_image(image, target_h, target_w):
    # image: np.array, h, w, c
    image = Image.fromarray(image)
    image = image.resize((target_w, target_h))
    return np.array(image)


def get_pts_angle_aeqa(init_pts, init_quat):
    '''
    quaternion to angle
    '''
    pts = np.asarray(init_pts)

    init_quat = quaternion.quaternion(*init_quat)
    angle, axis = quat_to_angle_axis(init_quat)

    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle


def get_pts_angle_goatbench(init_pos, init_rot):
    pts = np.asarray(init_pos)

    init_quat = quat_from_coeffs(init_rot)
    angle, axis = quat_to_angle_axis(init_quat)
    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle


def calc_agent_subtask_distance(curr_pts, viewpoints, pathfinder):
    # calculate the distance to the nearest view point
    all_distances = []
    for viewpoint in viewpoints:
        path = habitat_sim.ShortestPath()
        path.requested_start = curr_pts
        path.requested_end = viewpoint
        found_path = pathfinder.find_path(path)
        if not found_path:
            all_distances.append(np.inf)
        else:
            all_distances.append(path.geodesic_distance)
    return min(all_distances)


