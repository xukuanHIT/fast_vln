import os
import numpy as np
import logging
import habitat_sim
import quaternion
from habitat_sim.utils.common import quat_to_angle_axis, quat_from_coeffs


from habitat_sim.utils.common import (
    quat_to_coeffs,
    quat_from_angle_axis,
    quat_from_two_vectors,
)

def pos_normal_to_habitat(pts):
    # +90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))


def pos_habitat_to_normal(pts):
    # -90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))


def pose_habitat_to_normal(pose):
    # T_normal_cam = T_normal_habitat * T_habitat_cam
    return np.dot(
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), pose
    )


def pose_normal_to_tsdf(pose):
    return np.dot(
        pose, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    )


def pose_habitat_to_tsdf(pose):
    '''
    w: world
    c: camera
    h: habitat
    b: robot
    Thb -> Twc
    pose_habitat_to_normal: Thb -> Twb
    pose_normal_to_tsdf: Twb -> Twc
    '''
    return pose_normal_to_tsdf(pose_habitat_to_normal(pose))


def pose_normal_to_tsdf_real(pose):
    # This one makes sense, which is making x-forward, y-left, z-up to z-forward, x-right, y-down
    return pose @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


def get_pts_angle_aeqa(init_pts, init_quat):
    '''
    quaternion to angle
    '''
    pts = np.asarray(init_pts)

    init_quat = quaternion.quaternion(*init_quat)
    angle, axis = quat_to_angle_axis(init_quat)

    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle


def get_scene_bnds(pathfinder, floor_height):
    # Get mesh boundaries - this is for the full scene
    scene_bnds = pathfinder.get_bounds()
    scene_lower_bnds_normal = pos_habitat_to_normal(scene_bnds[0])
    scene_upper_bnds_normal = pos_habitat_to_normal(scene_bnds[1])
    scene_size = np.abs(
        np.prod(scene_upper_bnds_normal[:2] - scene_lower_bnds_normal[:2])
    )
    tsdf_bnds = np.array(
        [
            [
                min(scene_lower_bnds_normal[0], scene_upper_bnds_normal[0]),
                max(scene_lower_bnds_normal[0], scene_upper_bnds_normal[0]),
            ],
            [
                min(scene_lower_bnds_normal[1], scene_upper_bnds_normal[1]),
                max(scene_lower_bnds_normal[1], scene_upper_bnds_normal[1]),
            ],
            [
                floor_height - 0.2,
                floor_height + 3.5,
            ],
        ]
    )
    return tsdf_bnds, scene_size



def make_semantic_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    sim_cfg.load_semantic_mesh = True
    # sim_cfg.gpu_device_id = -1  # 强制 CPU / headless 渲染

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor, a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = habitat_sim.geo.UP * settings["sensor_height"]
    rgb_sensor_spec.orientation = [settings["camera_tilt"], 0.0, 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = habitat_sim.geo.UP * settings["sensor_height"]
    depth_sensor_spec.orientation = [settings["camera_tilt"], 0.0, 0.0]
    depth_sensor_spec.hfov = settings["hfov"]

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = habitat_sim.geo.UP * settings["sensor_height"]
    semantic_sensor_spec.orientation = [settings["camera_tilt"], 0.0, 0.0]
    semantic_sensor_spec.hfov = settings["hfov"]

    agent_cfg.sensor_specifications = [
        rgb_sensor_spec,
        depth_sensor_spec,
        semantic_sensor_spec,
    ]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = habitat_sim.geo.UP * settings["sensor_height"]
    rgb_sensor_spec.orientation = [settings["camera_tilt"], 0.0, 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = habitat_sim.geo.UP * settings["sensor_height"]
    depth_sensor_spec.orientation = [settings["camera_tilt"], 0.0, 0.0]
    depth_sensor_spec.hfov = settings["hfov"]

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def get_quaternion(angle, camera_tilt):
    normalized_angle = angle % (2 * np.pi)
    if np.abs(normalized_angle - np.pi) < 1e-6:
        return quat_to_coeffs(
            quaternion.quaternion(0, 0, 1, 0)
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()

    return quat_to_coeffs(
        quat_from_angle_axis(angle, np.array([0, 1, 0]))
        * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
    ).tolist()


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype="float32") / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype="uint8")


class HabitatData:

    def __init__(self, cfg):
        self.cfg = cfg

    def load_data(self, scene_id):
        split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
        scene_mesh_path = os.path.join(
            self.cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.glb"
        )
        navmesh_path = os.path.join(
            self.cfg.scene_data_path,
            split,
            scene_id,
            scene_id.split("-")[1] + ".basis.navmesh",
        )
        semantic_texture_path = os.path.join(
            self.cfg.scene_data_path,
            split,
            scene_id,
            scene_id.split("-")[1] + ".semantic.glb",
        )
        scene_semantic_annotation_path = os.path.join(
            self.cfg.scene_data_path,
            split,
            scene_id,
            scene_id.split("-")[1] + ".semantic.txt",
        )

        assert os.path.exists(
            scene_mesh_path
        ), f"scene_mesh_path: {scene_mesh_path} does not exist"
        assert os.path.exists(
            navmesh_path
        ), f"navmesh_path: {navmesh_path} does not exist"
        if not os.path.exists(semantic_texture_path) or not os.path.exists(
            scene_semantic_annotation_path
        ):
            logging.warning(
                f"semantic_texture_path: {semantic_texture_path} or scene_semantic_annotation_path: {scene_semantic_annotation_path} does not exist"
            )

        sim_settings = {
            "scene": scene_mesh_path,
            "default_agent": 0,
            "sensor_height": self.cfg.camera_height,
            "width": self.cfg.img_width,
            "height": self.cfg.img_height,
            "hfov": self.cfg.hfov,
            "scene_dataset_config_file": self.cfg.scene_dataset_config_path,
            "camera_tilt": self.cfg.camera_tilt_deg * np.pi / 180,
        }
        if os.path.exists(semantic_texture_path) and os.path.exists(
            scene_semantic_annotation_path
        ):
            sim_cfg = make_semantic_cfg(sim_settings)
        else:
            sim_cfg = make_simple_cfg(sim_settings)

        self.simulator = habitat_sim.Simulator(sim_cfg)
        self.pathfinder = self.simulator.pathfinder
        self.pathfinder.seed(self.cfg.seed)
        self.pathfinder.load_nav_mesh(navmesh_path)

        self.agent = self.simulator.initialize_agent(sim_settings["default_agent"])


    def get_observation(self, pts, angle):
        agent_state = habitat_sim.AgentState()
        agent_state.position = pts
        agent_state.rotation = get_quaternion(angle, 0)
        self.agent.set_state(agent_state)

        obs = self.simulator.get_sensor_observations()

        # get camera extrinsic matrix
        sensor = self.agent.get_state().sensor_states["depth_sensor"]
        quaternion_0 = sensor.rotation
        translation_0 = sensor.position
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
        cam_pose[:3, 3] = translation_0

        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])

        return obs, cam_pose
    

    def get_frontier_observation(self, pts, view_dir, camera_tilt=0.0):
        agent_state = habitat_sim.AgentState()

        # solve edge cases of viewing direction
        default_view_dir = np.asarray([0.0, 0.0, -1.0])
        if np.linalg.norm(view_dir) < 1e-3:
            view_dir = default_view_dir
        view_dir = view_dir / np.linalg.norm(view_dir)

        agent_state.position = pts
        # set agent observation direction
        if np.dot(view_dir, default_view_dir) / np.linalg.norm(view_dir) < -1 + 1e-3:
            # if the rotation is to rotate 180 degree, then the quaternion is not unique
            # we need to specify rotating along y-axis
            agent_state.rotation = quat_to_coeffs(
                quaternion.quaternion(0, 0, 1, 0)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()
        else:
            agent_state.rotation = quat_to_coeffs(
                quat_from_two_vectors(default_view_dir, view_dir)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()

        self.agent.set_state(agent_state)
        obs = self.simulator.get_sensor_observations()

        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])

        return obs