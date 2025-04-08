import numpy as np
import open3d as o3d
import os

from PIL import Image
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement

from drema.utils.depth_image_encoding import ImageToFloatArray
from drema.utils.point_cloud_utils import project_depth_extrinsics


def create_frame_from_observation(observation):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)

    # get the rotation matrix
    rotation = observation["gripper_pose"][3:]
    rotation = R.from_quat(rotation).as_matrix()

    # translate and rotate the frame
    frame.translate(observation["gripper_pose"][:3])
    frame.rotate(rotation, center=observation["gripper_pose"][:3])

    return frame

def create_visualization_frames(observations, keypoints):
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.30)
    keypoints_frame_list = []
    trajectory = []
    for i, obs in enumerate(observations):
        # add to trajectory
        trajectory.append(obs["gripper_pose"][:3])

        # if it is a keypoint
        if i in keypoints:
            # create a frame at the position of the observation and rotate it
            frame = create_frame_from_observation(obs)

            # add the frame to the list
            keypoints_frame_list.append(frame)

    trajectory_frame = o3d.geometry.LineSet()
    trajectory_frame.points = o3d.utility.Vector3dVector(trajectory)
    segments = [[i, i + 1] for i in range(len(trajectory) - 1)]
    trajectory_frame.lines = o3d.utility.Vector2iVector(segments)

    return origin_frame, keypoints_frame_list, trajectory_frame

def get_images_and_point_cloud(data_path, observation, frame_number, camera, color=None):
    # get the path to the images
    path_depth = os.path.join(data_path, camera + "_depth", str(frame_number) + ".png")
    path_rgb = os.path.join(data_path, camera + "_rgb", str(frame_number) + ".png")
    near = observation["%s_camera_near" % camera]
    far = observation["%s_camera_far" % camera]

    color_image = Image.open(path_rgb)
    depth_image = Image.open(path_depth)
    depth_array = ImageToFloatArray(depth_image, 2**24 - 1)
    depth_array = near + (far - near) * depth_array

    extrinsics = observation["%s_camera_extrinsics" % camera]
    intrinsics = observation["%s_camera_intrinsics" % camera]

    point_cloud = project_depth_extrinsics(depth_array, intrinsics, extrinsics)
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))

    if color is None:
        color_array = np.array(color_image)/255
        o3d_point_cloud.colors = o3d.utility.Vector3dVector(color_array.reshape(-1, 3))
    else:
        o3d_point_cloud.colors = o3d.utility.Vector3dVector(np.array([color] * point_cloud.shape[0]))

    return color_image, depth_array, o3d_point_cloud

def visualize_trajectory(client, trajectory, color=np.array([1, 0, 0, 1]), visualize_keypoints=True, keypoint_color=np.array([0, 1, 0, 1])):
    visualization_objects = []
    visualization_multiple_body = []

    # create the visualization objects
    visualization_objects.append(client.createVisualShape(shapeType=client.GEOM_SPHERE, radius=0.005, rgbaColor=color,
                                                          visualFramePosition=[0, 0, 0]))
    if visualize_keypoints:
        visualization_objects.append(client.createVisualShape(shapeType=client.GEOM_SPHERE, radius=0.01,
                                                              rgbaColor=keypoint_color, visualFramePosition=[0, 0, 0]))

    positions = trajectory.get_trajectory_positions()

    # create the visualization bodies
    for position in positions:
        id = client.createMultiBody(baseMass=0, baseInertialFramePosition=[0, 0, 0],
                                         baseVisualShapeIndex= visualization_objects[0], basePosition=position,
                                         useMaximalCoordinates=True)
        visualization_multiple_body.append(id)

    if visualize_keypoints:
        keypoint =  trajectory.keypoints
        for keypoint in keypoint:
            id = client.createMultiBody(baseMass=0, baseInertialFramePosition=[0, 0, 0],
                                             baseVisualShapeIndex= visualization_objects[1],
                                             basePosition=positions[keypoint], useMaximalCoordinates=True)
            visualization_multiple_body.append(id)

    return visualization_objects, visualization_multiple_body

def reset_visualization(client, visualization_objects, visualization_multiple_body):
    for obj in visualization_objects:
         client.removeBody(obj)
    for obj in  visualization_multiple_body:
         client.removeBody(obj)


def load_ply(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, scales, rots, opacities