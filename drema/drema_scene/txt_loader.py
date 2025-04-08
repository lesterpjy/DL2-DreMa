import numpy as np
import os

from scipy.spatial.transform import Rotation as R

from gaussin_splatting_utils.general_utils import read_pose_file
import scene.colmap_loader as colmap_loader


def read_coppelia_extrinsic(path):

    cam_extrinsics = {}
    files = os.path.join(path, "poses")
    extrinsics_files = [f for f in os.listdir(files) if os.path.isfile(os.path.join(files, f))]
    extrinsics_files = [f for f in extrinsics_files if len(f) == 8]
    for k, file in enumerate(extrinsics_files):
        give_rotation, give_translation, given_intrinsics = read_pose_file(os.path.join(files, file), separator=" ")
        inverted_rotation = R.from_matrix(give_rotation).inv().as_matrix()
        qvec = R.from_matrix(inverted_rotation).as_quat()
        inverted_translation = -np.dot(inverted_rotation, give_translation)
        qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])

        image_name = file.split(".")[0] + ".png"
        cam_extrinsics[k] = colmap_loader.Image(
            id=k, qvec=qvec, tvec=inverted_translation,
            camera_id=1, name=image_name,
            xys=[], point3D_ids=[])

    return cam_extrinsics


def read_coppelia_intrinsics(path):

        cameras = {}
        files = os.path.join(path, "poses")
        extrinsics_files = [f for f in os.listdir(files) if os.path.isfile(os.path.join(files, f))]
        extrinsics_files = [f for f in extrinsics_files if len(f) == 8]

        give_rotation, give_translation, given_intrinsics = read_pose_file(os.path.join(files, extrinsics_files[0]), separator=" ")

        camera_id = 1
        model_id = 1
        model_name = colmap_loader.CAMERA_MODEL_IDS[model_id].model_name
        width = given_intrinsics[0,2]*2 # no distortion
        height = given_intrinsics[1,2]*2
        params = np.array([given_intrinsics[0,0], given_intrinsics[1,1], given_intrinsics[0,2], given_intrinsics[1,2]])

        cameras[camera_id] = colmap_loader.Camera(id=camera_id,
                                    model=model_name,
                                    width=width,
                                    height=height,
                                    params=np.array(params))
