import os
import logging
import shutil

import numpy as np

from drema.gaussian_splatting_utils.colmap_converter import compute_colmap
from drema.utils.colmap_utils import read_colmap_images_txt, read_colmap_intrinsics_txt
from drema.utils.drema_camera_utils import read_pose_file
from drema.utils.utils import kabsch_umeyama


class FrameManager:
    def __init__(self, source_path, poses_path):
        self.source_path = source_path
        self.poses_path = poses_path

        self.names = []
        self.translations = np.array([])
        self.rotations = np.array([])
        self.intrinsics = np.eye(3)

        self.R = np.eye(3)
        self.s = 1
        self.t = np.zeros(3)

    def compute_colmap_poses(self):
        self.compute_colmap_for_scene()

        self.compute_transformation_from_poses()

        # rename directory poses to calibration_poses
        os.rename(self.poses_path, os.path.join(self.source_path, "calibration_poses"))

        self.align_using_transformation()

        # copy the adjusted poses to poses
        shutil.copytree(os.path.join(self.source_path, "adjusted_poses"), self.poses_path, dirs_exist_ok=True)


    def compute_colmap_for_scene(self, colmap_executable="", no_gpu=False, camera="PINHOLE"):
        compute_colmap(self.source_path, colmap_executable, no_gpu, camera)

        # Read the poses from colmap and the given poses
        colmap_images_path = os.path.join(self.source_path, "sparse", "0", "images.txt")
        colmap_names, colmap_translations, colmap_rotations = read_colmap_images_txt(colmap_images_path)
        colmap_intrinsics_path = os.path.join(self.source_path, "sparse", "0", "cameras.txt")
        colmap_intrinsics = read_colmap_intrinsics_txt(colmap_intrinsics_path)

        self.names = colmap_names
        self.translations = np.array(colmap_translations)
        self.rotations = np.array(colmap_rotations)
        self.intrinsics = colmap_intrinsics

    def compute_transformation_from_poses(self):
        given_translations = []
        for image_name in self.names:
            file_poses = os.path.join(self.poses_path, image_name.split(".")[0] + ".txt")
            _, give_translation, _ = read_pose_file(file_poses)
            given_translations.append(give_translation)
        give_translation = np.array(given_translations)

        self.R, self.s, self.t = kabsch_umeyama(give_translation, self.colmap_translations)

        path_transformation = os.path.join(self.source_path, "transformation")
        os.makedirs(path_transformation, exist_ok=True)
        np.save(os.path.join(path_transformation, "R.npy"), self.R)
        np.save(os.path.join(path_transformation, "s.npy"), self.s)
        np.save(os.path.join(path_transformation, "t.npy"), self.t)
        print("Scale factor c =", self.s)
        print("Rotation matrix R =", self.R)
        print("Translation vector t =", self.t)

    def align_using_transformation(self):
        aligned_translations = np.array([self.t + self.s * self.R @ b for b in self.colmap_translations])
        aligned_rotations = np.array([self.R @ b for b in self.colmap_rotations])

        # save the adjusted poses
        path_poses = os.path.join(self.source_path, "adjusted_poses")
        os.makedirs(path_poses, exist_ok=True)
        for i, image_name in enumerate(self.colmap_names):
            file_poses = os.path.join(path_poses, image_name.split(".")[0] + ".txt")

            # convert rotation and translation to extrinsic matrix
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = aligned_rotations[i]
            extrinsic[:3, 3] = aligned_translations[i]

            with open(file_poses, "w") as f:
                # write rotation and translation to file
                for row in extrinsic:
                    for value in row:
                        f.write(str(value) + " ")
                    f.write("\n")
                f.write("\n")
                # write intrinsics to file
                for row in self.colmap_intrinsics:
                    for value in row:
                        f.write(str(value) + " ")
                    f.write("\n")

