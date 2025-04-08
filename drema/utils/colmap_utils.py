import os
import open3d as o3d
import numpy as np

import scipy.spatial.transform as R

from drema.utils.drema_camera_utils import read_pose_file, write_pose_file


def read_colmap_images_txt(file_path):
    colmap_names = []
    colmap_translations = []
    colmap_rotations = []

    images = open(file_path, "r")
    lines = images.read().split("\n")

    # remove header lines and last empty line
    lines = lines[4:-1]
    # get only lines with extrinsics
    lines = lines[::2]

    for line in lines:
        splitted = line.split(" ")
        image_name = splitted[-1]

        # read colmap data
        colmap_translation = np.array([float(splitted[-5]), float(splitted[-4]), float(splitted[-3])])
        colmap_quaternion = np.array([float(splitted[-8]), float(splitted[-7]), float(splitted[-6]), float(splitted[-9])])
        colmap_rotation_matrix = R.from_quat(colmap_quaternion).inv().as_matrix()
        colmap_translation_inv = -np.dot(colmap_rotation_matrix, colmap_translation)
        colmap_translations.append(colmap_translation_inv)
        #colmap_rotations.append(R.from_quat(colmap_quaternion).as_matrix())
        colmap_rotations.append(colmap_rotation_matrix)
        colmap_names.append(image_name)

    return colmap_names, colmap_translations, colmap_rotations

def read_colmap_intrinsics_txt(file_path):
    images = open(file_path, "r")
    lines = images.read().split("\n")
    lines = lines[3:-1]
    params = lines[0].split(" ")
    intrisics = np.zeros((3, 3))
    intrisics[0, 0] = float(params[-4])
    intrisics[1, 1] = float(params[-3])
    intrisics[0, 2] = float(params[-2])
    intrisics[1, 2] = float(params[-1])
    intrisics[2, 2] = 1

    return intrisics


