#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
import shutil

def compute_colmap(source_path, colmap_executable="", no_gpu=False, camera="PINHOLE"):

    colmap_command = '"{}"'.format(colmap_executable) if len(colmap_executable) > 0 else "colmap"
    use_gpu = 1 if not no_gpu else 0

    os.makedirs(os.path.join(source_path, "distorted/sparse"), exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor " \
        "--database_path " + source_path + "/distorted/database.db \
        --image_path " + source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + source_path + "/distorted/database.db \
        --image_path " + source_path + "/input \
        --output_path " + source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Image undistortion
    ## TODO: include also undistorte.


    files = os.listdir(source_path + "/input")
    os.makedirs(source_path + "/images", exist_ok=True)
    for file in files:
        source_file = os.path.join(source_path, "input", file)
        destination_file = os.path.join(source_path, "images", file)
        shutil.copy(source_file, destination_file)

    files = os.listdir(source_path + "/distorted/sparse/0")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(source_path, "distorted", "sparse", "0", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    # convert binary to txt to be able to read it
    model_converter_cmd = (colmap_command + " model_converter \
        --input_path " + os.path.join(source_path, "sparse", "0") + "  \
        --output_path " + os.path.join(source_path, "sparse", "0") + " \
        --output_type TXT")
    exit_code = os.system(model_converter_cmd)
    if exit_code != 0:
        logging.error(f"Model converter failed with code {exit_code}. Exiting.")
        exit(exit_code)


