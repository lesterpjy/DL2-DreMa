import os
import shutil

import cv2
import numpy as np

from frame_manager import FrameManager

class DataManger:
    def __init__(self, config):
        self.source_path = config.source_path

        # labels params
        self.labels_path = os.path.join(self.source_path, config.training.preparation.labels_file) # labels path
        # check that labels are present
        if not os.path.exists(self.labels_path):
            raise Exception("Labels not found in the source path")

        # mask path
        self.objects_mask_path = os.path.join(self.source_path, config.training.preparation.object_mask_folder) # output path

        # rgb params
        self.rgb_path = os.path.join(self.source_path, config.training.preparation.rgb_folder) # rgb path
        self.input_path = os.path.join(self.source_path, "input") # input path

        # depth params
        self.depth_path = os.path.join(self.source_path, config.training.preparation.depth_folder)
        self.depth_format = config.training.preparation.depth_format # depth format
        self.depth_scale = config.training.preparation.depth_scale # depth scale
        self.depth_compressed = config.training.preparation.depth_compressed # depth compressed

        # poses params
        self.poses_path = os.path.join(self.source_path, config.training.reference_frame.poses_folder)
        self.use_colmap = config.training.preparation.reference_frame

        self.rgb_images = []
        self.depth_images = []
        self.depth_names = []

    def prepare(self):
        # move the rgb images to input
        self.prepare_rgb_images()

        # if it requires to convert compute poses
        if self.use_colmap:
            FrameManager(self.source_path, self.poses_path).compute_colmap_poses()
        else:
            # copy rgb images from input to images
            os.makedirs(os.path.join(self.source_path, "images"), exist_ok=True)
            shutil.copytree(self.input_path, os.path.join(self.source_path, "images"), dirs_exist_ok=True)

        self.prepared_depth_images()

        self.prepare_coppelia_labels()

    def prepared_depth_images(self):
        files = [f for f in os.listdir(self.depth_path) if os.path.isfile(os.path.join(self.depth_path, f))]
        files.sort()
        for file in files:
            file_name = file.split(".")[0]
            self.depth_names.append(file_name)

            # if image is numpy
            if self.depth_format == "npy":
                depth = np.load(os.path.join(self.depth_path, file))
            # if image is png
            elif self.depth_format == "png":
                depth = cv2.imread(os.path.join(self.depth_path, file), cv2.IMREAD_UNCHANGED)
            else:
                raise Exception("Depth format not supported")

            # scale depth
            depth *= self.depth_scale

            # if depth is compressed decompress it
            if self.depth_compressed:
                depth = self.decompress_depth(depth, file_name)

            self.depth_images.append(depth)

        # create the output folder
        os.makedirs(os.path.join(self.source_path, "depth_scaled"), exist_ok=True)

        for idx, file in enumerate(self.depth_names):
            depth = self.depth_images[idx]
            np.save(os.path.join(self.source_path, "depth_scaled", file + ".npy"), depth)

    def decompress_depth(self, depth, file_name):
        # if it is from coppelia read the far and near clipping plane
        with open(os.path.join(self.source_path, "poses", file_name+"_near_far.txt"), "r") as near_far_file:
            lines = near_far_file.read().split("\n")
            far = float(lines[0].split(" ")[1])
            near = float(lines[0].split(" ")[0])

        depth = (far - near) * depth + near
        return depth

    def prepare_rgb_images(self):
        # if the input directory does not exist, create it
        if not os.path.exists(self.input_path):
            os.makedirs(self.input_path)

        for file in os.listdir(self.rgb_path):
            # copy the file in the input directory
            rgb_path = os.path.join(self.rgb_path, file)
            output_file = os.path.join(self.input_path, file)

            # move the file using shutil
            shutil.move(rgb_path, output_file)

    def prepare_coppelia_labels(self):
        # read the ids from the file
        new_labels = []
        labels_to_remove = []
        new_label = 0
        with open(self.labels_path, "r") as file:
            lines = file.read().split("\n")
            for line in lines:
                if len(line.split(";")) == 1:
                    continue
                name, number = line.split(";")

                if "Default" in name or "Floor" in name or "Wall" in name or "spawn_boundary" in name:
                    labels_to_remove.append(int(number))
                    continue

                if "pillar" in name:
                    labels_to_remove.append(int(number))
                    continue
                elif "square_base" in name:
                    new_label = int(number)
                elif "shape_sorter_visual" in name:
                    labels_to_remove.append(int(number))
                    continue
                elif "shape_sorter" in name:
                    new_label = int(number)

                # add the new label
                new_labels.append((name, int(number)))

        # remove the labels from the masks
        for filename in os.listdir(self.objects_mask_path):
            if filename.endswith(".png"):
                mask = cv2.imread(os.path.join(self.objects_mask_path, filename), cv2.IMREAD_UNCHANGED)
                for label in labels_to_remove:
                    mask[mask == label] = new_label
                cv2.imwrite(os.path.join(self.objects_mask_path, filename), mask)

        # move the previous file in labels_original.txt
        shutil.copy(self.labels_path, os.path.join(self.source_path, "labels_original.txt"))

        # save the new labels
        with open(os.path.join(self.source_path, "labels.txt"), "w") as file:
            for label in new_labels:
                file.write(f"{label[0]};{label[1]}\n")


