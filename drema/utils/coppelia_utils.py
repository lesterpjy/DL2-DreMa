import os
import shutil
import cv2
import numpy as np

def read_labels(path, filter_labels=True):
    """
    Read the labels from the file and return a dictionary
    :param path: path the file.txt
    :param filter_labels: a boolean to filter the labels with lable less than 55 (background in coppelia)
    :return: a dictionary with the labels name and the integer value
    """
    labels = {}
    with open(path, "r") as file:
        lines = file.read().split("\n")
        for line in lines:
            if len(line.split(";")) == 1:
                continue
            name, number = line.split(";")
            if filter_labels and int(number) < 60:
                continue

            labels[name] = int(number)

    return labels


def merge_coppelia_labels(path):
    # read the ids from the file
    new_labels = []
    labels_to_remove = []
    new_label = 0
    with open(os.path.join(path, "labels.txt"), "r") as file:
        lines = file.read().split("\n")
        for line in lines:
            if len(line.split(";")) == 1:
                continue
            name, number = line.split(";")

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
    masks_path = os.path.join(path, "object_mask")
    for filename in os.listdir(masks_path):
        if filename.endswith(".png"):
            mask = cv2.imread(os.path.join(masks_path, filename), cv2.IMREAD_UNCHANGED)
            for label in labels_to_remove:
                mask[mask == label] = new_label
            cv2.imwrite(os.path.join(masks_path, filename), mask)

    # move the previous file in labels_original.txt
    shutil.move(os.path.join(path, "labels.txt"), os.path.join(path, "labels_original.txt"))

    # save the new labels
    with open(os.path.join(path, "labels.txt"), "w") as file:
        for label in new_labels:
            file.write(f"{label[0]};{label[1]}\n")

def decompress_coppelia_depth(path, real_data=False, real_data_scale=0.001):
    depth_path = os.path.join(path, "depth")

    # create the output folder
    os.makedirs(os.path.join(path, "depth_scaled"), exist_ok=True)

    # check images
    files = [f for f in os.listdir(depth_path) if os.path.isfile(os.path.join(depth_path, f))]
    files.sort()
    for file in files:
        # if real data then we only need to scale the depth
        file_name = file.split(".")[0]
        if real_data:
            depth = cv2.imread(os.path.join(depth_path, file), cv2.IMREAD_UNCHANGED)
            depth = depth * real_data_scale
        else:
            # if it is from coppelia read the far and near clipping plane
            with open(os.path.join(path, "poses", file_name+"_near_far.txt"), "r") as near_far_file:
                lines = near_far_file.read().split("\n")
                far = float(lines[0].split(" ")[1])
                near = float(lines[0].split(" ")[0])

            depth = np.load(os.path.join(depth_path, file))
            # here I should use the far clipping plane
            depth = (far - near) * depth + near
        np.save(os.path.join(path, "depth_scaled", file_name+".npy"), depth)

