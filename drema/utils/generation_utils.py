import os
import shutil

from PIL import Image

from drema.utils.depth_image_encoding import FloatArrayToRgbImage


def rotate_from_reference(position, orientation, r_position, rotation):
    """

    :param position: array of shape (3,) position to be rotated
    :param orientation: 2D array of shape (3, 3) orientation matrix to be rotated
    :param r_position: array of shape (3,) encoding the rotation center
    :param rotation: 2D array of shape (3, 3) encoding the rotation
    :return: rotated position and orientation as arrays of shape (3,) and (3, 3) respectively
    """
    # bring the position and orientation to the reference frame
    position = position - r_position

    # rotate the position and orientation
    rotated_orientation = rotation @ orientation
    rotated_position = rotation @ position

    # bring the rotated position and orientation back to the world frame
    world_position = rotated_position + r_position
    world_orientation = rotated_orientation

    return world_position, world_orientation


def save_images(output_path, images, original_data_path=None, DEPTH_SCALE=2 ** 24 - 1):
    """
    Save images to the output path
    :param output_path: path to save the images
    :param images:  list of tuples containing the names, images and depths
    :param original_data_path: if not None, copy the original data to the output path
    :return:
    """
    # save images
    os.makedirs(output_path, exist_ok=True)
    if original_data_path is not None:
        # input paths
        original_trajectory = os.path.join(original_data_path, "low_dim_obs.pkl")
        original_descriptions = os.path.join(original_data_path,"variation_descriptions.pkl")
        original_variation_number = os.path.join(original_data_path, "variation_number.pkl")

        # output paths
        trajectory_output = os.path.join(output_path, "original_low_dim_obs.pkl")
        description_output = os.path.join(output_path, "variation_descriptions.pkl")
        number_output = os.path.join(output_path, "variation_number.pkl")

        # copy the original data
        shutil.copyfile(original_trajectory, trajectory_output)
        shutil.copyfile(original_descriptions, description_output)
        shutil.copyfile(original_variation_number, number_output)

    for step, observation in enumerate(images):
        names, rgbs, depths = observation
        for i, name in enumerate(names):

            os.makedirs(os.path.join(output_path, name + "_rgb"), exist_ok=True)
            os.makedirs(os.path.join(output_path, name + "_depth"), exist_ok=True)
            Image.fromarray(rgbs[i]).save(os.path.join(output_path, name + "_rgb", str(step) + ".png"))
            FloatArrayToRgbImage(depths[i], scale_factor=DEPTH_SCALE).save(os.path.join(output_path, name + "_depth", str(step) + ".png"))
