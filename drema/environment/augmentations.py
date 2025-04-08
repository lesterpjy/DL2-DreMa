import numpy as np
import torch

from scipy.spatial.transform import Rotation as R

from drema.environment.robot_envirionment import RobotEnvironment
from drema.utils.generation_utils import rotate_from_reference

class AugmentationManager:
    def __init__(self, environment: RobotEnvironment):
        self.environment = environment

    def rotate_environment(self, center, rotation, target_positions):
        """
        Rotate the environment and the objects in it
        :param center:
        :param rotation:
        :param target_positions:
        :return:
        """

        # convert rotation in euler to quaternion
        rotation = R.from_euler('xyz', [0,0, rotation], degrees=True).as_matrix()

        # rotate the trajectory
        self.rotate_trajectory(rotation, center)

        new_target_positions = self._rotate_objects(center, rotation, target_positions, rotate_table=True)

        # get mask of the environment
        environment_mask = self.environment.gs.get_labels() == 0

        # translate the environment to the center
        rotation = torch.tensor(rotation).float().cuda()
        center = torch.tensor(center).float().cuda()
        self.environment.gs.rotate(environment_mask, rotation, center)

        return new_target_positions

    def translate_environment(self, translation, target_positions):
        """
        Translate the environment and the objects in it
        :param translation:
        :param target_positions:
        :return:
        """
        # translate the trajectory
        self.translate_trajectory(translation)

        new_target_positions = self._translate_objects(translation, target_positions)

        # get mask of the environment
        environment_mask = self.environment.gs.get_labels() == 0

        # translate the environment
        translation = torch.tensor(translation).float().cuda()
        self.environment.gs.translate(environment_mask, translation)

        return new_target_positions

    def rotate_objects(self, rotation, target_positions):
        """
        Rotate the objects in the environment
        :param center:
        :param rotation:
        :param target_positions:
        :return:
        """
        # convert rotation in euler to quaternion
        rotation = R.from_euler('xyz', [0, 0, rotation], degrees=True).as_matrix()

        # rotate the trajectory
        center = self.rotate_trajectory(rotation)

        # rotate the gaussians
        return self._rotate_objects(center, rotation, target_positions)

    def _rotate_objects(self, center, rotation, target_positions, rotate_table=False):
        """
        Rotate the objects in the environment
        :param center:
        :param rotation:
        :param target_positions:
        :return:
        """
        for obj in self.environment.objects.values():
            # get the current state of the object
            current_position, current_orientation = obj.get_state(self.environment.client)

            # get the new position and orientation
            new_position, new_orientation = rotate_from_reference(current_position, current_orientation, center, rotation)

            # set the new state
            obj.set_state(new_position, new_orientation, self.environment.client)

        if rotate_table:
            table = self.environment.flat_surface
            table_position, table_orientation = table.get_state(self.environment.client)
            new_table_position, new_table_orientation = rotate_from_reference(table_position, table_orientation, center, rotation)
            table.set_state(new_table_position, new_table_orientation, self.environment.client)

        new_target_positions = []
        if target_positions is not None:
            for i, target_position in enumerate(target_positions):
                new_target_position = rotate_from_reference(target_position, np.eye(3), center, rotation)[0]
                new_target_positions.append(new_target_position)

        return new_target_positions

    def _translate_objects(self, translation, target_positions, translate_table=False):
        """
        Translate the objects in the environment
        :param translation:
        :param target_positions:
        :return:
        """
        for obj in self.environment.objects.values():
            # get the current state of the object
            current_position, current_orientation = obj.get_state(self.environment.client)

            # get the new position
            new_position = current_position + translation

            # set the new state
            obj.set_state(new_position, current_orientation, self.environment.client)

        if translate_table:
            table = self.environment.flat_surface
            table_position, table_orientation = table.get_state(self.environment.client)
            new_table_position = table_position + translation
            table.set_state(new_table_position, table_orientation, self.environment.client)

        new_target_positions = []
        if target_positions is not None:
            for i, target_position in enumerate(target_positions):
                new_target_position = target_position + translation
                new_target_positions.append(new_target_position)

        return new_target_positions

    def translate_trajectory(self, translation):
        for i in range(len(self.environment.trajectory.demo)):
            self.environment.trajectory.demo[i]["gripper_pose"][:3] += translation

    def rotate_trajectory(self, rotation, center=None):
        if center is None:
            last_pose = self.environment.trajectory.demo[-1]["gripper_pose"]
            center = last_pose[:3]

        for i in range(len(self.environment.trajectory.demo)):
            pose = self.environment.trajectory.demo[i]["gripper_pose"]
            position = pose[:3]  # - last_position
            orientation = R.from_quat(pose[3:]).as_matrix()

            world_position, world_orientation = rotate_from_reference(position, orientation, center, rotation)
            #world_orientation = R.from_matrix(rotation.transpose() @ orientation).as_quat()
            world_orientation = R.from_matrix(world_orientation).as_quat()
            self.environment.trajectory.demo[i]["gripper_pose"] = np.concatenate((world_position, world_orientation))

        return center