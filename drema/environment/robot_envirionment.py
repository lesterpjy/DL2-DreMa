import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


from drema.environment.base_environment import Environment
from drema.environment.robot.robot import Robot


class RobotEnvironment(Environment):
    def __init__(self, objects, flat_surface, pipe_args, robot, trajectory=None, max_waypoint_steps=100,
                 waypoint_threshold=0.2, keypoint_threshold=0.015, sh_degrees=3, environment_path=None,
                 visualize=False):
        super().__init__(objects, flat_surface, pipe_args, sh_degrees, environment_path, visualize)

        # create the robot
        self.robot = Robot("Robot", urdf_path=robot["urdf_path"], gaussians_path=robot["gaussians_path"],
                           connected_joint=robot["connected_joints"], initial_joint_positions=robot["initial_joint_positions"],
                           end_effector_rotation=robot["end_effector_rotation"], grasp_index=robot["grasp_index"],
                           left_finger_index=robot["left_finger_index"], right_finger_index=robot["right_finger_index"],
                           initial_position=robot["initial_position"], initial_orientation=robot["initial_orientation"],
                           initial_gripper_joint_positions=robot["initial_gripper_joint_positions"] , fixed=robot["fixed"],
                           wrist_camera_index=robot["wrist_camera_index"])

        self.trajectory = trajectory
        self.waypoint_index = 0
        self.waypoint_steps = 0
        self.max_waypoint_steps = max_waypoint_steps
        self.waypoint_threshold = waypoint_threshold
        self.keypoint_threshold = keypoint_threshold

    def build_environment(self):
        super().build_environment()

        # load the robot
        self.robot.load(self.client)

        # load the robot gaussians
        robot_translation = torch.tensor(self.robot.initial_position).cuda()
        robot_labels = []
        for path in self.robot.gaussians_path:
            gaussians_robot_id = self.gs.add_object_guassians(path, translation=robot_translation)
            self.robot.set_gaussians_labels(gaussians_robot_id)
            robot_labels.append(gaussians_robot_id)

        robot_labels = torch.tensor(robot_labels, dtype=torch.int32, device="cuda")

        # set the parts labels for the robot
        self.robot.set_gaussians_labels(robot_labels)

        self.gs.filter_robot_gaussians(robot_labels)

        # set the gaussians mask for each object again
        labels = self.gs.get_labels()
        for obj in self.objects.values():
            obj.set_gaussians_mask(labels == obj.get_gaussians_labels())

        # set the gaussians mask for the robot
        masks = []
        for label in robot_labels:
            masks.append(labels == label)
        self.robot.set_gaussians_mask(masks)

        # set not collidable objects
        for obj in self.objects.values():
            if not obj.collidable:
                obj.set_not_collidable(self.client, [], [self.robot])

        # save the initial state
        self.gs.save_state_as_initial()

    def reset(self):
        super().reset()

        # reset the robot
        self.robot.reset(self.client)

        # reset the trajectory
        if self.trajectory is not None:
            self.trajectory.reset()

        # reset the waypoint index
        self.waypoint_index = 0
        self.waypoint_steps = 0

    def load_trajectory(self, trajectory):
        self.trajectory = trajectory

    def update_state(self):
        super().update_state()

        # update the robot state
        self.robot.update(self.client)


    def update_gaussians(self):
        super().update_gaussians()

        # update the robot gaussians
        current_position, current_orientation = self.robot.get_link_state(self.client)
        prev_position, prev_orientation = self.robot.get_previous_link_state(self.client)

        difference_rotation = np.matmul(current_orientation, prev_orientation.transpose((0, 2, 1)))
        difference_translation = current_position - prev_position
        center = prev_position

        difference_rotation = torch.tensor(difference_rotation, dtype=torch.float32, device="cuda")
        difference_translation = torch.tensor(difference_translation, dtype=torch.float32, device="cuda")
        center = torch.tensor(center, dtype=torch.float32, device="cuda")

        masks = self.robot.get_gaussians_mask()

        for i, link in enumerate(self.robot.connected_joint):
            if i == 0:
                continue

            self.gs.rototranslate_link(masks[i], difference_rotation[link], difference_translation[link], center[link])

        # update the previous state
        self.robot.update_previous_link_state()

    def move_robot_to_waypoint(self, trajectory):
        """
        Execute the trajectory
        :param trajectory: list of joint positions
        :return:
        """

        # set the target pose
        target_pose = trajectory.get_target_pose(self.waypoint_index)
        target_position = target_pose[:3]
        target_orientation = R.from_quat(target_pose[3:])

        self.robot.move_to_pose(self.client, target_position, target_orientation)

        self.waypoint_steps += 1

        return target_position, target_orientation

    def step(self):
        """
        Execute the step
        :return: -1 if the robot did not reach the target, 0 if the robot is still executing the trajectory, 1 if the robot reached the target
        """
        step_output = 0

        if self.trajectory is not None:
            target_position, target_orientation = self.move_robot_to_waypoint(self.trajectory)

            threshold = self.keypoint_threshold if self.trajectory.is_keypoint(self.waypoint_index) else self.waypoint_threshold

            # check if the robot reached the waypoint
            position_reached, orientation_reached = self.robot.target_position_reached(self.client, target_position, target_orientation, position_threshold=threshold)
            if position_reached and orientation_reached:

                # set gripper state
                if self.current_waypoint_is_keypoint():
                    target_gripper_open = self.trajectory.get_gripper_open(self.waypoint_index)
                    if target_gripper_open:
                        self.robot.open_gripper(self.client)
                    else:
                        self.robot.close_gripper(self.client)

                step_output = 1

            elif self.waypoint_steps >= self.max_waypoint_steps:
                step_output = -1

        super().step()

        return step_output

    def current_waypoint_is_keypoint(self):
        return self.trajectory.is_keypoint(self.waypoint_index)

    def next_waypoint(self):
        """
        Move to the next waypoint
        :return:
        """
        self.waypoint_index += 1
        self.waypoint_steps = 0

        if self.waypoint_index >= len(self.trajectory):
            self.waypoint_index = 0

        return self.waypoint_index

    def update_trajectory_with_current_data(self):
        """
        Update the trajectory with the current data
        :return:
        """
        #if self.trajectory is not None:

            #self.trajectory.update_with_current_data(self.robot, self.waypoint_index)
        pass

    def get_wrist_camera_extrinsics(self):
        """
        Get the wrist camera extrinsics
        :return: translation, rotation
        """
        return self.robot.get_wrist_camera_extrinsics(self.client)