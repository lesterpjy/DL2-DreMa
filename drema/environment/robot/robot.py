import numpy as np

from drema.environment.assets.object import ArticulatedObject


class Robot(ArticulatedObject):
    def __init__(self, name, urdf_path, gaussians_path, connected_joint, initial_joint_positions, end_effector_rotation, grasp_index, left_finger_index, right_finger_index,  initial_position=np.zeros(3),
                 initial_orientation=np.array([0,0,0,1]), initial_gripper_joint_positions=np.array([0.04,0.04]), wrist_camera_index=-1, fixed=True):
        super().__init__(name, urdf_path, gaussians_path, connected_joint, initial_joint_positions=initial_joint_positions,
                         initial_position=initial_position, initial_orientation=initial_orientation, collidable=True, fixed=fixed)

        self.grasp_index = grasp_index
        self.left_finger_index = left_finger_index
        self.right_finger_index = right_finger_index
        self.end_effector_rotation = end_effector_rotation # rotation of the end effector e.g. R.from_euler('xyz', [0, 0, 135], degrees=True)
        self.initial_gripper_joint_positions = initial_gripper_joint_positions
        self.wrist_camera_index = wrist_camera_index

        self.ll = []
        # upper limits for null space (todo: set them to proper range)
        self.ul = []
        # joint ranges for null space (todo: set them to proper range)
        self.jr = []


    def load(self, client):
        """
        Load the robot in the simulation environment
        :param client: PyBullet client
        :return: None
        """
        super().load(client)

        self.ll = [-7] * self.joints_number
        self.ul = [7] * self.joints_number
        self.jr = [7] * self.joints_number

    def close_gripper(self, client):
        """
        Close the gripper
        :param client: PyBullet client
        :return: None
        """
        client.setJointMotorControl2(self.id, self.left_finger_index, client.POSITION_CONTROL, 0.0)
        client.setJointMotorControl2(self.id, self.right_finger_index, client.POSITION_CONTROL, 0.0)

    def open_gripper(self, client):
        """
        Open the gripper
        :param client: pyblullet client
        :return:
        """
        client.setJointMotorControl2(self.id, self.left_finger_index, client.POSITION_CONTROL, 0.04)
        client.setJointMotorControl2(self.id, self.right_finger_index, client.POSITION_CONTROL, 0.04)

    def get_gripper_pose(self, client):
        """
        Get the gripper pose (position, orientation)
        :param client: PyBullet client
        :return: gripper pose
        """
        link_state =  client.getLinkState(self.id, self.grasp_index)
        position = link_state[0]
        orientation = link_state[1]
        return np.array(position), np.array(orientation)

    def move_to_pose(self, client, target_position, target_orientation):
        """
        Move the robot to a given pose
        :param client: PyBullet client
        :param position: position
        :param orientation: orientation encoded in scipy.spatial.transform.Rotation
        :return: None
        """
        target_orientation = (target_orientation * self.end_effector_rotation).as_quat()

        joint_poses = client.calculateInverseKinematics(self.id, self.grasp_index, target_position, target_orientation,
                                                        lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr,
                                                        restPoses=self.initial_joint_positions)

        for i, joint_pose in enumerate(joint_poses):
            client.setJointMotorControl2(self.id, i, client.POSITION_CONTROL, joint_pose, force=2*240., maxVelocity=2)

    def distance_to_target(self, client, target_position, target_orientation):
        """
        Compute the distance between the end effector and the target pose
        :param client: PyBullet client
        :param target_position: target position
        :param target_orientation: target orientation
        :return: distance angle_sim
        """
        current_position, current_orientation = self.get_gripper_pose(client)
        distance = np.linalg.norm(current_position - target_position)
        angle_sim = 1 - abs(np.dot(target_orientation, current_orientation))

        return distance, angle_sim

    def target_position_reached(self, client, target_position, target_orientation, position_threshold=0.01, angle_threshold=1e-1):
        """
        Check if the target position is reached
        :param client: PyBullet client
        :param target_position: target position
        :param threshold: threshold
        :return: True if the target position is reached, False otherwise
        """
        distance, sim = self.distance_to_target(client, target_position, target_orientation.as_quat())
        return distance < position_threshold, sim < angle_threshold

    def get_wrist_camera_extrinsics(self, client):
        current_wrist_pose = client.getLinkState(self.id, self.wrist_camera_index)
        current_wrist_position = np.array(current_wrist_pose[0])
        current_wrist_orientation = np.array(current_wrist_pose[1])

        current_wrist_orientation = np.array(client.getMatrixFromQuaternion(current_wrist_orientation)).reshape(3,3)

        return current_wrist_position, current_wrist_orientation

