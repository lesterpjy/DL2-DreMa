import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

class Object:
    def __init__(self, name, urdf_path, gaussians_path, initial_position=np.zeros(3), initial_orientation=np.zeros(4),
                 collidable=True, fixed=False, mass=4, lateral_friction=1, spinning_friction=1, rolling_friction=1):

        self.position = initial_position.copy()
        self.orientation = initial_orientation.copy()

        self.previous_position = initial_position.copy()
        self.previous_orientation = initial_orientation.copy()

        self.initial_position = initial_position
        self.initial_orientation = initial_orientation

        self.URDF_path = urdf_path
        self.gaussians_path = gaussians_path
        self.name = ""
        self.id = -1
        self.gaussians_labels = 0
        self.gaussians_mask = torch.zeros(0, dtype=torch.int32, device="cuda")

        self.collidable = collidable
        self.fixed = fixed
        self.mass = mass
        if self.fixed:
            self.mass = 0

        self.lateral_friction = lateral_friction
        self.spinning_friction = spinning_friction
        self.rolling_friction = rolling_friction

    def load(self, client):
        """
        Load the object in the simulation environment
        :param client: PyBullet client
        :return: None
        """

        self.id = client.loadURDF(self.URDF_path, self.initial_position, self.initial_orientation)
        client.changeDynamics(self.id, -1, mass=self.mass, lateralFriction=self.lateral_friction)

        # TODO: check if the object is collidable

        return self.id

    def reset(self, client):
        """
        Reset the object to its initial position and orientation
        :param client: PyBullet client
        :return: None
        """
        client.resetBasePositionAndOrientation(self.id, self.initial_position, self.initial_orientation)
        client.resetBaseVelocity(self.id, np.zeros(3), np.zeros(3))

        self.position = self.initial_position.copy()
        self.orientation = self.initial_orientation.copy()

        self.previous_position = self.initial_position.copy()
        self.previous_orientation = self.initial_orientation.copy()

    def update(self, client):

        position, orientation = client.getBasePositionAndOrientation(self.id)

        #previous_position = self.position.copy()
        #previous_orientation = self.orientation.copy()

        self.position = np.array(position)
        self.orientation = np.array(orientation)

        # convert the orientation to a rotation matrix
        rot_matrix_current = np.array(client.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        #rot_matrix_previous = np.array(client.getMatrixFromQuaternion(previous_orientation)).reshape(3, 3)

        #return previous_position, rot_matrix_previous, self.position, rot_matrix_current

        return self.position, rot_matrix_current

    def set_gaussians_labels(self, label):
        """
        Set the integer label of the object
        :param label:
        :return:
        """
        self.gaussians_labels = label

    def set_gaussians_mask(self, mask):
        """
        Set the gaussians mask of the object
        :param mask:
        :return:
        """
        self.gaussians_mask = mask

    def set_state(self, position, orientation, client):
        """
        Set the object state
        :param position: position
        :param orientation: orientation
        :return: None
        """
        self.position = position.copy()
        self.orientation = orientation.copy()
        orientation = R.from_matrix(orientation).as_quat()
        client.resetBasePositionAndOrientation(self.id, self.position, orientation)

    def set_fixed(self, client):
        """
        Set the object fixed
        :param client: pybullet client
        :return: None
        """
        client.changeDynamics(self.id, -1, linearDamping=0, angularDamping=0)
        client.changeDynamics(self.id, -1, mass=0)
        client.changeDynamics(self.id, -1, frictionAnchor=True)

    def set_not_collidable(self, client, objects, articulated_objects=[]):
        """
        Set the object not collidable
        :param client: pybullet client
        :param objects:
        :param articulated_objects:
        :return:
        """
        self.set_fixed(client)
        for obj in objects:
            client.setCollisionFilterPair(self.id, obj.id, -1, -1, 0)

        for obj in articulated_objects:
            num_joints = obj.joints_number
            for i in range(num_joints):
                client.setCollisionFilterPair(self.id, obj.id, i, -1, 0)



    def get_gaussians_mask(self):
        """
        Get the gaussians mask of the object
        :return: torch tensor
        """
        return self.gaussians_mask

    def get_gaussians_labels(self):
        """
        Get the integer label of the object
        :return:
        """
        return self.gaussians_labels

    def get_state(self, client):
        """
        Get the object state
        :return: position, orientation
        """
        rot_matrix_current = np.array(client.getMatrixFromQuaternion(self.orientation)).reshape(3, 3)
        return self.position, rot_matrix_current

    def get_previous_state(self, client):
        """
        Get the previous state of the object
        :return: position, orientation
        """
        rot_matrix_current = np.array(client.getMatrixFromQuaternion(self.previous_orientation)).reshape(3, 3)
        return self.previous_position, rot_matrix_current

    def update_previous_state(self):
        """
        Update the previous state of the object
        :return: None
        """
        self.previous_position = self.position.copy()
        self.previous_orientation = self.orientation.copy()

class FlatSurface(Object):
    def __init__(self, urdf_path, gaussians_path, initial_position=np.zeros(3), initial_orientation=np.zeros(4), mass=0,
                 lateral_friction=1, spinning_friction=1, rolling_friction=1):
        super().__init__("flat_surface", urdf_path, gaussians_path, initial_position, initial_orientation, True, True,
                         mass, lateral_friction, spinning_friction, rolling_friction)


class ArticulatedObject(Object):
    def __init__(self, name, urdf_path, gaussians_path, connected_joint, initial_joint_positions=None, initial_position=np.zeros(3), initial_orientation=np.zeros(4),
                 collidable=True, fixed=False, mass=4, lateral_friction=1, spinning_friction=1, rolling_friction=1):

        super().__init__(name, urdf_path, gaussians_path, initial_position, initial_orientation, collidable, fixed, mass,
                         lateral_friction, spinning_friction, rolling_friction)

        self.joints_number = -1
        self.joints_ids = []
        self.joint_rotations = []
        self.joint_parent_position = []

        self.link_positions = []
        self.link_orientations = []

        self.previous_link_positions = []
        self.previous_link_orientations = []

        self.initial_link_positions = []
        self.initial_link_orientations = []
        self.initial_joint_positions = initial_joint_positions

        # self.parts = parts # a list containing the links of the object

        # update gausians attributes to consider the object parts
        self.gaussians_path = gaussians_path
        self.gaussians_labels = []
        self.gaussians_mask = torch.zeros(0, dtype=torch.int32, device="cuda")
        self.connected_joint = connected_joint

        self.initial_gripper_joint_positions = None

    def load(self, client):
        """
        Load the object in the simulation environment
        :param client: PyBullet client
        :return: None
        """
        # call the parent class method
        super().load(client)

        # get the number of joints
        self.joints_number = client.getNumJoints(self.id)
        self.joints_ids = [client.getJointInfo(self.id, i)[0] for i in range(self.joints_number)]

        if self.initial_joint_positions is None:
            # get the joint positions
            self.initial_joint_positions = [client.getJointState(self.id, i)[0] for i in range(self.joints_number)]
        else:
            # set the joint positions to the initial values
            index = 0
            for j, value in enumerate(self.initial_joint_positions):
                client.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)
                info = client.getJointInfo(self.id, j)

                # joint_name = info[1]
                joint_type = info[2]
                if joint_type == client.JOINT_PRISMATIC:
                    client.resetJointState(self.id, j, value)
                    index = index + 1
                if joint_type == client.JOINT_REVOLUTE:
                    client.resetJointState(self.id, j, value)
                    index = index + 1

        if self.initial_gripper_joint_positions is not None:
            # set gripper joint
            client.resetJointState(self.id, self.left_finger_index, self.initial_gripper_joint_positions[0])
            client.resetJointState(self.id, self.right_finger_index, self.initial_gripper_joint_positions[1])

        # set joints rotations
        self.joint_rotations = np.eye(3)[None, :, :] * np.ones((self.joints_number+1, 1, 1))

        for i in range(self.joints_number):
            joint_info = client.getJointInfo(self.id, i)
            self.joint_parent_position.append(np.array(joint_info[-3]))
            joint_rotation_parent = np.array(client.getMatrixFromQuaternion(joint_info[-2])).reshape(3, 3)
            joint_rotation = joint_rotation_parent.transpose() @ self.joint_rotations[i]

            self.joint_rotations[i] = joint_rotation
            self.joint_rotations[i+1] = joint_rotation
        self.joint_rotations = self.joint_rotations[:-1]
        self.joint_parent_position = np.array(self.joint_parent_position)

        # get the link positions and orientations
        link_states = client.getLinkStates(self.id, self.joints_ids)
        link_positions = np.array([np.array(e[4]) for e in link_states])
        link_orientations = np.array([np.array(e[5]) for e in link_states])

        self.link_positions = link_positions
        self.link_orientations = link_orientations

        self.initial_link_positions = link_positions.copy()
        self.initial_link_orientations = link_orientations.copy()

        self.previous_link_positions = link_positions.copy()
        self.previous_link_orientations = link_orientations.copy()

        return self.id

    def reset(self, client):
        """
        Reset the object to its initial position and orientation
        :param client: PyBullet client
        :return: None
        """
        # call the parent class method
        super().reset(client)

        # reset the joint positions
        for j, value in enumerate(self.initial_joint_positions):
            client.resetJointState(self.id, j, value)

        if self.initial_gripper_joint_positions is not None:
            # set gripper joint
            client.resetJointState(self.id, self.left_finger_index, self.initial_gripper_joint_positions[0])
            client.resetJointState(self.id, self.right_finger_index, self.initial_gripper_joint_positions[1])
            client.setJointMotorControl2(self.id, self.left_finger_index, client.POSITION_CONTROL, targetPosition=self.initial_gripper_joint_positions[0])
            client.setJointMotorControl2(self.id, self.right_finger_index, client.POSITION_CONTROL, targetPosition=self.initial_gripper_joint_positions[1])

        self.link_positions = self.initial_link_positions.copy()
        self.link_orientations = self.initial_link_orientations.copy()

        self.previous_link_positions = self.initial_link_positions.copy()
        self.previous_link_orientations = self.initial_link_orientations.copy()

    def update(self, client):
        """
        Update the object position and orientation (only the link positions and orientations)
        :param client: PyBullet client
        :return: None
        """
        # call the parent class method
        #previous_position, previous_orientation, position, orientation = super().update(client)

        # get the link positions and orientations
        link_states = client.getLinkStates(self.id, self.joints_ids)
        link_positions = np.array([np.array(e[4]) for e in link_states])
        link_orientations = np.array([np.array(e[5]) for e in link_states])

        self.link_positions = link_positions
        self.link_orientations = link_orientations

        return self.link_positions, self.link_orientations

    def get_link_state(self, client):
        """
        Get the object state
        :return: position, orientation
        """
        # TODO: check if the orientation is correct
        rotation_matrices = np.array([np.array(client.getMatrixFromQuaternion(q)).reshape(3, 3) for q in self.link_orientations])
        #rotation_matrices = np.matmul(rotation_matrices, self.joint_rotations)

        return self.link_positions, rotation_matrices

    def get_previous_link_state(self, client):
        """
        Get the previous state of the object
        :return: position, orientation
        """
        # TODO: check if the orientation is correct
        rotation_matrices = np.array([np.array(client.getMatrixFromQuaternion(q)).reshape(3, 3) for q in self.previous_link_orientations])
        #rotation_matrices = np.matmul(rotation_matrices, self.joint_rotations)
        return self.previous_link_positions, rotation_matrices

    def update_previous_link_state(self):
        """
        Update the previous state of the object
        :return: None
        """
        self.previous_link_positions = self.link_positions.copy()
        self.previous_link_orientations = self.link_orientations.copy()

