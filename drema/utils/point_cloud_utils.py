import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2


def project_depth(depth, intrinsics):
  # depth: H x W
  # intrinsics: 3 x 3
  H, W = depth.shape
  K = intrinsics

  # create point cloud
  y, x = np.mgrid[0:H, 0:W]
  x = x.flatten()
  y = y.flatten()
  z = depth.flatten()
  points = np.stack([x, y, z], axis=1)

  constant_x = 1.0 / K[0, 0]
  constant_y = 1.0 / K[1, 1]
  centerX = K[0, 2]
  centerY = K[1, 2]

  points[:, 0] = (points[:, 0] - centerX) * points[:, 2] * constant_x
  points[:, 1] = (points[:, 1] - centerY) * points[:, 2] * constant_y

  return points


def project_depth_extrinsics(depth, intrinsics, extrinsics):
    # depth: H x W
    # intrinsics: 3 x 3
    # extrinsics: 4 x 4 (Transformation matrix from camera to world or another frame)
    H, W = depth.shape
    K = intrinsics

    # Create point cloud
    y, x = np.mgrid[0:H, 0:W]
    x = x.flatten()
    y = y.flatten()
    z = depth.flatten()
    points = np.stack([x, y, z, np.ones_like(z)], axis=1)  # Homogeneous coordinates

    constant_x = 1.0 / K[0, 0]
    constant_y = 1.0 / K[1, 1]
    centerX = K[0, 2]
    centerY = K[1, 2]

    points[:, 0] = (points[:, 0] - centerX) * points[:, 2] * constant_x
    points[:, 1] = (points[:, 1] - centerY) * points[:, 2] * constant_y

    # Apply extrinsics transformation
    points = (extrinsics @ points.T).T  # Transform to new coordinate system

    return points[:, :3]  # Return only x, y, z coordinates

def compute_rotoranslation_table(center, coordinates, path, visualize):
    if center is None:
        center = [0, 0, 0]
    if coordinates is None:
        coordinates = [0, 0, 1]

    # compute the normal vector of the plane
    normal = np.array([coordinates[0], coordinates[1], coordinates[2]])

    # compute the rotation matrix that aligns the normal vector with the z-axis
    rotation = R.align_vectors([normal], [[0, 0, 1]])[0].as_matrix()

    # compute the translation vector
    translation = -np.array(center)

    # save the transformation
    path_transformation = os.path.join(path, "transformation")
    os.makedirs(path_transformation, exist_ok=True)
    np.save(os.path.join(path_transformation, "R.npy"), rotation)
    np.save(os.path.join(path_transformation, "s.npy"), 1)
    np.save(os.path.join(path_transformation, "t.npy"), translation)

    if visualize:
        o3d_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        o3d_mesh.rotate(rotation)
        o3d_mesh.translate(translation)

        o3d_mesh_original = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        o3d.visualization.draw_geometries([o3d_mesh, o3d_mesh_original])

def depth_to_gaussian_frame(depth, intrinsics_gs, extrinsics_gs, intrinsics_sim, extrinsics_sim, rotation, scale, translation):

    point_camera_frame = project_depth(depth, intrinsics_sim)
    point_world_rl = np.dot(extrinsics_sim[:3, :3], point_camera_frame.transpose()).transpose() + extrinsics_sim[:3, 3]

    point_world_rl_gs = np.dot(rotation.transpose(), (point_world_rl - translation).transpose() / scale).transpose()

    inverted_extrinsics_gs = np.eye(4)
    inverted_extrinsics_gs[:3, :3] = extrinsics_gs[:3, :3].transpose()
    inverted_extrinsics_gs[:3, 3] = - inverted_extrinsics_gs[:3, :3] @ extrinsics_gs[:3, 3]

    point_gs_camera_frame = np.dot(inverted_extrinsics_gs[:3, :3], point_world_rl_gs.transpose()).transpose() + inverted_extrinsics_gs[:3, 3]

    image = np.zeros(depth.shape)
    x = (point_gs_camera_frame[:, 0] / point_gs_camera_frame[:, 2] * intrinsics_gs[0, 0] + intrinsics_gs[0, 2])
    y = (point_gs_camera_frame[:, 1] / point_gs_camera_frame[:, 2] * intrinsics_gs[1, 1] + intrinsics_gs[1, 2])
    x = x.astype(int)
    y = y.astype(int)
    mask = (0 <= x) & (x < image.shape[1]) & (0 <= y) & (y < image.shape[0])
    image[y[mask], x[mask]] = point_gs_camera_frame[mask, 2]

    return image




def filter_radious_outlier(depth, intrinsics, points=16, radius=0.05, visualize=False):
    # filter the depth
    size = depth.shape
    point_camera_frame = project_depth(depth, intrinsics)
    pc_gs = o3d.geometry.PointCloud()
    pc_gs.points = o3d.utility.Vector3dVector(point_camera_frame)
    cl, ind = pc_gs.remove_radius_outlier(nb_points=points, radius=radius)

    # get removed indices from ind
    ind_removed = np.where(np.isin(np.arange(len(point_camera_frame)), ind) == False)[0]

    # convert ind_removed to a mask
    mask = np.ones(size)
    mask = mask.flatten()
    mask[ind_removed] = 0
    mask = mask.reshape(size)

    depth = depth * mask

    # color the original points as green


    # color the removed points as red
    #pc_removed.paint_uniform_color([1, 0, 0])  # red

    # visualize the point clouds
    #o3d.visualization.draw_geometries([pc_gs, pc_removed])
    if visualize:
        pc_gs.paint_uniform_color([0, 1, 0])  # green
        cl = o3d.geometry.PointCloud()
        cl.points = o3d.utility.Vector3dVector(project_depth(depth, intrinsics))
        cl.paint_uniform_color([1, 0, 0])  # red
        o3d.visualization.draw_geometries([pc_gs, cl])

    return depth

def filter_scharr(depth, intrinsics, th=2, visualize=False):
    grad_x = cv2.Scharr(depth, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(depth, cv2.CV_64F, 0, 1)
    gradient_magnitude_scharr = np.sqrt(grad_x ** 2 + grad_y ** 2)
    depth[gradient_magnitude_scharr > th] = 0  # Remove outliers
    if visualize:
        # visualize the gradient magnitude alongside the filtered depth
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(gradient_magnitude_scharr)
        ax[1].imshow(gradient_magnitude_scharr > th)
        plt.show()

        pc_gs = o3d.geometry.PointCloud()
        pc_gs.points = o3d.utility.Vector3dVector(project_depth(depth, intrinsics))
        o3d.visualization.draw_geometries([pc_gs])

    return depth