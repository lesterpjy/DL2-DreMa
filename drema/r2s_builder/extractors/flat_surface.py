import numpy as np
import open3d as o3d

from drema.utils.point_cloud_utils import project_depth


def extract_table_points(depths, masks, depth_parameters, visualize=False):

    # read segmentation masks and depth maps to extract table points
    table_points = []
    for mask, depth, parameters in zip(masks, depths, depth_parameters):

        rotation, translation, intrinsics = parameters
        points = project_depth(depth, intrinsics)

        table_points_view = points[mask.reshape(-1), :]
        table_points_view = np.dot(rotation, table_points_view.T).T + translation

    table_points.append(table_points_view)
    table_points = np.concatenate(table_points, axis=0)

    return table_points


def extract_flat_surface_from_points(path, table_points, visualize=False):

    if table_points.shape[0] == 0:
        raise ValueError("No table points found.")

    # downsample the points
    if table_points.shape[0] > 50000:
        table_points = table_points[np.random.choice(table_points.shape[0], 50000, replace=False), :]

    # create point cloud
    table = o3d.geometry.PointCloud()
    table.points = o3d.utility.Vector3dVector(table_points)
    if visualize:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([table, mesh_frame])

    # remove outliers
    table, _ = table.remove_radius_outlier(nb_points=16, radius=0.05)

    # compute plane from the point cloud
    plane_coordinates, plane_index_points = table.segment_plane(0.01, 3, 1000)
    [a, b, c, d] = plane_coordinates
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # create a point cloud of the plane
    plane_cloud = table.select_by_index(plane_index_points)
    center = np.mean(np.array(plane_cloud.points), axis=0)
    plane_cloud = plane_cloud.translate(-center)
    if visualize:
        o3d.visualization.draw_geometries([plane_cloud])

    # compute hull of the plane
    hull, _ = plane_cloud.compute_convex_hull()
    hull_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    if visualize:
        hull_mesh.paint_uniform_color((1, 0, 0))
        o3d.visualization.draw_geometries([hull_mesh, plane_cloud])

    # save mesh
    return center, plane_coordinates, plane_cloud, hull_mesh
