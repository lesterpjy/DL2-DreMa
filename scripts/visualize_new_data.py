import argparse
import os
import pickle
import open3d as o3d
import matplotlib.pyplot as plt

from drema.utils.trajectory_utils import compute_keypoints
from drema.utils.visualization_utils import create_visualization_frames, create_frame_from_observation, \
    get_images_and_point_cloud, load_ply

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_data_path", type=str, default="/home/leonardo/workspace/git_repo/DreMa/outputs/generated_data")
    parser.add_argument("--reconstructed_data_path", type=str, default="/home/leonardo/workspace/git_repo/SimulationGaussianSplatting/data")
    parser.add_argument("--task", type=str, default="slide_block_to_color_target")
    parser.add_argument("--original_episode", type=int, default=0)
    parser.add_argument("--episode", type=int, default=2001)
    parser.add_argument("--cameras",  nargs='+', type=str, default=["left_shoulder", "front"])
    parser.add_argument("--frame_numbers", nargs='+', type=int, default=[0,-1])
    parser.add_argument("--show_keypoints", type=bool, default=True)
    parser.add_argument("--show_original_colors", type=bool, default=True)
    parser.add_argument("--show_images", type=bool, default=True)
    parser.add_argument("--show_gaussian_point_cloud", type=bool, default=True)

    args = parser.parse_args()

    # create color vector
    colors = [[1, 0, 0],  # Red
              [0, 1, 0],  # Green
              [0, 0, 1],  # Blue
              [1, 1, 0],  # Yellow
              [1, 0, 1],  # Magenta
              [0, 1, 1]]  # Cyan

    # path
    args.generated_data_path = os.path.join(args.generated_data_path, args.task + "_episode"+str(args.original_episode)+"_start", "episode"+str(args.episode))
    observation_path = os.path.join(args.generated_data_path, "generated_trajectory.pkl")

    # load observations
    with open(observation_path, 'rb') as f:
        observations = pickle.load(f)

    # compute keypoints
    keypoints, _, _ = compute_keypoints(observations)
    print("Found ", len(keypoints), " keypoints with index: ", keypoints)

    # create visualization frames
    origin_frame, keypoints_frame_list, trajectory_frame = create_visualization_frames(observations, keypoints)

    visualizer = o3d.visualization.draw_geometries([origin_frame, trajectory_frame] + keypoints_frame_list)

    frames_to_visualize = args.frame_numbers
    # change -1 to the last frame
    if frames_to_visualize[-1] == -1:
        frames_to_visualize[-1] = len(observations)-1
    if args.show_keypoints:
        # add the keypoints to the frames to visualize
        frames_to_visualize += keypoints
        # sort the frames
        frames_to_visualize = sorted(frames_to_visualize)

    # get the length of the episode from the rgb_images
    episode_length = len(os.listdir(os.path.join(args.generated_data_path, args.cameras[0] + "_rgb")))

    images = []
    for frame_number in frames_to_visualize:
        if frame_number >= episode_length:
            print("Frame number is greater than the episode length")
            break

        # get the observation
        obs = observations[frame_number]

        # create the frame
        frame = create_frame_from_observation(obs)

        # get the images and point clouds
        point_clouds = []
        observation_images = []
        for k, camera in enumerate(args.cameras):
            color_image, depth_image, point_cloud = get_images_and_point_cloud(args.generated_data_path, obs, frame_number,
                                                                               camera,
                                                                               None if args.show_original_colors else
                                                                               colors[k])

            point_clouds.append(point_cloud)
            observation_images.append([color_image, depth_image])

        images.append(observation_images)

        # visualize the frame
        visualizer = o3d.visualization.draw_geometries([frame, trajectory_frame, origin_frame] + point_clouds)

    # show the gaussian point cloud
    if args.show_gaussian_point_cloud:
        # load the reconstructed point cloud
        reconstructed_point_cloud_path = os.path.join(args.reconstructed_data_path, args.task + "_episode"+str(args.original_episode)+"_start", "output", "objects_ply")

        # get ply file in the path
        ply_files = [f for f in os.listdir(reconstructed_point_cloud_path) if f.endswith('.ply')]

        # for each ply file load the gaussian splatting point cloud
        for ply_file in ply_files:
            gaussian_point_cloud = os.path.join(reconstructed_point_cloud_path, ply_file)
            xyz, features_dc, features_extra, scales, rots, opacities = load_ply(gaussian_point_cloud)

            # create the point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(features_dc[:, :, 0])

            # visualize the point cloud
            visualizer = o3d.visualization.draw_geometries([pcd, origin_frame])

        # show images using matplotlib in the same figure for each observation
    if args.show_images:
        for observation_images in images:

            fig, axs = plt.subplots(len(observation_images), 2)
            for i, (color_image, depth_image) in enumerate(observation_images):
                axs[i][0].imshow(color_image)
                axs[i][0].set_title("Color")
                axs[i][1].imshow(depth_image)
                axs[i][1].set_title("Depth")
            plt.show()
            plt.close()