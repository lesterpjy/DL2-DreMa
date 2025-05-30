import sys
import os
import pybullet as p
import pybullet_data
import time
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
from scipy.spatial.transform import Rotation # Needed for discrete_euler_to_quaternion

# --- Helper function from PerAct's utils (copied for direct use) ---
def discrete_euler_to_quaternion(discrete_euler_indices, resolution_degrees):
    """
    Converts discrete Euler angle indices to a quaternion.
    Args:
        discrete_euler_indices: np.array of 3 integer indices for x, y, z rotations.
        resolution_degrees: The degree resolution for each discrete step.
    Returns:
        np.array: A quaternion [x, y, z, w].
    """
    # Ensure input is a numpy array for vectorized operations
    discrete_euler_indices = np.asarray(discrete_euler_indices)
    euler_angles_degrees = (discrete_euler_indices * resolution_degrees) - 180.0
    # Scipy's Rotation expects radians if not specified, or use degrees=True
    # The order 'xyz' means intrinsic rotations: first about x, then new y, then new z.
    # Or fixed axes Z, then Y, then X if interpreting that way. Match PerAct's convention.
    # Assuming 'xyz' intrinsic as commonly used.
    return Rotation.from_euler('xyz', euler_angles_degrees, degrees=True).as_quat()

# --- Ensure correct import paths for PerAct components ---
try:
    from agents.peract_bc.perceiver_lang_io import PerceiverVoxelLangEncoder
    from agents.peract_bc.qattention_peract_bc_agent import QAttentionPerActBCAgent
    from helpers.clip.core.clip import tokenize
    print("✅ Successfully imported PerAct components (Agent, Encoder, tokenize).")
except ImportError as e:
    print(f"❌ ImportError during PerAct component import: {e}")
    print("   PYTHONPATH:", os.environ.get('PYTHONPATH'))
    print("   sys.path:", sys.path)
    exit(1)
except Exception as e:
    print(f"❌ Some other exception during PerAct component import: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# --- PyTorch CUDA Diagnostics ---
print(f"--- PyTorch/CUDA Diagnostics ---")
pytorch_version = torch.__version__
print(f"PyTorch version: {pytorch_version}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        try:
            current_dev_idx = torch.cuda.current_device()
            print(f"Current CUDA device index: {current_dev_idx}")
            print(f"Device name: {torch.cuda.get_device_name(current_dev_idx)}")
        except Exception as e_cuda_info:
            print(f"Error getting CUDA device info: {e_cuda_info}")
else:
    print("CUDA is NOT available to PyTorch. Script will run on CPU.")
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")
print(f"--- End PyTorch/CUDA Diagnostics ---")

IMAGE_FEATURE_CHANNELS = 3 # For RGB

# --- Configuration (CRITICAL: Review placeholders based on your full training config) ---
cfg_perceiver_encoder = {
    'depth': 6,
    'iterations': 1,
    'voxel_size': 100,
    'initial_dim': IMAGE_FEATURE_CHANNELS + 7, # (3 for input RGB) + 3 (coords) + 3 (grid_pos_enc) + 1 (occupancy) = 10
    'low_dim_size': 4,
    'num_latents': 2048,
    'latent_dim': 512,                      ### USER: VERIFY/ADJUST ###
    'cross_heads': 1,
    'latent_heads': 8,
    'cross_dim_head': 64,
    'latent_dim_head': 64,
    'voxel_patch_size': 5,                  ### USER: VERIFY/ADJUST ###
    'voxel_patch_stride': 5,                ### USER: VERIFY/ADJUST ###
    'im_channels': 64,                      # Voxel Feature Dim from your info
    'activation': 'relu',                   ### USER: VERIFY/ADJUST ###
    'pos_encoding_with_lang': False,
    'lang_fusion_type': 'seq',              ### USER: VERIFY/ADJUST ###
    'layer':0, 'num_rotation_classes':72, 'num_grip_classes':2, 'num_collision_classes':2,
    'input_axis':3, 'weight_tie_layers':False, 'input_dropout':0.1, 'attn_dropout':0.1,
    'decoder_dropout':0.0, 'no_skip_connection':False, 'no_perceiver':False,
    'no_language':False,                    ### USER: VERIFY/ADJUST ###
    'final_dim':64,
}
cfg_agent = {
    'layer': 0,
    'coordinate_bounds': [-0.5, -0.5, 0.0, 0.5, 0.5, 1.0], ### USER: VERIFY/ADJUST ###
    'perceiver_encoder': None,
    'camera_names': ['front', 'left_shoulder', 'right_shoulder', 'wrist'],
    'batch_size': 1,
    'voxel_size': cfg_perceiver_encoder['voxel_size'],
    'bounds_offset': 0.05,                              ### USER: VERIFY/ADJUST ###
    'voxel_feature_size': IMAGE_FEATURE_CHANNELS, # Number of channels for input image features to VoxelGrid
    'image_crop_size': 128,                             ### USER: VERIFY/ADJUST ###
    'num_rotation_classes': cfg_perceiver_encoder['num_rotation_classes'],
    'rotation_resolution': 360.0 / cfg_perceiver_encoder['num_rotation_classes'] if cfg_perceiver_encoder['num_rotation_classes'] > 0 else 5.0,
    'include_low_dim_state': True if cfg_perceiver_encoder['low_dim_size'] > 0 else False,
    'image_resolution': [128, 128],                     ### USER: VERIFY/ADJUST ###
}
proprio_dim = cfg_perceiver_encoder['low_dim_size']

# --- Instantiate PerceiverVoxelLangEncoder ---
print("Attempting to instantiate PerceiverVoxelLangEncoder...")
try:
    perceiver_encoder_model = PerceiverVoxelLangEncoder(**cfg_perceiver_encoder).to(device)
    print(f"  Instantiated PerceiverVoxelLangEncoder's 'pos_encoding' shape: {perceiver_encoder_model.pos_encoding.shape}")
    print("✅ PerceiverVoxelLangEncoder instantiated.")
except Exception as e:
    print(f"❌ Error instantiating PerceiverVoxelLangEncoder: {e}")
    import traceback; traceback.print_exc(); exit(1)

# --- Instantiate QAttentionPerActBCAgent ---
cfg_agent['perceiver_encoder'] = perceiver_encoder_model
print("Attempting to instantiate QAttentionPerActBCAgent...")
try:
    agent = QAttentionPerActBCAgent(**cfg_agent)
    agent.build(training=False, device=device)
    print("✅ QAttentionPerActBCAgent instantiated and built successfully.")
except Exception as e:
    print(f"❌ Error instantiating or building QAttentionPerActBCAgent: {e}")
    import traceback; traceback.print_exc(); exit(1)

# --- Load Checkpoint ---
ckpt_path = "/root/dream-team/peract_weights/ckpts/multi/PERACT_BC/seed0/weights/600000/QAttentionAgent_layer0.pt" ### USER: VERIFY PATH ###
print(f"Loading checkpoint from: {ckpt_path}")
if not os.path.exists(ckpt_path):
    print(f"❌ Checkpoint file NOT FOUND at {ckpt_path}"); exit(1)
try:
    print("Attempting to load weights using agent.load_weights()...")
    agent.load_weights(os.path.dirname(ckpt_path))
    print("✅ Checkpoint loaded successfully via agent.load_weights().")
    if hasattr(agent, '_q') and isinstance(agent._q, torch.nn.Module):
        agent._q.eval()
        print("✅ agent._q (QFunction) set to evaluation mode.")
    else:
        print("⚠️ Could not find agent._q or it's not an nn.Module for eval mode.")
except Exception as e:
    print(f"❌ Error during checkpoint loading or setting to eval mode: {e}")
    import traceback; traceback.print_exc(); exit(1)

# --- PyBullet Setup & Observation Generation ---
print("Setting up PyBullet and generating observation...")
client_id = -1
try:
    client_id = p.connect(p.DIRECT)
    if client_id < 0: raise Exception("Failed to connect to PyBullet.")
    p.setGravity(0,0,-9.8); p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    panda = p.loadURDF("/urdfs/panda/panda.urdf",basePosition=[0,0,0.1], useFixedBase=True)
    print(f"✅ PyBullet scene loaded. Panda joints: {p.getNumJoints(panda)}")
    for _ in range(10): p.stepSimulation()

    cam_target_pos = [0.0, 0.0, 0.5]; cam_dist = 1.2; cam_yaw = 60; cam_pitch = -35
    img_width = cfg_agent['image_resolution'][1]; img_height = cfg_agent['image_resolution'][0]
    view_m = p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_dist, cam_yaw, cam_pitch, 0, 2)
    proj_m = p.computeProjectionMatrixFOV(60, float(img_width)/img_height, 0.1, 100.0)
    _, _, rgba_front, _, _ = p.getCameraImage(img_width,img_height,view_m,proj_m,renderer=p.ER_TINY_RENDERER)

    rgb_pybullet_front_np = np.array(rgba_front)[:,:,:3] # Numpy array for saving
    img_pil_front = Image.fromarray(rgb_pybullet_front_np) # PIL Image for transforms
    print(f"✅ Front Image rendered, shape: {rgb_pybullet_front_np.shape}")
    
    # --- SAVE THE RENDERED IMAGE ---
    try:
        # Create a directory for outputs if it doesn't exist (relative to WORKDIR in sbatch)
        output_image_dir = "pybullet_renders"
        os.makedirs(output_image_dir, exist_ok=True)
        image_save_path = os.path.join(output_image_dir, "pybullet_front_view.png")
        
        # Save the numpy array directly using Pillow (or opencv if you prefer)
        Image.fromarray(rgb_pybullet_front_np).save(image_save_path)
        print(f"✅ Rendered PyBullet image saved to: {image_save_path}")
        # If running in a Slurm job, this path will be inside the WORKDIR on the cluster.
        # You'll need to copy it from there to your local machine to view it.
        # Or, if your WORKDIR is a shared filesystem, you can access it directly.
    except Exception as e_save:
        print(f"⚠️ Warning: Could not save PyBullet image: {e_save}")
    
     # --- END IMAGE SAVING ---

    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
    
    observation_for_act = {}
    
    obs_front_rgb_tensor_inner = img_transform(rgb_pybullet_front_np).unsqueeze(0).to(device)
    observation_for_act['front_rgb'] = obs_front_rgb_tensor_inner.unsqueeze(0)
    observation_for_act['front_point_cloud'] = torch.zeros_like(obs_front_rgb_tensor_inner).unsqueeze(0).to(device)
    print(f"Input 'front_rgb' shape for agent.act: {observation_for_act['front_rgb'].shape}")

    for cam_name in ['left_shoulder', 'right_shoulder', 'wrist']:
        dummy_cam_tensor_inner = torch.zeros_like(obs_front_rgb_tensor_inner)
        observation_for_act[f'{cam_name}_rgb'] = dummy_cam_tensor_inner.unsqueeze(0)
        observation_for_act[f'{cam_name}_point_cloud'] = dummy_cam_tensor_inner.unsqueeze(0)
        print(f"Input '{cam_name}_rgb' shape for agent.act: {observation_for_act[f'{cam_name}_rgb'].shape}")

    if cfg_agent['include_low_dim_state']:
        proprio_tensor_inner = torch.rand(1, proprio_dim, device=device)
        observation_for_act['low_dim_state'] = proprio_tensor_inner.unsqueeze(0)
    else:
        observation_for_act['low_dim_state'] = None
    print(f"Input 'low_dim_state' shape: {observation_for_act['low_dim_state'].shape if observation_for_act['low_dim_state'] is not None else 'None'}")

    dummy_text = "put the red block in the green bowl"
    lang_tokens_tensor_inner = tokenize(dummy_text).to(device)
    observation_for_act['lang_goal_tokens'] = lang_tokens_tensor_inner.unsqueeze(0)
    print(f"Input 'lang_goal_tokens' shape: {observation_for_act['lang_goal_tokens'].shape}")

    observation_for_act['prev_layer_voxel_grid'] = None
    observation_for_act['prev_layer_bounds'] = None
    print("Observation dict prepared with 'batch-of-batches' structure for agent.act().")

except Exception as e:
    print(f"❌ Error during PyBullet setup or observation generation: {e}")
    import traceback; traceback.print_exc()
    if client_id >=0 and p.isConnected(client_id): p.disconnect(client_id)
    exit(1)

# --- Get Action from Agent and Interpret It ---
print("Attempting to get action from agent...")
with torch.no_grad():
    try:
        act_result = agent.act(step=0, observation=observation_for_act, deterministic=True)
        
        discrete_action_tuple = act_result.action
        print(f"✅ Raw discrete_action_tuple: {discrete_action_tuple}")

        continuous_translation = None
        if act_result.observation_elements and 'attention_coordinate' in act_result.observation_elements:
            continuous_translation_tensor = act_result.observation_elements['attention_coordinate']
            if continuous_translation_tensor is not None:
                continuous_translation = continuous_translation_tensor.cpu().numpy()
                if continuous_translation.shape[0] == 1: # Remove outer batch dim if present
                    continuous_translation = continuous_translation[0]
                print(f"  Interpreted Continuous Translation: {continuous_translation}")
        else:
            print("  Continuous Translation ('attention_coordinate') not found or is None in act_result.observation_elements.")

        if discrete_action_tuple and isinstance(discrete_action_tuple, tuple):
            if len(discrete_action_tuple) > 0 and discrete_action_tuple[0] is not None:
                discrete_trans_coords_tensor = discrete_action_tuple[0]
                discrete_trans_coords = discrete_trans_coords_tensor.cpu().numpy()
                if discrete_trans_coords.shape[0] == 1: # Remove outer batch dim
                    discrete_trans_coords = discrete_trans_coords[0]
                print(f"  Discrete Translation Voxel Coords: {discrete_trans_coords}")
            else:
                print("  Discrete Translation Voxel Coords: None or not found in action_tuple[0]")

            if len(discrete_action_tuple) > 1 and discrete_action_tuple[1] is not None:
                rot_grip_indices_tensor = discrete_action_tuple[1]
                # rot_grip_indices_tensor is likely shape [1, 4] (batch_size, num_actions)
                rot_grip_indices = rot_grip_indices_tensor.cpu().numpy()[0] # Get the [idx_r_x, idx_r_y, idx_r_z, idx_grip]

                discrete_rotation_indices = rot_grip_indices[:3].astype(int) # Ensure they are int for multiplication
                gripper_action_index = int(rot_grip_indices[3])

                print(f"  Discrete Rotation Indices (for R,P,Y or X,Y,Z): {discrete_rotation_indices}")
                
                rotation_resolution_deg = cfg_agent['rotation_resolution']
                raw_euler_angles_deg = discrete_rotation_indices * rotation_resolution_deg
                print(f"    Raw Euler Angles (degrees, approx based on resolution): {raw_euler_angles_deg}")

                # Use the imported discrete_euler_to_quaternion function
                quaternion = discrete_euler_to_quaternion(discrete_rotation_indices, rotation_resolution_deg)
                print(f"    Predicted Quaternion: {quaternion}")
                
                print(f"  Discrete Gripper Action Index: {gripper_action_index} (e.g., 0=closed, 1=open or vice-versa)")
            else:
                print("  Rotation and Grip Indices: None or not found in action_tuple[1]")
            
            if len(discrete_action_tuple) > 2 and discrete_action_tuple[2] is not None:
                ignore_collision_index_tensor = discrete_action_tuple[2]
                # ignore_collision_index_tensor is likely shape [1,1]
                ignore_collision_index = int(ignore_collision_index_tensor.cpu().numpy()[0][0])
                print(f"  Discrete Ignore Collisions Index: {ignore_collision_index} (e.g., 0=consider, 1=ignore)")
            else:
                print("  Ignore Collisions Index: None or not found in action_tuple[2]")
        else:
             print(f"  Predicted action_tuple is not a tuple or is empty: {discrete_action_tuple}")

    except Exception as e:
        print(f"❌ Error during agent.act() or action interpretation: {e}")
        import traceback; traceback.print_exc()

# --- Cleanup ---
if client_id >=0 and p.isConnected(client_id):
    print("Disconnecting from PyBullet...")
    p.disconnect(client_id)
print("✅ PyBullet smoke test script finished.")
