"""
This code is a modified version of: https://github.com/stepjam/ARM/blob/main/arm/demo_loading_utils.py
"""

import copy
import numpy as np


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs["gripper_open"] == demo[i + 1]["gripper_open"] and
             obs["gripper_open"] == demo[i - 1]["gripper_open"] and
             demo[i - 2]["gripper_open"] == demo[i - 1]["gripper_open"]))
    small_delta = np.allclose(obs['joint_velocities'], 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and (not next_is_not_final) and gripper_state_no_change)

    return stopped

def compute_keypoints(demo):
    keypoints = []
    episode_after_first_keypoint = []
    demo_after_first_keypoint = []

    inserted = False
    prev_gripper_open = demo[0]["gripper_open"]
    stopped_buffer = 0
    stopping_delta = 0.1
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1

        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs["gripper_open"] != prev_gripper_open or last or stopped):
            keypoints.append(i)
            inserted = True
        if inserted:
            episode_after_first_keypoint.append(i)
        prev_gripper_open = obs["gripper_open"]

    if len(keypoints) > 1 and (keypoints[-1] - 1) == keypoints[-2]:
        keypoints.pop(-2)

    for i in episode_after_first_keypoint:
        demo_after_first_keypoint.append(copy.deepcopy(demo[i]))

    return keypoints, episode_after_first_keypoint, demo_after_first_keypoint