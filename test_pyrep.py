#!/usr/bin/env python3
"""Test basic PyRep functionality with CoppeliaSim in headless mode for cluster environments."""

import os
import sys
import time


def print_section(title):
    """Print a section title with separation lines."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


# Set environment variable to force headless mode for Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

print_section("Testing PyRep and CoppeliaSim in Headless Mode")
try:
    from pyrep import PyRep

    # Check if CoppeliaSim path is set
    coppeliasim_path = os.environ.get("COPPELIASIM_ROOT")
    if not coppeliasim_path:
        print("ERROR: COPPELIASIM_ROOT not set")
        sys.exit(1)

    print(f"CoppeliaSim path: {coppeliasim_path}")

    # Try to launch a headless instance
    print("Launching headless CoppeliaSim instance...")
    pr = PyRep()
    pr.launch(os.path.join(coppeliasim_path, "coppeliaSim.sh"), headless=True)
    print("Successfully launched CoppeliaSim in headless mode!")

    # Start the simulation
    print("Starting simulation...")
    pr.start()
    print("Simulation started successfully!")

    # Let it run for a moment
    time.sleep(2)

    # Stop the simulation
    print("Stopping simulation...")
    pr.stop()
    print("Simulation stopped successfully!")

    # Shutdown
    print("Shutting down CoppeliaSim...")
    pr.shutdown()
    print("CoppeliaSim shut down successfully!")

    print("\n✅ PyRep and CoppeliaSim headless test PASSED!")
except Exception as e:
    print(f"\n❌ PyRep and CoppeliaSim headless test FAILED: {e}")
    import traceback

    traceback.print_exc()

# Now test RLBench environment with PyRep in headless mode
print_section("Testing RLBench with PyRep in Headless Mode")
try:
    # Change to the PerAct root directory to ensure paths are correct
    peract_root = os.environ.get("PERACT_ROOT", "/root/install/peract")
    os.chdir(peract_root)

    # Import RLBench environment components
    print("Importing RLBench environment components...")
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks import ReachTarget

    print("✅ RLBench environment components imported successfully")

    # Create a simple headless environment configuration
    print("\nSetting up and launching a simple RLBench headless environment...")
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.joint_velocities = True

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    # Create the environment
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True,  # This is critical for cluster environments
    )

    # Launch the environment
    print("Launching RLBench environment...")
    env.launch()
    print("✅ RLBench environment launched successfully in headless mode")

    # Load a task
    print("Loading ReachTarget task...")
    task = env.get_task(ReachTarget)
    print("✅ RLBench task loaded successfully")

    # Reset the task
    print("Resetting task...")
    descriptions, obs = task.reset()
    print("✅ RLBench task reset successful")

    # Clean up
    print("Shutting down environment...")
    env.shutdown()
    print("✅ RLBench environment shut down successfully")

    print("\n✅ RLBench with PyRep headless test PASSED!")
except Exception as e:
    print(f"\n❌ RLBench with PyRep headless test FAILED: {e}")
    import traceback

    traceback.print_exc()

# Overall summary
print_section("Overall Summary")
print(
    "This script tested PyRep and RLBench in headless mode, which is required for cluster environments."
)
print(
    "If all tests passed, your container is properly set up for running PerAct on a compute cluster."
)
print(
    "Remember to always use headless=True when running on SLURM or other cluster environments."
)
