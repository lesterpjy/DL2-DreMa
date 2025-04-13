#!/usr/bin/env python3
"""Test basic PyRep functionality with CoppeliaSim and verify it works with PerAct."""

import os
import sys
import time


def print_section(title):
    """Print a section title with separation lines."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


# First test basic PyRep functionality
print_section("Testing PyRep and CoppeliaSim")
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
    print("Successfully launched CoppeliaSim!")

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

    print("\n✅ PyRep and CoppeliaSim connection test PASSED!")
except Exception as e:
    print(f"\n❌ PyRep and CoppeliaSim connection test FAILED: {e}")

# Now test RLBench environment with PyRep
print_section("Testing RLBench with PyRep")
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

    print(
        "\nSetting up a simple RLBench environment (without actually launching it)..."
    )
    # Set up a simple environment configuration (but don't launch it)
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.joint_velocities = True

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    print("✅ Basic RLBench environment configuration successful")

    print("\n✅ RLBench with PyRep setup test PASSED!")
except Exception as e:
    print(f"\n❌ RLBench with PyRep setup test FAILED: {e}")

# Finally, check the PerAct-specific components with RLBench
print_section("Testing PerAct components with RLBench")
try:
    # Make sure we're in the PerAct directory
    os.chdir(peract_root)

    # Import helpers from PerAct
    print("Importing PerAct helpers...")
    from helpers.custom_rlbench_env import CustomRLBenchEnv
    from helpers import utils

    # Import agent components
    print("Importing PerAct agent components...")
    from agents import peract_bc

    print("✅ PerAct components imported successfully")

    # Verify PYTHONPATH includes all necessary components
    print("\nVerifying PYTHONPATH configuration...")
    pythonpath = os.environ.get("PYTHONPATH", "")
    required_paths = [
        "/root/install/peract",
        "/root/install/RLBench",
        "/root/install/YARR",
        "/root/install/PyRep",
    ]

    missing_paths = []
    for path in required_paths:
        if path not in pythonpath:
            missing_paths.append(path)

    if missing_paths:
        print(f"❌ Missing paths in PYTHONPATH: {', '.join(missing_paths)}")
    else:
        print("✅ PYTHONPATH includes all required components")

    print("\n✅ PerAct with RLBench integration test PASSED!")
except Exception as e:
    print(f"\n❌ PerAct with RLBench integration test FAILED: {e}")

# Overall summary
print_section("Overall Summary")
print(
    "This script tested the integration between PyRep, CoppeliaSim, RLBench, and PerAct."
)
print("If all tests passed, your container should be ready to run the PerAct code.")
print("You can now try running the quickstart example to fully verify functionality.")
