#!/usr/bin/env python3
"""Test basic PyRep functionality with CoppeliaSim."""

import os
import sys
from pyrep import PyRep

print("Testing PyRep connection to CoppeliaSim...")
try:
    # Create a PyRep instance
    pr = PyRep()

    # Check if CoppeliaSim path is set
    coppeliasim_path = os.environ.get("COPPELIASIM_ROOT")
    if not coppeliasim_path:
        print("ERROR: COPPELIASIM_ROOT not set")
        sys.exit(1)

    print(f"CoppeliaSim path: {coppeliasim_path}")

    # Try to launch a headless instance (should work even on compute node)
    print("Launching headless CoppeliaSim instance...")
    pr.launch(coppeliasim_path + "/coppeliaSim.sh", headless=True)
    print("Successfully launched CoppeliaSim!")

    # Start the simulation
    print("Starting simulation...")
    pr.start()
    print("Simulation started successfully!")

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
