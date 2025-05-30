import pybullet as p
import pybullet_data
import time
import os

print("--- MINIMAL PYBULLET TEST SCRIPT STARTING ---")
print(f"Python version: {os.sys.version}")
print(f"PyBullet data path: {pybullet_data.getDataPath()}")

client_id = -1
try:
    print("Attempting to connect to PyBullet (DIRECT mode)...")
    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        print("❌ FAILED to connect to PyBullet.")
        exit(1)
    print(f"✅ Connected to PyBullet with client ID: {client_id}")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    print("Loading plane.urdf...")
    plane_id = p.loadURDF("plane.urdf")
    print(f"✅ Loaded plane with ID: {plane_id}")

    # Path to Panda URDF inside Singularity (as bound by sbatch)
    panda_urdf_path_inside_container = "/urdfs/panda/panda.urdf"
    print(f"Attempting to load Panda from: {panda_urdf_path_inside_container}")
    # p.setAdditionalSearchPath("/urdfs/panda") # This might not be needed if full path is given
    panda_id = p.loadURDF(panda_urdf_path_inside_container, useFixedBase=True)
    print(f"✅ Loaded Panda with ID: {panda_id}")
    print(f"Panda has {p.getNumJoints(panda_id)} joints.")

    print("Stepping simulation a few times...")
    for i in range(100):
        p.stepSimulation()
        if i % 20 == 0:
            print(f"  Sim step {i}")
        # time.sleep(0.01) # Not needed for DIRECT mode, but can slow it down for observation

    print("✅ Minimal PyBullet simulation steps completed.")

except Exception as e:
    print(f"❌ ERROR during minimal PyBullet test: {e}")
    import traceback
    traceback.print_exc()

finally:
    if client_id >= 0 and p.isConnected(client_id):
        print("Disconnecting from PyBullet...")
        p.disconnect(client_id)
    elif p.isConnected(): # General check
        print("Disconnecting from PyBullet (general)...")
        p.disconnect()
    print("--- MINIMAL PYBULLET TEST SCRIPT FINISHED ---")