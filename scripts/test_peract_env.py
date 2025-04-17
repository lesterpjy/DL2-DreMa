#!/usr/bin/env python3
"""
Script to verify the PerAct environment and installed packages with GPU isolation.
This addresses GLIBC version conflicts when running in Singularity/Apptainer.
"""

import os
import sys
import subprocess
import importlib.util
import platform


def print_section(title):
    """Print a section title with separation lines."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def check_package(package_name):
    """Check if a package is installed and get its version."""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False, None

        # Try different ways of getting the version
        try:
            module = importlib.import_module(package_name)
            if hasattr(module, "__version__"):
                version = module.__version__
            elif hasattr(module, "VERSION"):
                version = module.VERSION
            elif hasattr(module, "version"):
                version = module.version
            else:
                # Try to get version using pip
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "show", package_name],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    for line in result.stdout.split("\n"):
                        if line.startswith("Version:"):
                            version = line.split(":", 1)[1].strip()
                            break
                    else:
                        version = "Unknown"
                except subprocess.CalledProcessError:
                    version = "Unknown"

            return True, version
        except Exception as e:
            return True, f"Error getting version: {str(e)}"

    except Exception as e:
        return False, str(e)


# Check system information
print_section("System Information")
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")
print(f"System: {platform.system()} {platform.release()}")

# Check CUDA availability
print_section("CUDA Information")
try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"Error checking CUDA: {e}")

# Check environment variables
print_section("Environment Variables")
important_vars = [
    "PYTHONPATH",
    "PERACT_ROOT",
    "COPPELIASIM_ROOT",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "PATH",
]
for var in important_vars:
    print(f"{var}: {os.environ.get(var, 'Not set')}")

# Check critical packages
print_section("Critical Packages")
packages_to_check = [
    # Core packages
    "torch",
    "torchvision",
    "pytorch3d",
    "numpy",
    "scipy",
    # PerAct-specific
    "clip",
    "einops",
    "hydra",
    "transformers",
    "pytorch3d",
    "trimesh",
    # RLBench and PyRep (imported differently below)
    "rlbench",
    "pyrep",
    # YARR
    "yarr",
]

for pkg in packages_to_check:
    installed, version = check_package(pkg)
    status = "✅ Installed" if installed else "❌ Not installed"
    version_str = f"(version: {version})" if version else ""
    print(f"{pkg}: {status} {version_str}")

# Check if main components are properly installed
print_section("Component Availability (Package Structure Check Only)")

# This section checks only package existence without initializing GPU-dependent code
print("Checking component file structure (avoiding OpenGL/GPU initialization)...")

# Check PyRep package existence without initializing
print("\nChecking PyRep package structure...")
try:
    # Check for PyRep module files without importing
    pyrep_path = importlib.util.find_spec("pyrep")
    if pyrep_path and os.path.exists(pyrep_path.origin):
        print(f"✅ PyRep package found at {os.path.dirname(pyrep_path.origin)}")
        # Check for key PyRep files
        backend_path = os.path.join(os.path.dirname(pyrep_path.origin), "backend")
        if os.path.exists(backend_path):
            print(f"✅ PyRep backend directory exists at {backend_path}")
        else:
            print("❌ PyRep backend directory not found")
    else:
        print("❌ PyRep package not found")
except Exception as e:
    print(f"❌ Error checking PyRep package: {e}")

# Check RLBench package existence without initializing
print("\nChecking RLBench package structure...")
try:
    # Check for RLBench module files without importing
    rlbench_path = importlib.util.find_spec("rlbench")
    if rlbench_path and os.path.exists(rlbench_path.origin):
        print(f"✅ RLBench package found at {os.path.dirname(rlbench_path.origin)}")
        # Check for key RLBench files
        env_path = os.path.join(os.path.dirname(rlbench_path.origin), "environment.py")
        if os.path.exists(env_path):
            print(f"✅ RLBench environment module exists at {env_path}")
        else:
            print("❌ RLBench environment module not found")
    else:
        print("❌ RLBench package not found")
except Exception as e:
    print(f"❌ Error checking RLBench package: {e}")

# Check YARR
print("\nTesting YARR import...")
try:
    import yarr

    print("✅ YARR is properly installed")
except Exception as e:
    print(f"❌ YARR error: {e}")

# Check PerAct modules (avoiding OpenGL-dependent modules)
print("\nChecking PerAct package structure...")
try:
    # Change to the PerAct root directory
    peract_root = os.environ.get("PERACT_ROOT", "/root/install/peract")
    os.chdir(peract_root)

    # Check that key directories and files exist without importing
    agent_dir = os.path.join(peract_root, "agents")
    helpers_dir = os.path.join(peract_root, "helpers")

    if os.path.exists(agent_dir):
        print(f"✅ PerAct agents directory exists at {agent_dir}")
        # Check for key agent files
        peract_bc_dir = os.path.join(agent_dir, "peract_bc")
        if os.path.exists(peract_bc_dir):
            print(f"✅ peract_bc directory exists at {peract_bc_dir}")
        else:
            print("❌ peract_bc directory not found")
    else:
        print("❌ PerAct agents directory not found")

    if os.path.exists(helpers_dir):
        print(f"✅ PerAct helpers directory exists at {helpers_dir}")
        # Check for key helper files
        utils_file = os.path.join(helpers_dir, "utils.py")
        if os.path.exists(utils_file):
            print(f"✅ helpers/utils.py exists at {utils_file}")
        else:
            print("❌ helpers/utils.py not found")
    else:
        print("❌ PerAct helpers directory not found")

    # Try importing modules that don't rely on OpenGL
    print("\nTesting safe PerAct module imports...")
    # These should be safe to import without OpenGL
    try:
        from helpers import utils

        print("✅ helpers.utils imported successfully")
    except Exception as e:
        print(f"❌ Error importing helpers.utils: {e}")

except Exception as e:
    print(f"❌ Error checking PerAct structure: {e}")

# Check if CoppeliaSim is available
print("\nChecking CoppeliaSim installation...")
if os.environ.get("COPPELIASIM_ROOT"):
    coppeliasim_path = os.environ.get("COPPELIASIM_ROOT")
    if os.path.exists(coppeliasim_path):
        print(f"✅ CoppeliaSim found at {coppeliasim_path}")

        # Check for the CoppeliaSim executable
        executable = os.path.join(coppeliasim_path, "coppeliaSim.sh")
        if os.path.exists(executable):
            print(f"✅ CoppeliaSim executable found at {executable}")
        else:
            print("❌ CoppeliaSim executable not found")
    else:
        print(f"❌ CoppeliaSim directory not found at {coppeliasim_path}")
else:
    print("❌ COPPELIASIM_ROOT environment variable not set")

# Check if eval.py exists and is valid syntax
print("\nChecking eval.py...")
try:
    peract_root = os.environ.get("PERACT_ROOT", "/root/install/peract")
    eval_path = os.path.join(peract_root, "eval.py")
    if os.path.exists(eval_path):
        print(f"✅ eval.py found at {eval_path}")

        # Check if the script can be imported (syntax check)
        with open(eval_path, "r") as f:
            script_content = f.read()
        compile(script_content, eval_path, "exec")
        print("✅ eval.py syntax is valid")
    else:
        print(f"❌ eval.py not found at {eval_path}")
except Exception as e:
    print(f"❌ Error checking eval.py: {e}")

# Display note about GLIBC version issue
print_section("Note on GLIBC Version")
print(
    "This script has been modified to avoid GLIBC version conflicts with OpenGL/GPU libraries."
)
print(
    "These conflicts typically occur when running in Singularity/Apptainer with the --nv flag."
)
print(
    "While some GPU-dependent imports may fail in the test environment, they should work"
)
print("correctly when running the actual PerAct code on compatible hardware.")

# Overall summary
print_section("Overall Summary")
print("This script checked the basic environment setup for PerAct.")
print(
    "PerAct is NOT a standard Python package that can be imported directly with 'import peract'."
)
print(
    "Instead, it's a collection of modules that must be imported from specific paths."
)
print(
    "If all component file structure checks passed, your container is likely set up correctly."
)
print(
    "Full OpenGL/GPU functionality may only be testable during actual task execution."
)
