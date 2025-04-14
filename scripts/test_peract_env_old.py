#!/usr/bin/env python3
"""
Script to verify the PerAct environment and installed packages.
This checks all the critical components needed for PerAct to work.
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
    "pyrender",
    # RLBench and PyRep
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
print_section("Component Availability")

# Check PyRep
print("Testing PyRep import...")
try:
    import pyrep
    from pyrep import PyRep

    print("✅ PyRep is properly installed")
except Exception as e:
    print(f"❌ PyRep error: {e}")

# Check RLBench
print("\nTesting RLBench import...")
try:
    import rlbench
    from rlbench.environment import Environment

    print("✅ RLBench is properly installed")
except Exception as e:
    print(f"❌ RLBench error: {e}")

# Check YARR
print("\nTesting YARR import...")
try:
    import yarr

    print("✅ YARR is properly installed")
except Exception as e:
    print(f"❌ YARR error: {e}")

# Check PerAct modules (as used in eval.py)
print("\nTesting PerAct modules...")
try:
    # Change to the PerAct root directory
    os.chdir(os.environ.get("PERACT_ROOT", "/root/install/peract"))

    # Import specific modules as used in eval.py
    from agents import peract_bc

    print("✅ agents.peract_bc import successful")

    from agents import c2farm_lingunet_bc

    print("✅ agents.c2farm_lingunet_bc import successful")

    from helpers import utils

    print("✅ helpers.utils import successful")

    from helpers.custom_rlbench_env import CustomRLBenchEnv

    print("✅ helpers.custom_rlbench_env import successful")

    print("✅ All PerAct modules are properly accessible")
except Exception as e:
    print(f"❌ PerAct modules error: {e}")

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

# Check if eval.py exists and is runnable
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
    "If all component imports show as successful, your container is set up correctly."
)
print(
    "If any errors appeared, they will need to be addressed before running the quickstart test."
)
