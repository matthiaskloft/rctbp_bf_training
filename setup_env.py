#!/usr/bin/env python3
"""
RCTBP BayesFlow Training - Environment Setup Script

Creates a Python virtual environment with GPU-enabled PyTorch and all dependencies.
Automatically detects CUDA version and installs appropriate PyTorch build.

Usage:
    python setup_env.py                 # Auto-detect GPU and create environment
    python setup_env.py --cpu-only      # Force CPU-only installation
    python setup_env.py --force         # Recreate environment from scratch
    python setup_env.py --verbose       # Show detailed installation output

Requirements:
    - Python 3.9 or higher
    - nvidia-smi (optional, for GPU support)
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Dict, Tuple


class ColoredOutput:
    """Handle colored terminal output with cross-platform support."""

    # ANSI color codes
    RED = "31"
    GREEN = "32"
    YELLOW = "33"
    BLUE = "34"
    CYAN = "36"
    BOLD = "1"

    def __init__(self):
        self.enabled = True
        # Disable Unicode symbols on Windows due to encoding issues
        # Use ASCII fallback characters instead
        self.use_unicode = platform.system() != "Windows"

        # Enable ANSI colors on Windows 10+
        if platform.system() == "Windows":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                self.enabled = False

    def colorize(self, text: str, color_code: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.enabled:
            return text
        return f"\033[{color_code}m{text}\033[0m"

    def success(self, text: str) -> str:
        """Green colored text."""
        prefix = "✓" if self.use_unicode else "[+]"
        return self.colorize(f"{prefix} {text}", self.GREEN)

    def error(self, text: str) -> str:
        """Red colored text."""
        prefix = "✗" if self.use_unicode else "[X]"
        return self.colorize(f"{prefix} {text}", self.RED)

    def warning(self, text: str) -> str:
        """Yellow colored text."""
        prefix = "⚠" if self.use_unicode else "[!]"
        return self.colorize(f"{prefix} {text}", self.YELLOW)

    def info(self, text: str) -> str:
        """Cyan colored text."""
        prefix = "ℹ" if self.use_unicode else "[i]"
        return self.colorize(f"{prefix} {text}", self.CYAN)

    def header(self, text: str) -> str:
        """Bold blue header."""
        return self.colorize(f"\n{'=' * 60}\n{text}\n{'=' * 60}", f"{self.BOLD};{self.BLUE}")


# Global ColoredOutput instance
output = ColoredOutput()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Set up Python virtual environment with GPU support for RCTBP BayesFlow Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect GPU and create environment
  python setup_env.py

  # Force CPU-only mode
  python setup_env.py --cpu-only

  # Recreate environment from scratch
  python setup_env.py --force

  # Use specific Python version
  python setup_env.py --python 3.11

  # Custom environment name
  python setup_env.py --name my-env

  # Verbose output
  python setup_env.py --verbose
        """
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and recreate virtual environment if it exists"
    )

    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Install CPU-only PyTorch (skip GPU detection)"
    )

    parser.add_argument(
        "--name",
        default="venv",
        help="Virtual environment name (default: venv)"
    )

    parser.add_argument(
        "--cuda-version",
        choices=["11.8", "12.6", "12.8", "cpu", "auto"],
        default="auto",
        help="Specific CUDA version to use (default: auto-detect)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed installation output"
    )

    return parser.parse_args()


def check_python_version() -> bool:
    """Check that Python version is 3.9 or higher."""
    if sys.version_info < (3, 9):
        print(output.error(f"Python 3.9 or higher required"))
        print(output.info(f"Current version: {sys.version}"))
        print(output.info("Download from: https://www.python.org/downloads/"))
        return False
    return True


def get_venv_paths(venv_dir: Path) -> Dict[str, str]:
    """Get platform-specific virtual environment paths."""
    if platform.system() == "Windows":
        return {
            "python": str(venv_dir / "Scripts" / "python.exe"),
            "pip": str(venv_dir / "Scripts" / "pip.exe"),
            "activate": str(venv_dir / "Scripts" / "activate.bat")
        }
    else:
        return {
            "python": str(venv_dir / "bin" / "python"),
            "pip": str(venv_dir / "bin" / "pip"),
            "activate": str(venv_dir / "bin" / "activate")
        }


def detect_cuda_version() -> Tuple[str, bool]:
    """
    Detect CUDA version from nvidia-smi output.

    Returns:
        Tuple of (cuda_version, detected) where:
        - cuda_version: str like "12.6" or "cpu"
        - detected: bool indicating if GPU was detected
    """
    print(output.info("Detecting CUDA version..."))

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(output.warning("nvidia-smi failed - using CPU-only PyTorch"))
            return ("cpu", False)

        # Parse CUDA version from output
        # Look for pattern like "CUDA Version: 12.4"
        cuda_match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)

        if not cuda_match:
            print(output.warning("Could not parse CUDA version - using CPU-only PyTorch"))
            return ("cpu", False)

        cuda_version = cuda_match.group(1)
        print(output.success(f"Detected CUDA version: {cuda_version}"))
        return (cuda_version, True)

    except FileNotFoundError:
        print(output.warning("nvidia-smi not found - using CPU-only PyTorch"))
        return ("cpu", False)
    except subprocess.TimeoutExpired:
        print(output.warning("nvidia-smi timed out - using CPU-only PyTorch"))
        return ("cpu", False)
    except Exception as e:
        print(output.warning(f"Error detecting CUDA: {e} - using CPU-only PyTorch"))
        return ("cpu", False)


def map_cuda_to_pytorch_build(cuda_version: str) -> str:
    """
    Map detected CUDA version to PyTorch build version.

    PyTorch 2.7+ (as of 2025) supports:
    - cu118 (CUDA 11.8)
    - cu126 (CUDA 12.6)
    - cu128 (CUDA 12.8)

    CUDA is backward compatible, so we pick the highest
    PyTorch-supported version <= detected CUDA version.

    Args:
        cuda_version: CUDA version string like "12.4" or "cpu"

    Returns:
        PyTorch build version like "12.6" or "cpu"
    """
    if cuda_version == "cpu":
        return "cpu"

    # Parse major.minor version
    try:
        version_parts = cuda_version.split(".")
        cuda_major = int(version_parts[0])
        cuda_minor = int(version_parts[1])
    except (ValueError, IndexError):
        print(output.warning(f"Invalid CUDA version format: {cuda_version}"))
        return "cpu"

    # Map to PyTorch build (backward compatible)
    if cuda_major >= 13:
        # CUDA 13.x - use cu128 (latest)
        selected = "12.8"
        print(output.info(f"CUDA {cuda_version} detected, using PyTorch CUDA 12.8 build"))
    elif cuda_major == 12:
        if cuda_minor >= 8:
            selected = "12.8"
        elif cuda_minor >= 6:
            selected = "12.6"
        else:
            # CUDA 12.0-12.5: cu121/cu124 removed, use cu126 (backward compatible)
            selected = "12.6"
            print(output.info(f"CUDA {cuda_version} detected, using PyTorch CUDA 12.6 build (backward compatible)"))
    elif cuda_major == 11 and cuda_minor >= 8:
        selected = "11.8"
    else:
        print(output.warning(f"CUDA {cuda_version} is older than supported versions - using CPU"))
        return "cpu"

    print(output.info(f"Selected PyTorch CUDA version: {selected}"))
    return selected


def create_or_verify_venv(venv_dir: Path, force: bool) -> bool:
    """
    Create virtual environment or verify existing one.

    Args:
        venv_dir: Path to virtual environment directory
        force: If True, remove existing venv and recreate

    Returns:
        True if successful, False otherwise
    """
    if venv_dir.exists():
        if force:
            print(output.warning("Removing existing virtual environment (--force)"))
            try:
                shutil.rmtree(venv_dir)
            except PermissionError:
                print(output.error(f"Cannot remove {venv_dir} (may be in use)"))
                print(output.info("Try deactivating the environment first"))
                return False
        else:
            print(output.info(f"Using existing virtual environment: {venv_dir}"))
            return True

    print(output.info(f"Creating virtual environment: {venv_dir}"))
    try:
        venv.create(venv_dir, with_pip=True)
        print(output.success("Virtual environment created"))
        return True
    except Exception as e:
        print(output.error(f"Failed to create virtual environment: {e}"))
        return False


def run_command(cmd: list, description: str, verbose: bool = False, timeout: int = 600) -> bool:
    """
    Run a command with progress indication.

    Args:
        cmd: Command and arguments as list
        description: User-facing description
        verbose: Whether to show command output in real-time
        timeout: Timeout in seconds (default: 600 = 10 minutes)

    Returns:
        True if successful, False otherwise
    """
    print(output.info(f"{description}..."))

    try:
        if verbose:
            # Stream output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            for line in process.stdout:
                print(f"  {line.rstrip()}")
            process.wait()
            success = process.returncode == 0
        else:
            # Capture output, show only on error
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            success = result.returncode == 0

            if not success:
                print(output.error(f"Failed: {description}"))
                if result.stdout:
                    print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)

        if success:
            print(output.success(f"Completed: {description}"))
        return success

    except subprocess.TimeoutExpired:
        print(output.error(f"Timeout: {description}"))
        return False
    except Exception as e:
        print(output.error(f"Error running command: {e}"))
        return False


def install_pytorch(venv_python: str, pytorch_build: str, verbose: bool = False) -> bool:
    """
    Install PyTorch with appropriate CUDA support.

    Args:
        venv_python: Path to venv Python executable
        pytorch_build: PyTorch build version ("11.8", "12.6", "12.8", or "cpu")
        verbose: Show detailed output

    Returns:
        True if successful, False otherwise
    """
    # Build PyTorch index URL
    if pytorch_build == "cpu":
        index_url = "https://download.pytorch.org/whl/cpu"
        print(output.info("Installing CPU-only PyTorch"))
    else:
        cuda_tag = pytorch_build.replace(".", "")  # "12.6" -> "126"
        index_url = f"https://download.pytorch.org/whl/cu{cuda_tag}"
        print(output.info(f"Installing PyTorch with CUDA {pytorch_build}"))

    # First upgrade pip
    if not run_command(
        [venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        "Upgrading pip, setuptools, and wheel",
        verbose
    ):
        return False

    # Install PyTorch (use 30-minute timeout for large CUDA builds ~2.9 GB)
    cmd = [
        venv_python, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", index_url
    ]

    return run_command(cmd, "Installing PyTorch", verbose, timeout=1800)


def install_package_with_extras(venv_python: str, project_root: Path, verbose: bool = False) -> bool:
    """
    Install package in editable mode with dev and notebooks extras.

    Args:
        venv_python: Path to venv Python executable
        project_root: Path to project root directory
        verbose: Show detailed output

    Returns:
        True if successful, False otherwise
    """
    # Check that pyproject.toml exists
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.exists():
        print(output.error(f"pyproject.toml not found at {pyproject_path}"))
        print(output.info("Make sure you're running the script from the project root"))
        return False

    # Install package with extras
    cmd = [
        venv_python, "-m", "pip", "install",
        "-e", f"{project_root}[dev,notebooks]"
    ]

    return run_command(cmd, "Installing package with dev and notebook dependencies", verbose)


def install_jupyter_kernel(venv_python: str, venv_dir: Path, kernel_name: str, display_name: str) -> bool:
    """
    Install the virtual environment as a Jupyter kernel.

    Installs in two locations for maximum compatibility:
    1. User location (--user): Available system-wide
    2. Venv location (--prefix): Available when using venv's Jupyter

    Args:
        venv_python: Path to venv Python executable
        venv_dir: Path to venv directory
        kernel_name: Internal kernel name
        display_name: Display name shown in Jupyter

    Returns:
        True if successful, False otherwise
    """
    print(output.info(f"Installing Jupyter kernel: {display_name}"))

    # Install to user location (system-wide)
    cmd_user = [
        venv_python, "-m", "ipykernel", "install",
        "--user",
        f"--name={kernel_name}",
        f"--display-name={display_name}"
    ]

    # Install to venv location (for venv's Jupyter)
    cmd_venv = [
        venv_python, "-m", "ipykernel", "install",
        f"--prefix={venv_dir}",
        f"--name={kernel_name}",
        f"--display-name={display_name}"
    ]

    success = True

    try:
        # Install to user location
        result = subprocess.run(
            cmd_user,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(output.success(f"Kernel installed to user location"))
        else:
            print(output.warning(f"Failed to install to user location: {result.stderr}"))
            success = False

        # Install to venv location
        result = subprocess.run(
            cmd_venv,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(output.success(f"Kernel installed to venv location"))
        else:
            print(output.warning(f"Failed to install to venv location: {result.stderr}"))
            success = False

        if success:
            print(output.success(f"Jupyter kernel '{display_name}' installed"))
            print(output.info("Restart Jupyter if it's already running to see the new kernel"))

        return success

    except subprocess.TimeoutExpired:
        print(output.warning("Jupyter kernel installation timed out"))
        return False
    except Exception as e:
        print(output.warning(f"Error installing Jupyter kernel: {e}"))
        return False


def verify_installation(venv_python: str, venv_dir: Path) -> bool:
    """
    Verify that all packages are installed correctly.

    Args:
        venv_python: Path to venv Python executable
        venv_dir: Path to venv directory

    Returns:
        True if all checks pass, False otherwise
    """
    print(output.info("Verifying installation..."))

    # Verification script to run in subprocess
    verify_script = """
import sys
import json

results = {}

# Check Python version
results['python'] = {
    'version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    'success': True
}

# Check packages
packages = {
    'rctbp_bf_training': '__version__',
    'torch': '__version__',
    'bayesflow': '__version__',
    'keras': '__version__',
    'numpy': '__version__'
}

for package, version_attr in packages.items():
    try:
        module = __import__(package)
        version = getattr(module, version_attr, 'unknown')
        results[package] = {'version': version, 'success': True}
    except ImportError as e:
        results[package] = {'error': str(e), 'success': False}

# Check CUDA availability
try:
    import torch
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(0)
        results['cuda'] = {
            'available': True,
            'version': cuda_version,
            'device': device_name,
            'success': True
        }
    else:
        results['cuda'] = {
            'available': False,
            'message': 'CPU-only mode',
            'success': True
        }
except Exception as e:
    results['cuda'] = {'error': str(e), 'success': False}

print(json.dumps(results))
"""

    try:
        result = subprocess.run(
            [venv_python, "-c", verify_script],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(output.error("Verification failed"))
            print(result.stderr)
            return False

        # Parse results
        import json
        results = json.loads(result.stdout)

        # Display results
        all_success = True

        # Python version
        if results['python']['success']:
            print(output.success(f"Python: {results['python']['version']}"))

        # Package versions
        for package in ['rctbp_bf_training', 'torch', 'bayesflow', 'keras', 'numpy']:
            if package in results:
                if results[package]['success']:
                    version = results[package].get('version', 'unknown')
                    print(output.success(f"{package}: {version}"))
                else:
                    error = results[package].get('error', 'unknown error')
                    print(output.error(f"{package}: {error}"))
                    all_success = False

        # CUDA availability
        if 'cuda' in results:
            if results['cuda']['success']:
                if results['cuda'].get('available', False):
                    cuda_ver = results['cuda'].get('version', 'unknown')
                    device = results['cuda'].get('device', 'unknown')
                    print(output.success(f"CUDA: {cuda_ver} ({device})"))
                else:
                    message = results['cuda'].get('message', 'Not available')
                    print(output.info(f"CUDA: {message}"))
            else:
                error = results['cuda'].get('error', 'unknown error')
                print(output.warning(f"CUDA check failed: {error}"))

        # Save verification report
        report_path = venv_dir / "venv_info.txt"
        with open(report_path, 'w') as f:
            f.write("RCTBP BayesFlow Training - Virtual Environment Info\n")
            f.write("=" * 60 + "\n\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        print(output.info(f"Verification report saved to: {report_path}"))

        return all_success

    except Exception as e:
        print(output.error(f"Verification error: {e}"))
        return False


def print_activation_instructions(activate_path: str):
    """Print instructions for activating the virtual environment."""
    print(output.header("Setup Complete!"))
    print("\nTo activate the virtual environment:\n")

    if platform.system() == "Windows":
        print(f"  {activate_path}")
        print("\nOr in PowerShell:")
        print(f"  {activate_path.replace('activate.bat', 'Activate.ps1')}")
    else:
        print(f"  source {activate_path}")

    print("\nTo verify the installation:")
    print('  python -c "import rctbp_bf_training; import torch; print(torch.cuda.is_available())"')

    print("\nTo use in Jupyter:")
    print("  # Option 1: Start Jupyter from the activated venv (recommended)")
    if platform.system() == "Windows":
        print(f"  {activate_path} && jupyter notebook examples/")
    else:
        print(f"  source {activate_path} && jupyter notebook examples/")
    print("\n  # Option 2: Start Jupyter from anywhere and select the kernel:")
    print("  jupyter notebook examples/")
    print("  # Then select: 'Python (rctbp_bf_training - venv)' from Kernel menu")
    print("\n  Note: If kernel doesn't appear, restart Jupyter")

    print("\nTo run tests:")
    print("  pytest tests/")


def main():
    """Main execution flow."""
    args = parse_arguments()

    # Print header
    print(output.header("RCTBP BayesFlow Training - Environment Setup"))

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    print(output.success(f"Python version: {sys.version.split()[0]}"))

    # Determine project root (script's parent directory)
    project_root = Path(__file__).parent.absolute()
    venv_dir = project_root / args.name

    print(output.info(f"Project root: {project_root}"))
    print(output.info(f"Virtual environment: {venv_dir}"))

    # Create or verify virtual environment
    if not create_or_verify_venv(venv_dir, args.force):
        sys.exit(1)

    # Get venv paths
    venv_paths = get_venv_paths(venv_dir)

    # Detect or use specified CUDA version
    if args.cpu_only:
        cuda_version = "cpu"
        cuda_detected = False
        print(output.info("CPU-only mode (--cpu-only)"))
    elif args.cuda_version != "auto":
        cuda_version = args.cuda_version
        cuda_detected = True
        print(output.info(f"Using specified CUDA version: {cuda_version}"))
    else:
        cuda_version, cuda_detected = detect_cuda_version()

    # Map to PyTorch build
    pytorch_build = map_cuda_to_pytorch_build(cuda_version)

    # Install PyTorch
    print(output.header("Installing PyTorch"))
    if not install_pytorch(venv_paths["python"], pytorch_build, args.verbose):
        print(output.error("PyTorch installation failed"))
        sys.exit(1)

    # Install package with extras
    print(output.header("Installing Package and Dependencies"))
    if not install_package_with_extras(venv_paths["python"], project_root, args.verbose):
        print(output.error("Package installation failed"))
        sys.exit(1)

    # Verify installation
    print(output.header("Verifying Installation"))
    if not verify_installation(venv_paths["python"], venv_dir):
        print(output.warning("Installation may be incomplete"))
        print(output.info("You can try running the script again with --force"))

    # Install Jupyter kernel
    print(output.header("Installing Jupyter Kernel"))
    kernel_name = f"rctbp-{args.name}"
    kernel_display_name = f"Python (rctbp_bf_training - {args.name})"
    install_jupyter_kernel(venv_paths["python"], venv_dir, kernel_name, kernel_display_name)

    # Print activation instructions
    print_activation_instructions(venv_paths["activate"])


if __name__ == "__main__":
    main()
