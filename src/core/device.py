import subprocess
import platform

class Device:
    def __init__(self, device_type):
        self.device_type = device_type

    def is_cuda(self):
        return self.device_type == "cuda"

    def is_metal(self):
        return self.device_type == "metal"

    def is_cpu(self):
        return self.device_type == "cpu"

def get_default_device():
    # Check for NVIDIA GPU
    if has_nvidia_gpu():
        return Device("cuda")
    # Check for Mac and Metal support
    elif is_mac() and supports_metal():
        return Device("metal")
    # Fallback to CPU
    else:
        return Device("cpu")

def has_nvidia_gpu():
    try:
        # Run the nvidia-smi command and check its output
        output = subprocess.check_output(["nvidia-smi"])
        return "NVIDIA" in output.decode("utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def is_mac():
    return platform.system() == "Darwin"

def supports_metal():
    # Check if macOS version is 10.14 (Mojave) or later
    mac_version = platform.mac_ver()[0]
    return [int(v) for v in mac_version.split(".")] >= [10, 14]
