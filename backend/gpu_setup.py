"""Make the CUDA runtime DLLs shipped by the nvidia-cublas-cu12 / nvidia-cudnn-cu12
pip packages discoverable. Unlike the Linux wheels, the Windows wheels don't get
added to the DLL search path automatically -- ctranslate2 fails to load them at
GPU-inference time otherwise, even though the packages are installed. Must run
before any faster_whisper/ctranslate2 import triggers a CUDA DLL load.
"""

import os
import sys


def register_nvidia_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            module = __import__(pkg, fromlist=["__path__"])
        except ImportError:
            continue
        # nvidia.* are namespace packages (no __file__), so use __path__ instead.
        for package_dir in module.__path__:
            bin_dir = os.path.join(package_dir, "bin")
            if os.path.isdir(bin_dir):
                # os.add_dll_directory alone doesn't reliably cover DLLs that
                # ctranslate2's native extension loads transitively -- PATH does.
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
