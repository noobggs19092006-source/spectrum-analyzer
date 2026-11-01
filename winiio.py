import os
import ctypes.util

PROJECT_DIR = r"C:\spectrum_rf"
LIBIIO_DIR = r"C:\libiio\Release"

# Let Python load DLLs from the project folder and the libiio folder
os.add_dll_directory(LIBIIO_DIR)
os.add_dll_directory(PROJECT_DIR)
os.environ["PATH"] = PROJECT_DIR + os.pathsep + LIBIIO_DIR + os.pathsep + os.environ["PATH"]

# Names Windows can't resolve on its own
_orig_find_library = ctypes.util.find_library
_fallback = {"c": "msvcrt.dll", "iio": "libiio.dll", "rtlsdr": "librtlsdr.dll"}

def _patched_find_library(name):
    return _orig_find_library(name) or _fallback.get(name)

ctypes.util.find_library = _patched_find_library