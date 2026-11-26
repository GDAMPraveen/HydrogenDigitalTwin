
from pathlib import Path
import shutil

# Project directory
PROJECT = Path(r"C:\Users\PRAVEEN\HydrogenDigitalTwin")

# Destination folders
WIND_DIR = PROJECT / "data" / "nlr" / "raw" / "wind_experiments"
CHAR_DIR = PROJECT / "data" / "nlr" / "raw" / "characterization"

WIND_DIR.mkdir(parents=True, exist_ok=True)
CHAR_DIR.mkdir(parents=True, exist_ok=True)

# 11 valid wind experiment archives
wind_archives = [
    "windIEA3.4-5ms-A_2-200.zip",
    "windIEA3.4-5ms-A_2-400.zip",
    "windIEA3.4-5ms-C_2-200.zip",
    "windIEA3.4-5ms-C_2-400.zip",
    "windIEA3.4-7ms-A_1-200.zip",
    "windIEA3.4-7ms-A_1-400.zip",
    "windIEA3.4-7ms-C_1-200.zip",
    "windIEA3.4-7ms-C_1-400.zip",
    "windIEA3.4-9ms-A_1-400.zip",
    "windIEA3.4-9ms-C_1-200.zip",
    "windIEA3.4-9ms-C_1-400.zip",
]

# Characterization archive
characterization_archive = "characterization_200.zip"

# Locate existing downloaded files
# Update this if your downloads are stored elsewhere.
DOWNLOADS = Path.home() / "Downloads"

def move_archive(filename, destination):
    source = DOWNLOADS / filename

    if not source.exists():
        print(f"MISSING: {filename}")
        return

    target = destination / filename

    if target.exists():
        print(f"ALREADY EXISTS: {filename}")
        return

    shutil.copy2(source, target)
    print(f"COPIED: {filename}")

print("\n--- WIND EXPERIMENTS ---")
for filename in wind_archives:
    move_archive(filename, WIND_DIR)

print("\n--- CHARACTERIZATION ---")
move_archive(characterization_archive, CHAR_DIR)

print("\nArchive organization completed.")
