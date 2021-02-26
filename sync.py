import argparse
import subprocess
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dst")
    args = parser.parse_args()

    src_path = os.path.dirname(__file__)
    dst_path = f"{args.dst}:dev/"
    cmd = ["rsync", "-av", src_path, dst_path, "--exclude=env", "--exclude=__pycache__", "--exclude=.DS_Store", "--exclude=.git", "--exclude=.vscode"]
    ret = subprocess.run(cmd)
    if ret.returncode == 0:
        print("Successfully sync code")
    else:
        print("Failed to sync code")
