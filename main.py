
import subprocess
import sys

def main():
    subprocess.run([sys.executable, "load.py"], check=True)
    subprocess.run([sys.executable, "train.py"], check=True)
    print("\ncompleted")

if __name__ == "__main__":
    main()
