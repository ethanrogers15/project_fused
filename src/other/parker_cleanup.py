from pathlib import Path
import os

def main():
    data_path = Path('/project_fused/data')
    for root, dirs, files in os.walk(data_path):
        if root.split("/")[-1] == 'lidar':
            for bad_file in Path(root).glob('*.png'):
                bad_file.unlink()


if __name__ == '__main__':
    main()