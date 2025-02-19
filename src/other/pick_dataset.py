from pathlib import Path
import os
import cv2
import re
import numpy as np


def main():
    data_dir = Path('/project_fused/data')
    output_dir = Path('/project_fused/output/dataset')
    unwanted_roots = ['/project_fused/data/Testing_Dark_Gunner_Walking_Bad',
                      '/project_fused/data/Testing_Dark_Gunner_Walking_Bad/lidar',
                      '/project_fused/data/Testing_Dark_Gunner_Walking_Bad/lidar_color',
                      '/project_fused/data/Testing_Dark_Gunner_Walking_Bad/thermal',
                      '/project_fused/data/Testing_Dark_Gunner_Walking_Bad/thermal_color',  
                      '/project_fused/data/Testing_Dark_Gunner_Walking_Bad/webcam',  
                      '/project_fused/data/Testing_Mannequin_Sitting_Further',
                      '/project_fused/data/Testing_Mannequin_Sitting_Further/lidar',
                      '/project_fused/data/Testing_Mannequin_Sitting_Further/lidar_color',
                      '/project_fused/data/Testing_Mannequin_Sitting_Further/thermal',
                      '/project_fused/data/Testing_Mannequin_Sitting_Further/thermal_color',
                      '/project_fused/data/Testing_Mannequin_Sitting_Further/webcam']
    unwanted_dirs = ['Testing_Dark_Gunner_Walking_Bad', 'Testing_Mannequin_Sitting_Further']
    for root, dirs, files in os.walk(data_dir):
        if root in unwanted_roots:
            continue
        skip = False
        for directory in dirs:
            if directory in unwanted_dirs:
                skip = True
                break
        if skip:
            continue
        if root.rpartition("/")[-1] == 'lidar_color' or root.rpartition("/")[-1] == 'thermal_color':
            continue
        if files:
            files = sorted(Path(root).glob("*"), key=lambda p: int(re.search(r'_(\d+)', p.stem).group(1)))
            num_files = len(files)
            start = round(0.3*num_files)
            end = round(num_files - 0.3*num_files)
            indexes = np.linspace(start, end, 10, dtype=int)
            for index in indexes:
                image_path = files[int(index)]
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                cv2.imwrite(output_dir.joinpath(root.rsplit("/",2)[-2:][0] + '_' + image_path.stem + image_path.suffix), image)
                

if __name__ == '__main__':
    main()