from pathlib import Path
import cv2
import numpy as np


def main():
    data_dir = Path('/project_fused/output/dataset')
    lidar_files = data_dir.glob('*.tiff')
    output_dir = Path('/project_fused/output/new_lidar')
    for lidar_file in lidar_files:
        lidar_image = cv2.imread(lidar_file, cv2.IMREAD_UNCHANGED)
        max_depth = np.max(lidar_image)
        lidar_image_clipped = np.clip(lidar_image, 0, max_depth)
        lidar_image_mm = lidar_image_clipped * 1000
        lidar_image_normalized = cv2.normalize(lidar_image_mm, None, 0, 65535, cv2.NORM_MINMAX)
        lidar_image_8bit = cv2.convertScaleAbs(lidar_image_normalized, alpha=(255.0 / np.max(lidar_image_normalized)))
        lidar_image_equalized = cv2.equalizeHist(lidar_image_8bit)
        cv2.imwrite(output_dir.joinpath(lidar_file.stem + '.png'), lidar_image_equalized)
        
        
if __name__ == '__main__':
    main()