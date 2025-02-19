from pathlib import Path
import cv2


def main():
    data_dir = Path('/project_fused/data')
    for image_file in data_dir.joinpath('Analysis_Dataset/unlabeled_images').glob('*'):
        if 'lidar' in str(image_file):
            category = image_file.stem.rsplit('_',3)[0]
            number = image_file.stem.rsplit('_',1)[-1]
            lidar_tiff_file = data_dir.joinpath(category + '/lidar/' + 'lidar_image_' + number + '.tiff')
            lidar_tiff_image = cv2.imread(lidar_tiff_file, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(data_dir.joinpath('Analysis_Dataset/unlabeled_images_new/' + category + '_lidar_image_' + number + '.tiff'), lidar_tiff_image)
        else:
            continue
    

if __name__ == '__main__':
    main()