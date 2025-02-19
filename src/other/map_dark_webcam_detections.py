from pathlib import Path
import cv2
from numpy import array
import json
from numpy.linalg import inv

T_l2t = array([[1, 0, 0, 0.028],
                    [0, 1, 0, -0.038],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
T_l2w = array([[1, 0, 0, 0.083],
                    [0, 1, 0, -0.035],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

# Set extrinsic rotation matrices from stereo calibration
R_t2cₜ = array([[0.804905, 0.593319, 0.010014],
                        [-0.588094, 0.795337, 0.146920],
                        [0.079206, -0.124146, 0.989098]])
R_l2cₜ = array([[0.813639, 0.571181, 0.108367],
                        [-0.580035, 0.784919, 0.217856],
                        [0.039376, -0.240112, 0.969946]])
R_w2cᵣ = array([[0.903012, -0.397065, -0.164039],
                        [0.397183, 0.917127, -0.033513],
                        [0.163751, -0.034891, 0.985884]])
R_l2cᵣ = array([[0.909488, -0.399788, -0.114025],
                        [0.399705, 0.916314, -0.024592],
                        [0.114314, -0.023211, 0.993173]])

# Set intrinsic matrices for the three sensors
Kₗ = array([[205.046875, 0.0, 107.55435943603516],
                    [0.0, 205.046875, 82.43924713134766],
                    [0.0, 0.0, 1.0]])
Kₜ = array([[161.393925, 0.000000, 78.062273],
                    [0.000000, 161.761028, 59.925115], 
                    [0.000000, 0.000000, 1.000000]])
Kᵣ = array([[446.423112, 0.000000, 163.485603], 
                    [0.000000, 446.765896, 131.217485],
                    [0.000000, 0.000000, 1.000000]])

def main():
    data_dir = Path('/project_fused/data/Analysis_Dataset')
    
    with open(data_dir.joinpath('labels.json'), 'r') as json_file:
        labels_data = json.load(json_file)
        
    num_images = len(labels_data['images'])
    num_annotations = len(labels_data['annotations'])
    
    for image_file in data_dir.joinpath('labeled_images').glob('*'):
        if 'Testing_Dark_Gunner_Walking_Good_webcam' in str(image_file):
            number = image_file.stem.rsplit('_',1)[-1]
            lidar_file = data_dir.joinpath('labeled_images/Testing_Dark_Gunner_Walking_Good_lidar_image_' + number + '.tiff')
            lidar_image = cv2.imread(lidar_file, cv2.IMREAD_UNCHANGED)
            
            for image_info in labels_data['images']:
                if image_info['file_name'] == 'Testing_Dark_Gunner_Walking_Good_lidar_image_' + number + '.tiff':
                    image_id = image_info['id']
                else:
                    continue
            
            for annotations_info in labels_data['annotations']:
                if annotations_info['id'] == int(image_id):
                    bbox = annotations_info['bbox']
                else:
                    continue
            
            x1, y1 = bbox[0], bbox[1] # Top left
            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3] # Bottom right

            uₗ = round((x1 + x2) / 2)
            vₗ = round((y1 + y2) / 2)
            try:
                zₗ = lidar_image[vₗ,uₗ]
            except IndexError:
                if uₗ >= lidar_image.shape[1]:
                    uₗ = lidar_image.shape[1] - 1
                if vₗ >= lidar_image.shape[0]:
                    vₗ = lidar_image.shape[0] - 1
                zₗ = lidar_image[vₗ, uₗ]

            if zₗ > 1E-3:
                x1ₗₜ, y1ₗₜ, x1ₗᵣ, y1ₗᵣ = transform(zₗ, x1, y1)
                x2ₗₜ, y2ₗₜ, x2ₗᵣ, y2ₗᵣ = transform(zₗ, x2, y2)
                
            width, height = x2ₗᵣ - x1ₗᵣ, y2ₗᵣ - y1ₗᵣ
            
            if x1ₗᵣ < 0:
                x1ₗᵣ = 0
            if y1ₗᵣ < 0:
                y1ₗᵣ = 0
                
            width, height = x2ₗᵣ - x1ₗᵣ, y2ₗᵣ - y1ₗᵣ
            
            if x1ₗᵣ + width > 320:
                width = 320 - x1ₗᵣ 
            if y1ₗᵣ + height > 240:
                height = 240 - y1ₗᵣ 
            
            image_data = {
                'width': 320,
                'height': 240,
                'id': num_images,
                'file_name': 'Testing_Dark_Gunner_Walking_Good_webcam_image_' + number + '.png' 
            }
            
            labels_data['images'].append(image_data)
            
            annotations_data = {
                'id': num_annotations,
                'image_id': num_images,
                'category_id': 1,
                'segmentation': [],
                'bbox': [x1ₗᵣ, 
                         y1ₗᵣ,
                         width,
                         height],
                'ignore': 0,
                'iscrowd': 0,
                'area': width * height
            }
            
            labels_data['annotations'].append(annotations_data)
            
            num_images += 1
            num_annotations += 1
            
        else:
            continue
        
    with open(data_dir.joinpath('new_labels.json'), 'w') as out_file:
        json.dump(labels_data, out_file, indent=2)
            

def transform(zₗ, uₗ, vₗ):
    """Perform transformations to map a pixel from the LiDAR's camera frame onto the thermal and webcam camera frames

    Args:
        zₗ (float): Depth of the pixel, in meters
        uₗ (int): LiDAR pixel coordinate on the x axis
        vₗ (int): LiDAR pixel coordinate on the y axis

    Returns:
        uₜ, vₜ, uᵣ, vᵣ (tuple): Thermal and webcam pixel coordinates, respectively
    """
    # Calculate the 3D physical coordinate of the center of the LiDAR image
    pₗ = array([uₗ, vₗ, 1])
    l̂ₗ = inv(Kₗ) @ pₗ
    r̄ₗ = zₗ * l̂ₗ
    
    # Perform extrinsic translations to the thermal sensor and webcam
    r̄ₜ = (inv(R_t2cₜ) @ (R_l2cₜ @ r̄ₗ)) + array([T_l2t[0, 3], T_l2t[1, 3], 0]).T
    r̄ᵣ = (inv(R_w2cᵣ) @ (R_l2cᵣ @ r̄ₗ)) + array([T_l2w[0, 3], T_l2w[1, 3], 0]).T
    
    # Transform 3D coordinate to thermal and webcam pixel coordinates
    r̃ₜ = array([r̄ₜ[0]/r̄ₜ[2], r̄ₜ[1]/r̄ₜ[2], r̄ₜ[2]/r̄ₜ[2]])
    r̃ᵣ = array([r̄ᵣ[0]/r̄ᵣ[2], r̄ᵣ[1]/r̄ᵣ[2], r̄ᵣ[2]/r̄ᵣ[2]])
    pₜ = Kₜ @ r̃ₜ
    pᵣ = Kᵣ @ r̃ᵣ
    uₜ, vₜ = pₜ[0], pₜ[1]
    uᵣ, vᵣ = pᵣ[0], pᵣ[1]
    
    return uₜ, vₜ, uᵣ, vᵣ


if __name__ == '__main__':
    main()