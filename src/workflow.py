# Senior Seminar Workflow Code
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy import array
import cv2
import numpy as np
import mediapipe as mp
from numpy.linalg import inv
from mediapipe.tasks.python.components.containers import Detection, DetectionResult, BoundingBox, Category
import re


class Fused_Workflow:
    def __init__(self, iou_threshold, decision_making_mode, output_dir, max_results, score_threshold):
        """Project FUSED Workflow Algorithm

        Args:
            iou_threshold (float): Intersection-over-Union fraction threshold for evaluating bounding box overlaps
            decision_making_mode (str): Which sensors need to agree for the detection to be considered valid? 'all', 'thermal', or 'webcam'
            output_dir (Path object): Output directory for fused images to be stored
            score_threshold (float): Minimum confidence score that the models must have for the detection to be kept
        """
        self.iou_threshold = iou_threshold
        self.decision_making_mode = decision_making_mode
        self.output_dir = output_dir
        
        # Step 3: Initialize the object detection models
        base_options_webcam = python.BaseOptions(model_asset_path='/project_fused/models/efficientdet_lite0.tflite')
        options_webcam = vision.ObjectDetectorOptions(base_options=base_options_webcam, running_mode=vision.RunningMode.IMAGE, max_results=max_results, score_threshold=score_threshold)
        self.webcam_detector = vision.ObjectDetector.create_from_options(options_webcam)

        # Initialize the thermal object detection model
        base_options_thermal = python.BaseOptions(model_asset_path='/project_fused/models/thermal.tflite')
        options_thermal = vision.ObjectDetectorOptions(base_options=base_options_thermal, running_mode=vision.RunningMode.IMAGE, max_results=max_results, score_threshold=score_threshold)
        self.thermal_detector = vision.ObjectDetector.create_from_options(options_thermal)

        # Initialize the LiDAR object detection model
        base_options_lidar = python.BaseOptions(model_asset_path='/project_fused/models/lidar.tflite')
        options_lidar = vision.ObjectDetectorOptions(base_options=base_options_lidar, running_mode=vision.RunningMode.IMAGE, max_results=max_results, score_threshold=score_threshold)
        self.lidar_detector = vision.ObjectDetector.create_from_options(options_lidar)

        # Step 4: Define the transformation matrices
        # Set extrinsic translation matrices based on physical measurements, no z translation assumed
        self.T_l2t = array([[1, 0, 0, 0.028],
                            [0, 1, 0, -0.038],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.T_l2w = array([[1, 0, 0, 0.083],
                            [0, 1, 0, -0.035],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Set extrinsic rotation matrices from stereo calibration
        self.R_t2cₜ = array([[0.804905, 0.593319, 0.010014],
                             [-0.588094, 0.795337, 0.146920],
                             [0.079206, -0.124146, 0.989098]])
        self.R_l2cₜ = array([[0.813639, 0.571181, 0.108367],
                             [-0.580035, 0.784919, 0.217856],
                             [0.039376, -0.240112, 0.969946]])
        self.R_w2cᵣ = array([[0.903012, -0.397065, -0.164039],
                             [0.397183, 0.917127, -0.033513],
                             [0.163751, -0.034891, 0.985884]])
        self.R_l2cᵣ = array([[0.909488, -0.399788, -0.114025],
                             [0.399705, 0.916314, -0.024592],
                             [0.114314, -0.023211, 0.993173]])

        # Set intrinsic matrices for the three sensors
        self.Kₗ = array([[205.046875, 0.0, 107.55435943603516],
                         [0.0, 205.046875, 82.43924713134766],
                         [0.0, 0.0, 1.0]])
        self.Kₜ = array([[161.393925, 0.000000, 78.062273],
                         [0.000000, 161.761028, 59.925115], 
                         [0.000000, 0.000000, 1.000000]])
        self.Kᵣ = array([[446.423112, 0.000000, 163.485603], 
                         [0.000000, 446.765896, 131.217485],
                         [0.000000, 0.000000, 1.000000]])

        # Initialize visualization parameters
        self.TEXT_COLOR = (255, 255, 255)
        self.BOX_THICKNESS = 3
        self.MARGIN = 5
        self.ROW_SIZE = -15
        self.FONT_SIZE = 0.5
        self.FONT_THICKNESS = 1
        
    def fuse(self, lidar_image, thermal_image, webcam_image):
        """Main FUSED workflow function: perform fusion based alignment of object detection
        bounding boxes on synchronized LiDAR, thermal, and webcam images for decision making

        Args:
            lidar_image (OpenCV image): Synchronized LiDAR image
            thermal_image (OpenCV image): Synchronized thermal image
            webcam_image (OpenCV image): Synchronized webcam image

        Returns:
            fused_images (tuple): LiDAR, thermal, and webcam fused images with bounding boxes
            drawn after decision making
        """
        # Perform LiDAR image processing
        max_depth = np.max(lidar_image)
        lidar_image_clipped = np.clip(lidar_image, 0, max_depth)
        lidar_image_mm = lidar_image_clipped * 1000
        lidar_image_normalized = cv2.normalize(lidar_image_mm, None, 0, 65535, cv2.NORM_MINMAX)
        lidar_image_8bit = cv2.convertScaleAbs(lidar_image_normalized, alpha=(255.0 / np.max(lidar_image_normalized)))
        lidar_image_equalized = cv2.equalizeHist(lidar_image_8bit)

        # Step 5c: Convert OpenCV images to RGB format
        lidar_image_rgb = cv2.cvtColor(lidar_image_equalized, cv2.COLOR_GRAY2RGB)
        thermal_image_rgb = cv2.cvtColor(thermal_image, cv2.COLOR_GRAY2RGB)
        webcam_image_rgb = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)

        # Step 5d: Convert RGB images to MediaPipe images
        lidar_image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=lidar_image_rgb)
        thermal_image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=thermal_image_rgb)
        webcam_image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=webcam_image_rgb)

        # Step 5e: Perform object detection on the MediaPipe images
        lidar_detection_result = self.lidar_detector.detect(lidar_image_mp)
        thermal_detection_result = self.thermal_detector.detect(thermal_image_mp)
        webcam_detection_result = self.webcam_detector.detect(webcam_image_mp)

        # Initialize lists for keeping track of detections to be kept out of the next iteration
        thermal_exclude_idx = []
        webcam_exclude_idx = []
        
        # Initialize detection lists for the fused results
        lidar_fused_detections = []
        thermal_fused_detections = []
        webcam_fused_detections = []

        # Step 5f: For loop through each LiDAR detection in the detection result
        if lidar_detection_result.detections:
            for detection in lidar_detection_result.detections:
                if detection.categories[0].category_name != 'Person':
                    continue
                # Step 5f1: Define the top left and bottom right points of the detection
                bbox = detection.bounding_box
                x1, y1 = bbox.origin_x, bbox.origin_y # Top left
                x2, y2 = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height # Bottom right

                # Step 5f2: Find the depth on the LiDAR image at the center of the box
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

                # Step 5f3 & 5f4: If depth is not zero, then compute transformed u and v on webcam and thermal frames
                if zₗ > 1E-3:
                    x1ₗₜ, y1ₗₜ, x1ₗᵣ, y1ₗᵣ = self.transform(zₗ, x1, y1)
                    x2ₗₜ, y2ₗₜ, x2ₗᵣ, y2ₗᵣ = self.transform(zₗ, x2, y2)

                    # Step 5f5: Calculate IoU between the mapped bounding box and all detection results from the webcam and thermal images
                    thermal_mapped_box = (x1ₗₜ, y1ₗₜ, x2ₗₜ, y2ₗₜ)
                    if thermal_detection_result.detections and len(thermal_detection_result.detections) != len(thermal_exclude_idx):
                        thermal_ious = []
                        for idxₜ, thermal_detection in enumerate(thermal_detection_result.detections):
                            if thermal_detection.categories[0].category_name != 'Person':
                                thermal_ious.append(0.0)
                                continue
                            if idxₜ in thermal_exclude_idx:
                                thermal_ious.append(0.0)
                                continue
                            thermal_bbox = thermal_detection.bounding_box
                            x1ₜ, y1ₜ = thermal_bbox.origin_x, thermal_bbox.origin_y
                            x2ₜ, y2ₜ = thermal_bbox.origin_x + thermal_bbox.width, thermal_bbox.origin_y + thermal_bbox.height
                            thermal_box = (x1ₜ, y1ₜ, x2ₜ, y2ₜ)
                            thermal_ious.append(self.calc_iou(thermal_box, thermal_mapped_box))

                    webcam_mapped_box = (x1ₗᵣ, y1ₗᵣ, x2ₗᵣ, y2ₗᵣ)
                    if webcam_detection_result.detections and len(webcam_detection_result.detections) != len(webcam_exclude_idx):
                        webcam_ious = []
                        for idxᵣ, webcam_detection in enumerate(webcam_detection_result.detections):
                            if webcam_detection.categories[0].category_name != 'person':
                                webcam_ious.append(0.0)
                                continue
                            if idxᵣ in webcam_exclude_idx:
                                webcam_ious.append(0.0)
                                continue
                            webcam_bbox = webcam_detection.bounding_box
                            x1ᵣ, y1ᵣ = webcam_bbox.origin_x, webcam_bbox.origin_y
                            x2ᵣ, y2ᵣ = webcam_bbox.origin_x + webcam_bbox.width, webcam_bbox.origin_y + webcam_bbox.height
                            webcam_box = (x1ᵣ, y1ᵣ, x2ᵣ, y2ᵣ)
                            webcam_ious.append(self.calc_iou(webcam_box, webcam_mapped_box))

                    # Step 5f6: Choose the thermal or webcam detection result corresponding to the LiDAR mapped result whose IoU is the 
                    #           largest and also above the defined Combination IoU threshold. In the next iterations of the for loop,
                    #           the thermal or webcam detection result that was chosen should not be chosen again to match with another
                    #           LiDAR mapped result
                    valid_thermal_iou = None
                    valid_webcam_iou = None
                    if thermal_detection_result.detections and len(thermal_detection_result.detections) != len(thermal_exclude_idx):
                        max_thermal_iou = max(thermal_ious)
                        max_thermal_iou_index = thermal_ious.index(max_thermal_iou)
                        valid_thermal_iou = 0
                        if max_thermal_iou > self.iou_threshold:
                            valid_thermal_iou, valid_thermal_idx = max_thermal_iou, max_thermal_iou_index
                            thermal_exclude_idx.append(valid_thermal_idx)
                    
                    if webcam_detection_result.detections and len(webcam_detection_result.detections) != len(webcam_exclude_idx):
                        max_webcam_iou = max(webcam_ious)
                        max_webcam_iou_index = webcam_ious.index(max_webcam_iou)
                        valid_webcam_iou = 0
                        if max_webcam_iou > self.iou_threshold:
                            valid_webcam_iou, valid_webcam_idx = max_webcam_iou, max_webcam_iou_index
                            webcam_exclude_idx.append(valid_webcam_idx)

                    # Step 5f7: Depending on the decision making mode, choose to either keep the mapped result or not based on whether there 
                    #           is agreement between all 3 or only two sensors
                    # Step 5f8: If the mapped result is not being kept, then go to the next iteration of the loop. If it is being kept, then
                    #           keep the original detections that have been agreed upon according to the decision making mode. For the 
                    #           detection that has not been agreed upon, check if it agrees with LiDAR. If it does, keep it. If it does not,
                    #           then use the mapped LiDAR detection onto the appropriate camera frame instead
                    # Step 5f9: Store the three fused detection results at each iteration
                    if self.decision_making_mode == 'all':
                        if valid_thermal_iou and valid_webcam_iou:
                            lidar_fused_detections.append(detection)
                            thermal_fused_detections.append(thermal_detection_result.detections[valid_thermal_idx])
                            webcam_fused_detections.append(webcam_detection_result.detections[valid_webcam_idx])
                        else:
                            continue

                    if self.decision_making_mode == 'thermal':
                        if valid_thermal_iou:
                            lidar_fused_detections.append(detection)
                            thermal_fused_detections.append(thermal_detection_result.detections[valid_thermal_idx])
                            if valid_webcam_iou:
                                webcam_fused_detections.append(webcam_detection_result.detections[valid_webcam_idx])
                            else:
                                webcam_fused_detections.append(self.create_detection(detection, webcam_mapped_box))
                        else:
                            continue

                    if self.decision_making_mode == 'webcam':
                        if valid_webcam_iou:
                            lidar_fused_detections.append(detection)
                            webcam_fused_detections.append(webcam_detection_result.detections[valid_webcam_idx])
                            if valid_thermal_iou:
                                thermal_fused_detections.append(thermal_detection_result.detections[valid_thermal_idx])
                            else:
                                thermal_fused_detections.append(self.create_detection(detection, thermal_mapped_box))
                        else:
                            continue
                else:
                    continue

        # Step 5g: With all of the fused detections, create detection results
        # Step 5h: Draw the bounding boxes on the images corresponding to the results
        if not lidar_fused_detections:
            lidar_fused_image = lidar_image_equalized
        else:
            lidar_fused_detection_result = DetectionResult(detections=lidar_fused_detections)
            lidar_fused_image = self.visualize(lidar_image_equalized, lidar_fused_detection_result)
            
        if not thermal_fused_detections:
            thermal_fused_image = thermal_image
        else:
            thermal_fused_detection_result = DetectionResult(detections=thermal_fused_detections)
            thermal_fused_image = self.visualize(thermal_image, thermal_fused_detection_result)
            
        if not webcam_fused_detections:
            webcam_fused_image = webcam_image
        else:
            webcam_fused_detection_result = DetectionResult(detections=webcam_fused_detections)
            webcam_fused_image = self.visualize(webcam_image, webcam_fused_detection_result)
            
        return lidar_fused_image, thermal_fused_image, webcam_fused_image
            
    def transform(self, zₗ, uₗ, vₗ):
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
        l̂ₗ = inv(self.Kₗ) @ pₗ
        r̄ₗ = zₗ * l̂ₗ
        
        # Perform extrinsic translations to the thermal sensor and webcam
        r̄ₜ = (inv(self.R_t2cₜ) @ (self.R_l2cₜ @ r̄ₗ)) + array([self.T_l2t[0, 3], self.T_l2t[1, 3], 0]).T
        r̄ᵣ = (inv(self.R_w2cᵣ) @ (self.R_l2cᵣ @ r̄ₗ)) + array([self.T_l2w[0, 3], self.T_l2w[1, 3], 0]).T
        
        # Transform 3D coordinate to thermal and webcam pixel coordinates
        r̃ₜ = array([r̄ₜ[0]/r̄ₜ[2], r̄ₜ[1]/r̄ₜ[2], r̄ₜ[2]/r̄ₜ[2]])
        r̃ᵣ = array([r̄ᵣ[0]/r̄ᵣ[2], r̄ᵣ[1]/r̄ᵣ[2], r̄ᵣ[2]/r̄ᵣ[2]])
        pₜ = self.Kₜ @ r̃ₜ
        pᵣ = self.Kᵣ @ r̃ᵣ
        uₜ, vₜ = pₜ[0], pₜ[1]
        uᵣ, vᵣ = pᵣ[0], pᵣ[1]
        
        return uₜ, vₜ, uᵣ, vᵣ
    
    def calc_iou(self, box_1, box_2):
        """Calculate the Intersection-over-Union between two bounding boxes

        Args:
            box_1 (tuple): Tuple of top-left and bottom-right pixel coordinates for the first bounding box
            box_2 (tuple): Tuple of top-left and bottom-right pixel coordinates for the second bounding box

        Returns:
            iou (float): Intersection-over-Union ratio
        """
        # Get corner values from both boxes
        x1, y1, x2, y2 = box_1
        x3, y3, x4, y4 = box_2
        
        # Get corner values for the intersection box
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)
        
        # Calculate the area of the intersection box
        width_inter = max(0, x_inter2 - x_inter1)
        height_inter = max(0, y_inter2 - y_inter1)
        area_inter = width_inter * height_inter
        
        # Calculate the areas of the two boxes
        width_box1 = x2 - x1
        height_box1 = y2 - y1
        width_box2 = x4 - x3
        height_box2 = y4 - y3
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2
        
        # Calculate the area of the full union of the two boxes
        area_union = area_box1 + area_box2 - area_inter
        
        # If union area is zero, return 0
        if area_union == 0:
            return 0.0
        
        # Calculate the IoU
        iou = area_inter / area_union

        return iou
    
    def create_detection(self, lidar_detection, other_detection_box):
        """Create a MediaPipe detection object

        Args:
            lidar_detection (detection object): Original LiDAR MediaPipe detection object
            other_detection_box (tuple): Tuple with bounding box coordinates for mapped LiDAR box onto either
            webcam or thermal camera frames

        Returns:
            detection (detection object): MediaPipe detection object for the mapped LiDAR box onto one of the 
            two other camera frames
        """
        # Get bounding box coordinates and score
        x1, y1, x2, y2 = other_detection_box
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        score = lidar_detection.categories[0].score
        
        # Define data dictionary
        data = {
            "bounding_box": (x1, y1, x2 - x1, y2 - y1),
            "score": score,
            "category_name": "Person"
        }
        
        # Use MediaPipe functions to build the detection object
        bounding_box = BoundingBox(
            origin_x=data["bounding_box"][0],
            origin_y=data["bounding_box"][1],
            width=data["bounding_box"][2],
            height=data["bounding_box"][3]
        )
        
        category = Category(
            index=None, # Optional
            score=data["score"],
            display_name=None, # Optional
            category_name=data["category_name"]
        )
        
        detection = Detection(
            bounding_box=bounding_box,
            categories=[category],
            keypoints=[] # Optional
        )

        return detection

    def visualize(self, image, detection_result):
        """Draw bounding boxes on OpenCV images

        Args:
            image (OpenCV image): OpenCV image that the box must be drawn on
            detection_result (MediaPipe detection result): MediaPipe detection result containing bounding box coordinates and labels

        Returns:
            image (OpenCV image): OpenCV image with the boxes and labels drawn
        """
        # Start for loop for all detections 
        for detection in detection_result.detections:
            # Draw the bounding box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, self.TEXT_COLOR, self.BOX_THICKNESS)

            # Write the label
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (self.MARGIN + bbox.origin_x,
                                self.MARGIN + self.ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                        self.FONT_SIZE, self.TEXT_COLOR, self.FONT_THICKNESS, cv2.LINE_AA)
            
        return image


def main():
    # Step 1: Get the number of images in the chosen test directory
    test_dir = Path('/project_fused/data/Testing_Dark_Gunner_Walking_Good')
    num_images = len(list(test_dir.joinpath('thermal').glob('*')))

    # Step 2: Define the output directory
    output_dir = Path('/project_fused/output').joinpath(test_dir.stem)
    lidar_output_dir = output_dir.joinpath('lidar')
    thermal_output_dir = output_dir.joinpath('thermal')
    webcam_output_dir = output_dir.joinpath('webcam')
    lidar_output_dir.mkdir(parents=True, exist_ok=True)
    thermal_output_dir.mkdir(parents=True, exist_ok=True)
    webcam_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Workflow Inputs:
    iou_threshold = 0.4
    decision_making_mode = 'thermal' # options are 'all', 'thermal', and 'webcam'
    max_results = 3
    score_threshold = 0.5 # Will set higher to mitigate random sensor agreement due to sheer
                          # number of garbage baxes being returned otherwise
    
    # Initialize FUSED class - Steps 3-4
    fused = Fused_Workflow(iou_threshold, decision_making_mode, output_dir, max_results, score_threshold)

    # Step 5: For loop through each set of synchronized images
    for i in range(num_images):
        # Step 5a: Get three synchronized images
        lidar_path = test_dir.joinpath(f'lidar/lidar_image_{i+1}.tiff')
        thermal_path = test_dir.joinpath(f'thermal/thermal_image_{i+1}.png')
        webcam_path = test_dir.joinpath(f'webcam/webcam_image_{i+1}.png')

        # Step 5b: Read in the images as OpenCV images
        lidar_image = cv2.imread(lidar_path, cv2.IMREAD_UNCHANGED)
        thermal_image = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
        webcam_image = cv2.imread(webcam_path, cv2.IMREAD_UNCHANGED)
        
        # Call the fuse method - Steps 5c-5h
        lidar_fused_image, thermal_fused_image, webcam_fused_image = fused.fuse(lidar_image, thermal_image, webcam_image)
        
        # Step 5i: View & output the 3 new images (LiDAR, thermal, webcam) with bounding boxes 
        #          drawn on them to the output directory for viewing
        cv2.imwrite(lidar_output_dir.joinpath(f'lidar_fused_image_{i+1}.png'), lidar_fused_image)
        cv2.imwrite(thermal_output_dir.joinpath(f'thermal_fused_image_{i+1}.png'), thermal_fused_image)
        cv2.imwrite(webcam_output_dir.joinpath(f'webcam_fused_image_{i+1}.png'), webcam_fused_image)
    
    # Generate videos    
    frame_rate = 5
    name_list = ['lidar_output_video.mp4', 'thermal_output_video.mp4', 'webcam_output_video.mp4']
    output_dir_list = [lidar_output_dir, thermal_output_dir, webcam_output_dir]
    
    lidar_images = sorted(lidar_output_dir.glob("lidar_fused_image_*.png"), key=lambda p: int(re.search(r'_(\d+)', p.stem).group(1)))
    thermal_images = sorted(thermal_output_dir.glob("thermal_fused_image_*.png"), key=lambda p: int(re.search(r'_(\d+)', p.stem).group(1)))
    webcam_images = sorted(webcam_output_dir.glob("webcam_fused_image_*.png"), key=lambda p: int(re.search(r'_(\d+)', p.stem).group(1)))
    images_list = [lidar_images, thermal_images, webcam_images]
    
    for i in range(3):
        output_video = name_list[i]
        output_path = output_dir_list[i].joinpath(output_video)
        images = images_list[i]
        
        # Read first image to get dimensions
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape

        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
        video = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

        # Add images to video
        for img in images:
            frame = cv2.imread(img)
            video.write(frame)

        video.release()
        
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()
