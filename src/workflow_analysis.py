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
import json
import matplotlib.pyplot as plt


class Fused_Workflow:
    def __init__(self, iou_threshold, decision_making_mode, max_results, score_threshold):
        """Project FUSED Workflow Algorithm (refactored for analysis)

        Args:
            iou_threshold (float): Intersection-over-Union fraction threshold for evaluating bounding box overlaps
            decision_making_mode (str): Which sensors need to agree for the detection to be considered valid? 'all', 'thermal', or 'webcam'
            score_threshold (float): Minimum confidence score that the models must have for the detection to be kept
        """
        self.iou_threshold = iou_threshold
        self.decision_making_mode = decision_making_mode
        
        # Initialize the object detection models
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

        # Define the transformation matrices
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
        
        # Initialize image resolutions
        self.lidar_width = 224
        self.lidar_height = 172
        self.thermal_width = 160
        self.thermal_height = 120
        self.webcam_width = 320
        self.webcam_height = 240
        
    def fuse(self, lidar_image, thermal_image, webcam_image):
        """Main FUSED workflow function: perform fusion based alignment of object detection
        bounding boxes on synchronized LiDAR, thermal, and webcam images for decision making

        Args:
            lidar_image (OpenCV image): Synchronized LiDAR image
            thermal_image (OpenCV image): Synchronized thermal image
            webcam_image (OpenCV image): Synchronized webcam image

        Returns:
            detection results (tuple): 6 detection results - 3 from individual sensors, 3 from FUSED workflow
        """
        # Perform LiDAR image processing
        max_depth = np.max(lidar_image)
        lidar_image_clipped = np.clip(lidar_image, 0, max_depth)
        lidar_image_mm = lidar_image_clipped * 1000
        lidar_image_normalized = cv2.normalize(lidar_image_mm, None, 0, 65535, cv2.NORM_MINMAX)
        lidar_image_8bit = cv2.convertScaleAbs(lidar_image_normalized, alpha=(255.0 / np.max(lidar_image_normalized)))
        lidar_image_equalized = cv2.equalizeHist(lidar_image_8bit)

        # Convert OpenCV images to RGB format
        lidar_image_rgb = cv2.cvtColor(lidar_image_equalized, cv2.COLOR_GRAY2RGB)
        thermal_image_rgb = cv2.cvtColor(thermal_image, cv2.COLOR_GRAY2RGB)
        webcam_image_rgb = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)

        # Convert RGB images to MediaPipe images
        lidar_image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=lidar_image_rgb)
        thermal_image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=thermal_image_rgb)
        webcam_image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=webcam_image_rgb)

        # Perform object detection on the MediaPipe images
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

        # For loop through each LiDAR detection in the detection result
        if lidar_detection_result.detections:
            for detection in lidar_detection_result.detections:
                if detection.categories[0].category_name != 'Person':
                    continue
                # Define the top left and bottom right points of the detection
                bbox = detection.bounding_box
                x1, y1 = bbox.origin_x, bbox.origin_y # Top left
                x2, y2 = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height # Bottom right

                # Find the depth on the LiDAR image at the center of the box
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

                # If depth is not zero, then compute transformed u and v on webcam and thermal frames
                if zₗ > 1E-3:
                    x1ₗₜ, y1ₗₜ, x1ₗᵣ, y1ₗᵣ = self.transform(zₗ, x1, y1)
                    x2ₗₜ, y2ₗₜ, x2ₗᵣ, y2ₗᵣ = self.transform(zₗ, x2, y2)

                    # Calculate IoU between the mapped bounding box and all detection results from the webcam and thermal images
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

                    # Choose the thermal or webcam detection result corresponding to the LiDAR mapped result whose IoU is the 
                    # largest and also above the defined Combination IoU threshold. In the next iterations of the for loop,
                    # the thermal or webcam detection result that was chosen should not be chosen again to match with another
                    # LiDAR mapped result
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

                    # Depending on the decision making mode, choose to either keep the mapped result or not based on whether there 
                    # is agreement between all 3 or only two sensors
                    # If the mapped result is not being kept, then go to the next iteration of the loop. If it is being kept, then
                    # keep the original detections that have been agreed upon according to the decision making mode. For the 
                    # detection that has not been agreed upon, check if it agrees with LiDAR. If it does, keep it. If it does not,
                    # then use the mapped LiDAR detection onto the appropriate camera frame instead
                    # Store the three fused detection results at each iteration
                    if self.decision_making_mode == 'all':
                        if valid_thermal_iou and valid_webcam_iou:
                            lidar_fused_detections.append(self.truncate(detection, self.lidar_width, self.lidar_height))
                            thermal_fused_detections.append(self.truncate(thermal_detection_result.detections[valid_thermal_idx], self.thermal_width, self.thermal_height))
                            webcam_fused_detections.append(self.truncate(webcam_detection_result.detections[valid_webcam_idx], self.webcam_width, self.webcam_height))
                        else:
                            continue

                    if self.decision_making_mode == 'thermal':
                        if valid_thermal_iou:
                            lidar_fused_detections.append(self.truncate(detection, self.lidar_width, self.lidar_height))
                            thermal_fused_detections.append(self.truncate(thermal_detection_result.detections[valid_thermal_idx], self.thermal_width, self.thermal_height))
                            if valid_webcam_iou:
                                webcam_fused_detections.append(self.truncate(webcam_detection_result.detections[valid_webcam_idx], self.webcam_width, self.webcam_height))
                            else:
                                webcam_fused_detections.append(self.truncate(self.create_detection(detection, webcam_mapped_box), self.webcam_width, self.webcam_height))
                        else:
                            continue

                    if self.decision_making_mode == 'webcam':
                        if valid_webcam_iou:
                            lidar_fused_detections.append(self.truncate(detection, self.lidar_width, self.lidar_height))
                            webcam_fused_detections.append(self.truncate(webcam_detection_result.detections[valid_webcam_idx], self.webcam_width, self.webcam_height))
                            if valid_thermal_iou:
                                thermal_fused_detections.append(self.truncate(thermal_detection_result.detections[valid_thermal_idx], self.thermal_width, self.thermal_height))
                            else:
                                thermal_fused_detections.append(self.truncate(self.create_detection(detection, thermal_mapped_box), self.thermal_width, self.thermal_height))
                        else:
                            continue
                else:
                    continue

        # With all of the fused detections, create detection results
        if not lidar_fused_detections:
            lidar_fused_detection_result = None
        else:
            lidar_fused_detection_result = DetectionResult(detections=lidar_fused_detections)
            
        if not thermal_fused_detections:
            thermal_fused_detection_result = None
        else:
            thermal_fused_detection_result = DetectionResult(detections=thermal_fused_detections)
            
        if not webcam_fused_detections:
            webcam_fused_detection_result = None
        else:
            webcam_fused_detection_result = DetectionResult(detections=webcam_fused_detections)
            
        return lidar_detection_result, thermal_detection_result, webcam_detection_result, lidar_fused_detection_result, thermal_fused_detection_result, webcam_fused_detection_result
            
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
    
    def truncate(self, detection, img_width, img_height): 
        """Truncate a bounding box detection so that the coordinates remain within the image
        and don't go over the edges

        Args:
            detection (object): MediaPipe detection object
            img_width (int): Image width (pixels)
            img_height (int): Image height (pixels)

        Returns:
            new_detection (object): Updated MediaPipe detection object with truncated bounding box coordinates
        """
        # Extract bounding box coordinates
        bbox = detection.bounding_box
        box_x1 = bbox.origin_x
        box_y1 = bbox.origin_y
        box_width = bbox.width
        box_height = bbox.height
        box_x2 = box_x1 + box_width
        box_y2 = box_y1 + box_height
        
        # Fix box corners if they are off the image
        if box_x1 < 0:
            box_x1 = 0
            
        if box_x2 > img_width:
            box_x2 = img_width
            
        if box_y1 < 0:
            box_y1 = 0
            
        if box_y2 > img_height:
            box_y2 = img_height
        
        # Calculate new width and height    
        box_width = box_x2 - box_x1
        box_height = box_y2 - box_y1
    
        # Redefine bounding box coordinates
        detection.bounding_box.origin_x = box_x1
        detection.bounding_box.origin_y = box_y1
        detection.bounding_box.width = box_width
        detection.bounding_box.height = box_height
        
        return detection
    
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


def gather_results(i, directory, fused, idv_lidar_det_results, idv_thermal_det_results, idv_webcam_det_results, \
        fused_lidar_det_results, fused_thermal_det_results, fused_webcam_det_results):
    """Loop through and sort the images required for the FUSED workflow, call the FUSED workflow, and
    store detection results and image filenames in arrays

    Args:
        i (int): Iterator for the for loop
        directory (Path object): Path for the directory containing the images
        fused (class): Instance of the Fused_Workflow class
        idv_lidar_det_results (numpy array): Empty numpy array
        idv_thermal_det_results (numpy array): Empty numpy array
        idv_webcam_det_results (numpy array): Empty numpy array
        fused_thermal_det_results (numpy array): Empty numpy array
        fused_webcam_det_results (numpy array): Empty numpy array

    Returns:
        detection results (tuple): All numpy arrays with detections and image filenames added
    """
    
    # For loop through each set of synchronized images
    for path in directory.glob('*'):
        # Extract category
        category = str(path).rsplit('_')[-3]
        
        # If LiDAR, grab the two corresponding thermal and webcam paths
        if category == 'lidar': 
            # Redefine path variable
            lidar_path = path
            
            # Pull scenario and number for matching files
            scenario = '_'.join(str(path).rsplit('/')[-1].rsplit('_')[:-3])
            number = str(path).rsplit('_')[-1].rsplit('.')[0]
           
            # Thermal and webcam path re-creation
            thermal_path = directory.joinpath(scenario + '_thermal_image_' + number + '.png')
            webcam_path = directory.joinpath(scenario + '_webcam_image_' + number + '.png')
           
            # Read in images
            lidar_image = cv2.imread(lidar_path, cv2.IMREAD_UNCHANGED)
            thermal_image = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
            webcam_image = cv2.imread(webcam_path, cv2.IMREAD_UNCHANGED)
            
            # Call the fuse method
            idv_lidar, idv_thermal, idv_webcam, fused_lidar, fused_thermal, fused_webcam = fused.fuse(lidar_image, thermal_image, webcam_image)
            
            # Add individual results to matrices
            idv_lidar_det_results[i,0] = lidar_path
            idv_lidar_det_results[i,1] = idv_lidar
            idv_thermal_det_results[i,0] = thermal_path
            idv_thermal_det_results[i,1] = idv_thermal
            idv_webcam_det_results[i,0] = webcam_path
            idv_webcam_det_results[i,1] = idv_webcam
            
            # Add fused results to matrices
            fused_lidar_det_results[i,0] = lidar_path
            fused_lidar_det_results[i,1] = fused_lidar
            fused_thermal_det_results[i,0] = thermal_path
            fused_thermal_det_results[i,1] = fused_thermal
            fused_webcam_det_results[i,0] = webcam_path
            fused_webcam_det_results[i,1] = fused_webcam
            
            # Update iterator
            i += 1
            
        else: 
            continue
    
    return i, idv_lidar_det_results, idv_thermal_det_results, idv_webcam_det_results, fused_lidar_det_results, fused_thermal_det_results, fused_webcam_det_results

def ap(det_results, ground_truth, iou_threshold, fused, title):
    # Find total number of images from the matrix
    n = det_results.shape[0]
    
    # Find total number of algorithm detections
    det_results_list = det_results[:,1]
    total_num_algorithm_detections = 0
    for det_res in det_results_list:
        if det_res is not None:
            total_num_algorithm_detections = total_num_algorithm_detections + len(det_res.detections)
    
    # Initialize output
    ap_detections = np.empty((total_num_algorithm_detections, 4), dtype=object)
    
    # Find total number of ground truth detections - LiDAR, thermal, or webcam
    det_results_category = str(det_results[0,0]).rsplit('_', 3)[-3]
    total_num_ground_truth_detections = 0
    for annotation in ground_truth['annotations']:
        image_id = annotation['image_id']
        for image in ground_truth['images']:
            if image['id'] == image_id:
                filename = image['file_name']
                category = str(filename).rsplit('_', 3)[-3]
                if category == det_results_category:
                    total_num_ground_truth_detections += 1
    
    # Initialize iterator for filling in data
    j = 0
    
    # Loop through each image
    for i in range(n):
        # Define the image filename and the detection result from that image
        image_path = det_results[i,0]
        filename = image_path.stem + image_path.suffix
        det_result = det_results[i,1]
        
        # Loop through all ground truth images to find the correct ground truth image corresponding to the algorithm image
        image_id = None
        for image in ground_truth['images']:
            if filename == image['file_name']:
                image_id = image['id']
                break
                
        # If the image ID is not None (there are ground truth detections in the image), then do the following
        if image_id is not None:
            # Grab all ground truth detections for the correct image
            ground_truth_detections = []
            for annotation in ground_truth['annotations']:
                if annotation['image_id'] == image_id:
                    ground_truth_detections.append(annotation)
            
            # Define the excluding index list
            exclude_idx = []
            
            # Define counters to see if there are leftover algorithm detections that do not match ground truth detections
            num_ground_truth_detections = len(ground_truth_detections)
            num_algorithm_detections_so_far = 0
            
            # Loop through all ground truth detections
            for ground_truth_detection in ground_truth_detections:
                ground_truth_bbox = ground_truth_detection['bbox']
                ground_truth_box = (ground_truth_bbox[0], ground_truth_bbox[1], ground_truth_bbox[0] + ground_truth_bbox[2], ground_truth_bbox[1] + ground_truth_bbox[3])
                
                # If there are no algorithm detections, then skip
                # Also, if the number of algorithm detections is the same as the number of excluded algorithm detections, then skip
                # Compute IoUs between all algorithm detections that have not been excluded and the single ground truth detection
                if det_result is not None:
                    if det_result.detections and len(det_result.detections) != len(exclude_idx):
                            ious = []
                            for idx, algorithm_detection in enumerate(det_result.detections):
                                category = algorithm_detection.categories[0].category_name
                                if category != 'Person' and category != 'person':
                                    ious.append(0.0)
                                    continue
                                if idx in exclude_idx:
                                    ious.append(0.0)
                                    continue
                                bbox = algorithm_detection.bounding_box
                                x1, y1 = bbox.origin_x, bbox.origin_y
                                x2, y2 = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                                algorithm_box = (x1, y1, x2, y2)
                                ious.append(fused.calc_iou(algorithm_box, ground_truth_box))
                
                # If the maximum IoU from the list is larger than the threshold, then the algorithm detection agrees with the ground truth
                if det_result is not None:
                    if det_result.detections and len(det_result.detections) != len(exclude_idx):
                        max_iou = max(ious)
                        max_iou_index = ious.index(max_iou)
                        if max_iou > iou_threshold:
                            valid_idx = max_iou_index
                            exclude_idx.append(valid_idx)
                            ap_detections[j, 0] = det_result.detections[valid_idx].categories[0].score
                            ap_detections[j, 1] = det_result.detections[valid_idx]
                            ap_detections[j, 2] = True
                            ap_detections[j, 3] = filename
                            j += 1
                            num_algorithm_detections_so_far += 1
                            
                        # If the maximum IoU is not larger than the threshold, then continue
                        else:
                            continue
                    
            # If there are leftover algorithm detections that did not pair with ground truth, then add those as False
            if num_algorithm_detections_so_far != num_ground_truth_detections:
                if det_result is not None:
                    if det_result.detections:
                        for idx, algorithm_detection in enumerate(det_result.detections):
                            category = algorithm_detection.categories[0].category_name
                            if category != 'Person' and category != 'person':
                                continue
                            if idx in exclude_idx:
                                continue
                            ap_detections[j, 0] = algorithm_detection.categories[0].score
                            ap_detections[j, 1] = algorithm_detection
                            ap_detections[j, 2] = False
                            ap_detections[j, 3] = filename
                            j += 1
            
        # If image ID is None, then there are no ground truth detections and any algorithm detections are False
        else:
            if det_result is not None:
                if det_result.detections:
                    for algorithm_detection in det_result.detections:
                        ap_detections[j, 0] = algorithm_detection.categories[0].score
                        ap_detections[j, 1] = algorithm_detection
                        ap_detections[j, 2] = False
                        ap_detections[j, 3] = filename
                        j += 1
    
    # Sort the detections by order of decreasing confidence
    ap_detections = ap_detections[ap_detections[:, 0].astype(float).argsort()[::-1]]
    
    # Calculate precision and recall for each detection in the list
    precisions = np.empty(total_num_algorithm_detections, dtype=float)
    recalls = np.empty(total_num_algorithm_detections, dtype=float)
    trues = 0
    falses = 0
    for k in range(total_num_algorithm_detections):
        ap_detection = ap_detections[k,:]
        if ap_detection[2] == True:
            trues += 1
        if ap_detection[2] == False:
            falses += 1
        precisions[k] = trues / (trues + falses)
        recalls[k] = trues / total_num_ground_truth_detections
        
    # Smooth out the PR curve
    # p_prev = 0
    # for p in precisions[::-1]:
    #     p_next = p
    #     if p_next < p_prev:
    #         p_next = p_prev
    #     idx = np.argwhere(precisions == p)
    #     precisions[idx] = p_next
    #     p_prev = p_next
    
    # Compute AP by finding area under the PR curve
    ap = np.trapz(precisions, recalls)
    
    # Create PR Curve for showing
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recalls, precisions, linestyle='-', color='b')
    
    # Labels and title
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve - {title}")
    
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    
    # Grid and legend
    ax.grid(True)
    ax.legend()
    
    fig.savefig(Path('/project_fused/output/figures').joinpath(title + '.png'), dpi=300, bbox_inches='tight')  
    plt.close(fig)
    
    return ap

def main():
    # Get the number of images in the chosen test directory
    analysis_dataset_dir = Path('/project_fused/data/Analysis_Dataset')
    num_images_labeled = len(list(analysis_dataset_dir.joinpath('labeled_images').glob('*')))
    num_images_unlabeled = len(list(analysis_dataset_dir.joinpath('unlabeled_images').glob('*')))
    size = int((num_images_labeled + num_images_unlabeled)/3)
    
    # Choose whether to employ PR curve monotonicity correction
    smoothing = 'Without Curve Smoothing' # 'With Curve Smoothing' or 'Without Curve Smoothing'
    
    # Workflow Inputs:
    iou_threshold = 0.4
    decision_making_mode = 'thermal' # options are 'all', 'thermal', and 'webcam'
    max_results = 3
    score_threshold = 0.5 # Will set higher to mitigate random sensor agreement due to sheer
                          # number of garbage baxes being returned otherwise
    # Read in ground truth
    with open(analysis_dataset_dir.joinpath('labels.json'), 'r') as json_file:
        ground_truth = json.load(json_file)
        
    # Initialize FUSED class
    fused = Fused_Workflow(iou_threshold, decision_making_mode, max_results, score_threshold)

    # Initialize individual arrays
    idv_lidar_det_results = np.empty((size, 2), dtype=object)
    idv_thermal_det_results = np.empty((size, 2), dtype=object)
    idv_webcam_det_results = np.empty((size, 2), dtype=object)
    
    # Initialize fused arrays
    fused_lidar_det_results = np.empty((size, 2), dtype=object)
    fused_thermal_det_results = np.empty((size, 2), dtype=object)
    fused_webcam_det_results = np.empty((size, 2), dtype=object)
    
    # Gather results for both labeled and unlabeled images
    labeled_dir = analysis_dataset_dir.joinpath('labeled_images')
    unlabeled_dir = analysis_dataset_dir.joinpath('unlabeled_images')
    i = 0
    i, idv_lidar_det_results, idv_thermal_det_results, idv_webcam_det_results, fused_lidar_det_results, fused_thermal_det_results, fused_webcam_det_results = gather_results(i, labeled_dir, fused, idv_lidar_det_results, idv_thermal_det_results, idv_webcam_det_results, fused_lidar_det_results, fused_thermal_det_results, fused_webcam_det_results)
    i, idv_lidar_det_results, idv_thermal_det_results, idv_webcam_det_results, fused_lidar_det_results, fused_thermal_det_results, fused_webcam_det_results = gather_results(i, unlabeled_dir, fused, idv_lidar_det_results, idv_thermal_det_results, idv_webcam_det_results, fused_lidar_det_results, fused_thermal_det_results, fused_webcam_det_results)
        
    # Calculate AP
    ap_idv_lidar = ap(idv_lidar_det_results, ground_truth, iou_threshold, fused, f'Individual LiDAR {smoothing}')
    ap_idv_thermal = ap(idv_thermal_det_results, ground_truth, iou_threshold, fused, f'Individual Thermal {smoothing}')
    ap_idv_webcam = ap(idv_webcam_det_results, ground_truth, iou_threshold, fused, f'Individual Webcam {smoothing}')
    ap_fused_lidar = ap(fused_lidar_det_results, ground_truth, iou_threshold, fused, f'Fused LiDAR {smoothing}')
    ap_fused_thermal = ap(fused_thermal_det_results, ground_truth, iou_threshold, fused, f'Fused Thermal {smoothing}')
    ap_fused_webcam = ap(fused_webcam_det_results, ground_truth, iou_threshold, fused, f'Fused Webcam {smoothing}')
    
    # Print APs
    print(f'Individual LiDAR AP: {ap_idv_lidar}\n')
    print(f'Individual Thermal AP: {ap_idv_thermal}\n')
    print(f'Individual Webcam AP: {ap_idv_webcam}\n')
    print(f'Fused LiDAR AP: {ap_fused_lidar}\n')
    print(f'Fused Thermal AP: {ap_fused_thermal}\n')
    print(f'Fused Webcam AP: {ap_fused_webcam}\n')
    

if __name__ == '__main__':
    main()