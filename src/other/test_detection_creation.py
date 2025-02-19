from mediapipe.tasks.python.components.containers import Detection, DetectionResult, BoundingBox, Category

def create_detection_result(detections_data):
    """
    Create a DetectionResult object with multiple detections.

    Args:
        detections_data (list): List of detections. Each detection is a dict with:
            - bounding_box: tuple (origin_x, origin_y, width, height)
            - score: float
            - category_name: str

    Returns:
        DetectionResult: A MediaPipe DetectionResult object.
    """
    detections = []
    for data in detections_data:
        # Create BoundingBox
        bounding_box = BoundingBox(
            origin_x=data["bounding_box"][0],
            origin_y=data["bounding_box"][1],
            width=data["bounding_box"][2],
            height=data["bounding_box"][3]
        )
        
        # Create Category
        category = Category(
            index=None,  # Optional, set if you have an index
            score=data["score"],
            display_name=None,  # Optional, set if you have a display name
            category_name=data["category_name"]
        )
        
        # Create Detection
        detection = Detection(
            bounding_box=bounding_box,
            categories=[category],
            keypoints=[]  # Optional, add keypoints if needed
        )
        
        detections.append(detection)
    
    # Create DetectionResult
    detection_result = DetectionResult(detections=detections)
    return detection_result

# Example usage
detections_data = [
    {
        "bounding_box": (24, 36, 42, 76),  # origin_x, origin_y, width, height
        "score": 0.9420368671417236,
        "category_name": "Person"
    },
    {
        "bounding_box": (23, 68, 10, 31),
        "score": 0.045699600130319595,
        "category_name": "Person"
    }
]

detection_result = create_detection_result(detections_data)
print(detection_result)
