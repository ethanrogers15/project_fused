def average_detections(self, zₗ, lidar_detection, thermal_detection, webcam_detection, thermal_mapped_box, webcam_mapped_box):
    # Get detection coordinates
    thermal_bbox = thermal_detection.bounding_box
    x1ₜ, y1ₜ = thermal_bbox.origin_x, thermal_bbox.origin_y
    x2ₜ, y2ₜ = thermal_bbox.origin_x + thermal_bbox.width, thermal_bbox.origin_y + thermal_bbox.height
    webcam_bbox = webcam_detection.bounding_box
    x1ᵣ, y1ᵣ = webcam_bbox.origin_x, webcam_bbox.origin_y
    x2ᵣ, y2ᵣ = webcam_bbox.origin_x + webcam_bbox.width, webcam_bbox.origin_y + webcam_bbox.height
    x1ₗₜ, y1ₗₜ, x2ₗₜ, y2ₗₜ = thermal_mapped_box
    x1ₗᵣ, y1ₗᵣ, x2ₗᵣ, y2ₗᵣ = webcam_mapped_box
    
    # Average coordinates between mapped and original thermal / webcam results
    x1_avgₜ = (x1ₜ + x1ₗₜ) / 2
    x2_avgₜ = (x2ₜ + x2ₗₜ) / 2
    y1_avgₜ = (y1ₜ + y1ₗₜ) / 2
    y2_avgₜ = (y2ₜ + y2ₗₜ) / 2
    x1_avgᵣ = (x1ᵣ + x1ₗᵣ) / 2
    x2_avgᵣ = (x2ᵣ + x2ₗᵣ) / 2
    y1_avgᵣ = (y1ᵣ + y1ₗᵣ) / 2
    y2_avgᵣ = (y2ᵣ + y2ₗᵣ) / 2
    
    # Average scores between mapped and original thermal / webcam / LiDAR results
    thermal_avg_score = (lidar_detection.categories[0].score + thermal_detection.categories[0].score) / 2
    webcam_avg_score = (lidar_detection.categories[0].score + webcam_detection.categories[0].score) / 2
    lidar_avg_score = (lidar_detection.categories[0].score + thermal_detection.categories[0].score + webcam_detection.categories[0].score) / 3
    
    # Transform average results from thermal / webcam frames to the LiDAR frame
    u1ₗₜ, v1ₗₜ, u1ₗᵣ, v1ₗᵣ = self.transform_back(zₗ, x1_avgₜ, y1_avgₜ, x1_avgᵣ, y1_avgᵣ)
    u2ₗₜ, v2ₗₜ, u2ₗᵣ, v2ₗᵣ = self.transform_back(zₗ, x2_avgₜ, y2_avgₜ, x2_avgᵣ, y2_avgᵣ)
    
    # Average the two transformed average results to get the workflow result on the LiDAR frame
    x1_avgₗ = (u1ₗₜ + u1ₗᵣ) / 2
    x2_avgₗ = (u2ₗₜ + u2ₗᵣ) / 2
    y1_avgₗ = (v1ₗₜ + v1ₗᵣ) / 2
    y2_avgₗ = (v2ₗₜ + v2ₗᵣ) / 2
    
    # Create new detections for the averaged detections
    avg_lidar_detection = {
        "bounding_box": (x1_avgₗ, y1_avgₗ, x2_avgₗ - x1_avgₗ, y2_avgₗ - y1_avgₗ),
        "score": lidar_avg_score,
        "category_name": "Person"
    }
    avg_lidar_detection = self.create_detection(avg_lidar_detection)
    
    avg_thermal_detection = {
        "bounding_box": (x1_avgₜ, y1_avgₜ, x2_avgₜ - x1_avgₜ, y2_avgₜ - y1_avgₜ),
        "score": thermal_avg_score,
        "category_name": "Person"
    }
    avg_thermal_detection = self.create_detection(avg_thermal_detection)
    
    avg_webcam_detection = {
        "bounding_box": (x1_avgᵣ, y1_avgᵣ, x2_avgᵣ - x1_avgᵣ, y2_avgᵣ - y1_avgᵣ),
        "score": webcam_avg_score,
        "category_name": "Person"
    }
    avg_webcam_detection = self.create_detection(avg_webcam_detection)
    
    return avg_lidar_detection, avg_thermal_detection, avg_webcam_detection

def transform_back(self, zₗ, uₜ, vₜ, uᵣ, vᵣ):
    """Perform transformations to map pixel coordinates from thermal and webcam camera frames back to the LiDAR camera frame

    Args:
        zₗ (float): Depth, in meters
        uₜ (int): Thermal pixel coordinate on the x axis
        vₜ (int): Thermal pixel coordinate on the y axis
        uᵣ (int): Webcam pixel coordinate on the x axis
        vᵣ (int): Webcam pixel coordinate on the y axis

    Returns:
        uₗₜ, vₗₜ, uₗᵣ, vₗᵣ (tuple): LiDAR frame pixel coordinates transformed from both thermal and webcam frames, respectively
    """
    # Perform intrinsic transformations to get line of sight vectors
    pₜ = array([uₜ, vₜ, 1])
    l̂ₜ = inv(self.Kₜ) @ pₜ 
    pᵣ = array([uᵣ, vᵣ, 1])
    l̂ᵣ = inv(self.Kᵣ) @ pᵣ
    
    # Add depth for position vectors
    r̄ₜ = zₗ * l̂ₜ 
    r̄ᵣ = zₗ * l̂ᵣ
    
    # Perform extrinsic transformations to the LiDAR sensor
    r̄ₗₜ = (inv(self.R_l2cₜ) @ (self.R_t2cₜ @ r̄ₜ)) - array([self.T_l2t[0, 3], self.T_l2t[1, 3], 0]).T
    r̄ₗᵣ = (inv(self.R_l2cᵣ) @ (self.R_w2cᵣ @ r̄ᵣ)) - array([self.T_l2w[0, 3], self.T_l2w[1, 3], 0]).T
    
    # Transform to pixel coordinates
    r̃ₗₜ = array([r̄ₗₜ[0]/r̄ₗₜ[2], r̄ₗₜ[1]/r̄ₗₜ[2], r̄ₗₜ[2]/r̄ₗₜ[2]])
    r̃ₗᵣ = array([r̄ₗᵣ[0]/r̄ₗᵣ[2], r̄ₗᵣ[1]/r̄ₗᵣ[2], r̄ₗᵣ[2]/r̄ₗᵣ[2]])
    pₗₜ = self.Kₗ @ r̃ₗₜ 
    pₗᵣ = self.Kₗ @ r̃ₗᵣ 
    uₗₜ, vₗₜ = pₗₜ[0], pₗₜ[1]
    uₗᵣ, vₗᵣ = pₗᵣ[0], pₗᵣ[1]
    
    return uₗₜ, vₗₜ, uₗᵣ, vₗᵣ 