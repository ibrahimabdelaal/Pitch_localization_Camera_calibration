import numpy as np

def get_standard_pitch_points():
    """
    Returns standard soccer pitch points in meters.
    Based on FIFA regulations for standard pitch dimensions.
    """
    # Standard pitch dimensions (in meters)
    LENGTH = 105.0  # FIFA standard length
    WIDTH = 68.0    # FIFA standard width
    
    # Define key points (in meters from center of pitch)
    points = {
        # Center circle
        'center': (0, 0),
        
        # Penalty areas
        'penalty_left': (-LENGTH/2 + 16.5, 0),
        'penalty_right': (LENGTH/2 - 16.5, 0),
        
        # Goal areas
        'goal_left': (-LENGTH/2 + 5.5, 0),
        'goal_right': (LENGTH/2 - 5.5, 0),
        
        # Corner points
        'corner_tl': (-LENGTH/2, WIDTH/2),
        'corner_tr': (LENGTH/2, WIDTH/2),
        'corner_bl': (-LENGTH/2, -WIDTH/2),
        'corner_br': (LENGTH/2, -WIDTH/2),
        
        # Halfway line endpoints
        'halfway_t': (0, WIDTH/2),
        'halfway_b': (0, -WIDTH/2),
    }
    
    # Convert to numpy array
    points_array = np.array(list(points.values()), dtype=np.float32)
    
    return points_array

def map_image_to_pitch_points(image_points):
    """
    Maps image points to their corresponding pitch coordinates.
    
    Args:
        image_points: numpy array of shape (N, 2) containing image coordinates
    
    Returns:
        pitch_points: numpy array of shape (N, 2) containing pitch coordinates
    """
    pitch_points = get_standard_pitch_points()
    
    # Ensure we have the same number of points
    if len(image_points) != len(pitch_points):
        raise ValueError(f"Number of image points ({len(image_points)}) does not match "
                       f"number of pitch points ({len(pitch_points)})")
    
    return pitch_points 