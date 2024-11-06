def get_center_of_bbox(bbox):
    """
    Get the center of the bounding box
    Args:
        bbox: list of 4 elements representing the bounding box
    Returns:
        list of 2 elements representing the center of the bounding box
    """
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_width_of_bbox(bbox):
    """
    Get the width of the bounding box
    Args:
        bbox: list of 4 elements representing the bounding box
    Returns:
        int representing the width of the bounding box
    """
    return bbox[2]-bbox[0]

def measure_distance(point1, point2):
    """
    Measure the distance between two points
    Args:
        point1: list of 2 elements representing the first point
        point2: list of 2 elements representing the second point
    Returns:
        float representing the distance between the two points
    """
    return ((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    """
    Measure the distance between two points
    Args:
        p1: list of 2 elements representing the first point
        p2: list of 2 elements representing the second point
    Returns:
        float representing the distance between the two points
    """
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    """
    Get the foot position of the bounding box
    Args:
        bbox: list of 4 elements representing the bounding box
    Returns:
        list of 2 elements representing the foot position of the bounding box
    """
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2),int(y2)
    