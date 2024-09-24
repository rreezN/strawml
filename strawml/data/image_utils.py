import cv2
import numpy as np



def rotate_point(x: float, 
                 y: float, 
                 cx: float, 
                 cy: float, 
                 angle_radians: float) -> tuple:
    """
    Rotates a point (x, y) around a center point (cx, cy) by a given angle in radians.
    
    ...

    Parameters
    ----------
    x : float
        The x-coordinate of the point to rotate.
    y : float
        The y-coordinate of the point to rotate.
    cx : float
        The x-coordinate of the center of rotation.
    cy : float
        The y-coordinate of the center of rotation.
    angle_radians : float
        The rotation angle in radians.

    Returns
    -------
    tuple
        A tuple containing the rotated x and y coordinates.
    """
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    # Translate point back to origin
    x -= cx
    y -= cy

    # Rotate point
    x_new = x * cos_angle - y * sin_angle
    y_new = x * sin_angle + y * cos_angle

    # Translate point back
    x_new += cx
    y_new += cy

    return x_new, y_new

def rotate_bbox(bbox: list|np.ndarray, 
                img_width: int, 
                img_height: int, 
                angle_degrees: float) -> np.ndarray:
    """
    Rotates the bounding box defined by four corner points around the center of the image by a given angle.

    ...

    Parameters
    ----------
    bbox : list or np.ndarray
        Bounding box in [x1, y1, x2, y2, x3, y3, x4, y4] format.
    img_width : int
        Width of the image.
    img_height : int
        Height of the image.
    angle_degrees : float
        Angle to rotate the bounding box in degrees.

    Returns
    -------
    np.ndarray
        Rotated bounding box coordinates in [x1, y1, x2, y2, x3, y3, x4, y4] format.
    """
    # Convert angle to radians
    angle_radians = np.deg2rad(-angle_degrees)
    
    # Center of the image (we rotate the bounding box around this point)
    cx, cy = img_width / 2, img_height / 2

    # Initialize a list for rotated corner points
    rotated_coords = []

    # Rotate each corner point of the bbox
    for i in range(0, len(bbox), 2):
        x, y = bbox[i], bbox[i + 1]
        x_new, y_new = rotate_point(x, y, cx, cy, angle_radians)
        rotated_coords.append((x_new, y_new))

    # Clip the rotated coordinates to the image boundaries,
    # and flatten the coordinates to match [x1, y1, x2, y2, x3, y3, x4, y4] format
    rotated_coords = np.array(rotated_coords).flatten()
    rotated_bbox = clip_bbox_to_image(rotated_coords, img_width, img_height)

    return rotated_bbox


def clip_bbox_to_image(bbox: np.ndarray, 
                       img_width: int,
                       img_height: int) -> np.ndarray:
    """
    Clips the bounding box coordinates to ensure they stay within the image bounds.
    
    ...

    Parameters
    ----------
    bbox : np.ndarray
        Bounding box coordinates as a flattened array [x1, y1, x2, y2, ...].
    img_width : int
        Width of the image.
    img_height : int
        Height of the image.
    
    Returns
    -------
    np.ndarray
        Clipped bounding box coordinates as a flattened array.
    """
    bbox[0::2] = np.clip(bbox[0::2], 0, img_width - 1)  # Clip x-coordinates
    bbox[1::2] = np.clip(bbox[1::2], 0, img_height - 1)  # Clip y-coordinates
    return bbox


def rotate_image_and_bbox(image: np.ndarray, 
                          image_diff: np.ndarray, 
                          bbox: np.ndarray, 
                          angle_degrees: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotates the image and bounding box by the specified angle and clips the bounding box 
    coordinates to remain within the image borders.

    ...

    Parameters
    ----------
    image : np.ndarray
        The original image.
    image_diff : np.ndarray
        The image difference.
    bbox : np.ndarray
        Bounding box coordinates in the format [x1, y1, x2, y2, ...].
    angle_degrees : float
        Angle to rotate the image and bounding box in degrees.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the rotated image, rotated image difference, and rotated bounding box.
    """
    # Step 1: Rotate the image and image_diff
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_degrees, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    rotated_image_diff = cv2.warpAffine(image_diff, M, (w, h), flags=cv2.INTER_LINEAR)

    # Step 2: Rotate the bounding box corners
    img_height, img_width = rotated_image.shape[:2]
    rotated_bbox = rotate_bbox(bbox, img_width, img_height, angle_degrees)
    
    return rotated_image, rotated_image_diff, rotated_bbox