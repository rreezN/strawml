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

def rotate_bbox(bbox: np.ndarray, 
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
    rotated_image, rotated_image_diff, rotated_bbox = SpecialRotate(image, image_diff, bbox, angle_degrees)
    # # rotated_image_diff, _, _ = SpecialRotate(image_diff, bbox, angle_degrees)
    # h, w = image.shape[:2]
    # cx, cy = w // 2, h // 2
    # M = cv2.getRotationMatrix2D((cx, cy), angle_degrees, 1.0)
    # rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    # rotated_image_diff = cv2.warpAffine(image_diff, M, (w, h), flags=cv2.INTER_LINEAR)

    # # Step 2: Rotate the bounding box corners
    # img_height, img_width = rotated_image.shape[:2]
    # rotated_bbox = rotate_bbox(bbox, img_width, img_height, angle_degrees)
    
    return rotated_image, rotated_image_diff, rotated_bbox


def internal_image_operations(image: np.ndarray) -> np.ndarray:
    """
    Perform random operations on the image of size (1440, 1250) to simulate the chute environment,
    without including the chute. The chute has numbers on it itself, and without labels on the chute numbers,
    they might cause confusion in the model during training. This functions crops the image to 500 x 500 pixels 
    and randomly rotates the image between -180 and 180 degrees, while ensuring that the cropped image is inside 
    the original image, and the resulting image has no black borders from rotation.
    """
    # sample a random angle between -180 and 180 degrees
    angle = np.random.randint(-180, 181)
    # rotate the image
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # randomly crop the image to 500 x 500 pixels
    x = np.random.randint(0, image.shape[1] - 500)
    y = np.random.randint(0, image.shape[0] - 500)
    image = image[y:y+500, x:x+500]
    return image 

def SpecialRotate(image, image_diff=None, bbox=None, angle=None, return_affine=False): 
    """
    Rotate image without cutting off sides and make borders transparent.
    Inspiration from: https://www.geeksforgeeks.org/rotate-image-without-cutting-off-sides-using-python-opencv/
    """
    # Taking image height and width 
    height, width = image.shape[:2]
    # get the image centers
    image_center = (width/2, height/2)

    rotation_arr = cv2.getRotationMatrix2D(image_center, float(angle), 1)

    abs_cos = abs(rotation_arr[0,0])
    abs_sin = abs(rotation_arr[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_arr[0, 2] += bound_w/2 - image_center[0]
    rotation_arr[1, 2] += bound_h/2 - image_center[1]

    rotated_image = cv2.warpAffine(image, rotation_arr, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 255))
    affine_warp = np.vstack((rotation_arr, np.array([0,0,1])))
    bbox_ = np.expand_dims(bbox.reshape(-1, 2), 1)
    bbox_ = cv2.perspectiveTransform(bbox_, affine_warp)
    rotated_bbox = np.squeeze(bbox_, 1).flatten()
    
    if image_diff is None:
        if return_affine:
            return rotated_image, None, rotated_bbox, affine_warp
        return rotated_image, None, rotated_bbox
    
    rotated_image_diff = cv2.warpAffine(image_diff, rotation_arr, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 255))
    
    if return_affine:
        return rotated_image, rotated_image_diff, rotated_bbox, affine_warp
    return rotated_image, rotated_image_diff, rotated_bbox

def overlay_image(bg, fg, default_rot, fg_size):
    # get the dimensions of the background image
    bg_h, bg_w, _ = bg.shape
    # get the dimensions of the foreground image
    fg_h, fg_w, _ = fg.shape
    # get the location to place the foreground image
    x = np.random.randint(0, bg_w - fg_w)
    y = np.random.randint(0, bg_h - fg_h)
    # get the alpha mask of the foreground image        
    fg_alpha = fg[:, :, 3] / 255.0
    # get the alpha mask of the background image
    bg_alpha = 1.0 - fg_alpha
    # overlay the images
    for c in range(0, 3):
        bg[y:y+fg_h, x:x+fg_w, c] = (fg_alpha * fg[:, :, c] + bg_alpha * bg[y:y+fg_h, x:x+fg_w, c])

    # Get the rotated bounding box coordinates of the digit image on the chute image, x1, y1, x2, y2, x3, y3, x4, y4
    # The default rotation matrix is used to rotate the bounding box coordinates from the center of the fg image.
    # Here we also import the original image sizes of the digits.
    fg_h_, fg_w_ = fg_size
    center_x, center_y = x + fg_w / 2, y + fg_h / 2
    bbox = np.array([x - center_x + abs(fg_w - fg_w_), y - center_y + abs(fg_h - fg_h_),
                    (x + fg_w) - center_x, y - center_y + abs(fg_h - fg_h_),
                    (x + fg_w) - center_x, (y + fg_h) - center_y,
                    x - center_x + abs(fg_w - fg_w_), (y + fg_h) - center_y])
    bbox = np.dot(np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]], [bbox[6], bbox[7]]]), default_rot[:, :2].T) + np.array([center_x, center_y])
    fg_h, fg_w = fg_size
    bbox[:, 0] /= bg_w  # normalize the x-coordinates
    bbox[:, 1] /= bg_h  # normalize the y-coordinates
    bbox = bbox.flatten()
    return bg, bbox

def create_random_permutations_with_repeats(data, labels, total_permutations, min_size=3, max_size=5):
    """
    Creates random permutations of images, ensuring all images are used at least once before allowing repetitions.
    
    Parameters:
        data (np.ndarray): The dataset of images of shape (N, 28, 28), where N is the number of images.
        labels (np.ndarray): The corresponding labels of the dataset.
        total_permutations (int): Total number of permutations required.
        min_size (int): Minimum size of permutation. Default is 3.
        max_size (int): Maximum size of permutation. Default is 5.
        
    Returns:
        List of tuples: Each tuple contains a random permutation of images and corresponding labels.
    """
    num_images = data.shape[0]
    indices = np.arange(num_images)
    np.random.shuffle(indices)  # Shuffle indices to randomize initial ordering
    
    permutations = []
    
    # Stage 1: Use all images at least once
    used_images = 0
    while used_images < num_images:
        # Randomly choose a size between min_size and max_size for the current permutation
        perm_size = np.random.randint(min_size, max_size + 1)
        
        # Ensure we don't exceed the number of images
        if used_images + perm_size > num_images:
            perm_size = num_images - used_images
        
        # Get the indices for this permutation
        perm_indices = indices[used_images:used_images + perm_size]
        
        # Collect the images and labels for this permutation
        perm_data = data[perm_indices]
        perm_labels = labels[perm_indices]
        
        # Append the permutation to the list
        permutations.append((perm_data, perm_labels))
        
        # Move the index forward by the size of the current permutation
        used_images += perm_size
    
    # Stage 2: Allow overlapping permutations to meet the required total_permutations
    while len(permutations) < total_permutations:
        # Randomly choose a size between min_size and max_size
        perm_size = np.random.randint(min_size, max_size + 1)
        
        # Randomly sample indices from the full dataset, allowing repetitions
        perm_indices = np.random.choice(indices, size=perm_size, replace=False)
        
        # Collect the images and labels for this permutation
        perm_data = data[perm_indices]
        perm_labels = labels[perm_indices]
        
        # Append the new permutation
        permutations.append((perm_data, perm_labels))
    
    return permutations


def resize_all_images_in_dir(data_path, new_size=(640, 640)):
    """
    function that finds and resizes all images in a directory including subdirectories to a new size.
    """

    from pathlib import Path
    from tqdm import tqdm
    import os
    # Get all image file paths
    image_paths = list(Path(data_path).rglob("*.jpg"))
    label_paths = list(Path(data_path).rglob("*.txt"))

    og_w, og_h = 2560, 1440

    def resize_bbox(label_path, og_w, og_h, new_size):
        # we now take and resize the bounding box coordinates x1, y1, x2, y2, x3, y3, x4, y4, knowing they have been normalized with the original image size
        with open(label_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            cls, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.split())
            # Unnormalize the bounding box coordinates to the original image size
            x1, x2, x3, x4 = [x * og_w for x in [x1, x2, x3, x4]]
            y1, y2, y3, y4 = [y * og_h for y in [y1, y2, y3, y4]]
            # resize the coordinates to the new image size
            h_scale = new_size[1] / og_h
            w_scale = new_size[0] / og_w
            x1, x2, x3, x4 = [x * w_scale for x in [x1, x2, x3, x4]]
            y1, y2, y3, y4 = [y * h_scale for y in [y1, y2, y3, y4]]
            # now we normalize the coordinates to the new image size
            x1, x2, x3, x4 = [x / new_size[0] for x in [x1, x2, x3, x4]]
            y1, y2, y3, y4 = [y / new_size[1] for y in [y1, y2, y3, y4]]
            new_lines.append(f"{cls} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
        with open(label_path, 'w') as f:
            f.writelines(new_lines)

    def resize_image(img_path, new_size):
        img = cv2.imread(img_path)
        # first check if the image is already the correct size
        img = cv2.resize(img, new_size)
        cv2.imwrite(img_path, img)

    # Resize all images
    for img_path, label_path in tqdm(zip(image_paths, label_paths)):
        resize_image(img_path, new_size)
        resize_bbox(label_path, og_w, og_h, new_size)

def verify_resize(data_path, stats=False, plot=False):
    from pathlib import Path
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import random
    # Get all image file paths
    image_paths = list(Path(data_path).rglob("*.jpg"))
    label_paths = list(Path(data_path).rglob("*.txt"))
    n = len(image_paths)
    if stats:
        x1s, x2s, x3s, x4s = [], [], [], []
        y1s, y2s, y3s, y4s = [], [], [], []
    for img_path, label_path in tqdm(zip(image_paths, label_paths), total=n):
        
        img = cv2.imread(str(img_path))
        with open(label_path, 'r') as f:
            lines = f.readlines()
        if plot:
            for line in lines:
                cls, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.split())
                # Unnormalize the bounding box coordinates to the image scale
                x1, x2, x3, x4 = [x * img.shape[1] for x in [x1, x2, x3, x4]]
                y1, y2, y3, y4 = [y * img.shape[0] for y in [y1, y2, y3, y4]]
                x1, y1, x2, y2, x3, y3, x4, y4 = map(int, [x1, y1, x2, y2, x3, y3, x4, y4])
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), 2)
                cv2.line(img, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.line(img, (x4, y4), (x1, y1), (0, 255, 0), 2)
            plt.imshow(img)
            plt.show()
        elif stats:
            for line in lines:
                cls, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.split())
                # add the coordinates to the lists
                x1s.append(x1)
                x2s.append(x2)
                x3s.append(x3)
                x4s.append(x4)
                y1s.append(y1)
                y2s.append(y2)
                y3s.append(y3)
                y4s.append(y4)
    if stats:
        # plot the histograms of the coordinates
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs[0, 0].hist(x1s, bins=50)
        axs[0, 0].set_title('x1')
        axs[0, 1].hist(x2s, bins=50)
        axs[0, 1].set_title('x2')
        axs[0, 2].hist(x3s, bins=50)
        axs[0, 2].set_title('x3')
        axs[0, 3].hist(x4s, bins=50)
        axs[0, 3].set_title('x4')
        axs[1, 0].hist(y1s, bins=50)
        axs[1, 0].set_title('y1')
        axs[1, 1].hist(y2s, bins=50)
        axs[1, 1].set_title('y2')
        axs[1, 2].hist(y3s, bins=50)
        axs[1, 2].set_title('y3')
        axs[1, 3].hist(y4s, bins=50)
        axs[1, 3].set_title('y4')
        plt.show()


if __name__ == '__main__':
    data_path = 'D:/HCAI/msc/strawml/data/processed/yolo_format_bbox_straw_5fold/split_2'
    # resize_all_images_in_dir(data_path, new_size=(640, 640))
    verify_resize(data_path, stats=False, plot=True)