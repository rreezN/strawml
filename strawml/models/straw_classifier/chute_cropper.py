from PIL import Image
import numpy as np
import torch




def rotate_to_bbox(image: np.array|torch.Tensor, bbox: list) -> tuple:
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox

    # Find the rotation angle of the bounding box
    dy = y4 - y2
    dx = x4 - x2
    m = dy/dx
    angle = np.arctan(m)
    d = angle - 45
    
    

if __name__ == '__main__':
    # Load the image
    image = Image.open('data/raw/chute_images/IMG_0001.jpg')
    image = np.array(image)
    
    # Load the bounding box
    bbox = [0, 0, 0, 0, 0, 0, 0, 0]
    
    # Rotate the image to the bounding box
    rotated_image, rotated_bbox = rotate_to_bbox(image, bbox)
    
    # Save the rotated image
    rotated_image = Image.fromarray(rotated_image)
    rotated_image.save('data/processed/rotated_image.jpg')
    
    print('Image rotated successfully!')



