import cv2
import numpy as np
import os

drawing = False
ix, iy = -1, -1

def load_image(image_path):
    image = cv2.imread(image_path)
    return image


def display_image(image, title='Image'):
    global draw_img, tmp_img
    draw_img = image.copy()
    tmp_img = image.copy()
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, click_event)
    image_copy = image.copy()
    while True:
        if drawing == False:
            cv2.imshow(title, draw_img)
        
        # Reset image by pressing R
        if cv2.waitKey(1) & 0xFF == ord('r'):
            draw_img = image.copy()
            tmp_img = image.copy()
        
        # Quit by pressing escape
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    return


def click_event(event, x, y, flags, params):
    global drawing, ix, iy, tmp_img, draw_img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
           tmp_img = draw_img.copy()
           cv2.rectangle(tmp_img, (ix, iy), (x, y), (0, 255, 0), 2)
           cv2.imshow("Image", tmp_img)
    elif event == cv2.EVENT_LBUTTONUP:
        draw_img = cv2.rectangle(draw_img, (ix, iy), (x, y), (0, 255, 0), 2)
        drawing = False
        save_bbox(ix, iy, x, y)
    return


def save_bbox(ix, iy, x, y):
    path = 'data/processed'
    np.savetxt(f'{path}/bbox.txt', [ix, iy, x, y], delimiter=',')



if __name__ == '__main__':
    path = 'data/raw/images/frame_test.png'
    image = load_image(path)
    display_image(image)




