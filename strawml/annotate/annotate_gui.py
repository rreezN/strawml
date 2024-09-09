# Code inspired from: https://github.com/Arka-Bhowmik/bounding_box_gui/tree/main


from __init__ import *

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import datetime
import h5py
import os

from strawml.data.make_dataset import decode_binary_image



class ImageBox(ttk.Frame):
    def __init__(self, parent, images_hdf5='data/raw/images/images.hdf5', *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        self.images_hdf5 = images_hdf5
        self.annotated_hdf5 = 'data/processed/annotated_images.hdf5'
        self.image = None
        self.image_size = None
        self.current_image_group = None
        self.chute_annotated = False
        self.straw_annotated = False
        self.rect = None
        self.start_x = None
        self.start_y = None
        
        self.rect2 = None
        self.start_x2 = None
        self.start_y2 = None
        
        
        self.set_image()
        
        self.canvas = tk.Canvas(self, cursor="cross", width=self.image_size[1], height=self.image_size[0])
        self.canvas.pack(side="top", fill="both", expand=True)

        self.display_image(self.image)
    
    def set_image(self, image_group=None):
        # TODO: rewrite to load frames from both files into a list, and then check from there....
        # Load image from raw images HDF5 file
        with h5py.File(self.images_hdf5, 'r') as hf:
            if image_group is None:
                frame = list(hf.keys())[0]
            else:
                frame = image_group
            self.current_image_group = frame
            image_bytes = hf[frame]['image'][...]
        
        # If the image already exists in annotated, then we load it instead (overwrite)
        with h5py.File(self.annotated_hdf5, 'r') as hf:
            if image_group is None:
                frame = list(hf.keys())[0]
            else:
                frame = image_group
            if frame in hf.keys():
                self.current_image_group = frame
                image_bytes = hf[frame]['image'][...]
                
                if 'annotations' in hf[frame].keys():
                    # load bboxes
                    if 'bbox_chute' in hf[frame]['annotations'].keys():
                        self.chute_annotated = True
                        self.start_x = hf[frame]['annotations']['bbox_chute'][...][0]//2
                        self.start_y = hf[frame]['annotations']['bbox_chute'][...][1]//2
                        self.curX = hf[frame]['annotations']['bbox_chute'][...][2]//2
                        self.curY = hf[frame]['annotations']['bbox_chute'][...][3]//2
                    if 'bbox_straw' in hf[frame]['annotations'].keys():
                        self.straw_annotated = True
                        self.start_x2 = hf[frame]['annotations']['bbox_straw'][...][0]//2
                        self.start_y2 = hf[frame]['annotations']['bbox_straw'][...][1]//2
                        self.curX2 = hf[frame]['annotations']['bbox_straw'][...][2]//2
                        self.curY2 = hf[frame]['annotations']['bbox_straw'][...][3]//2
                    
                    # load fullness and obstructed
                    if 'fullness' in hf[frame]['annotations'].keys():
                        self.parent.fullness_box.full_amount.set(hf[frame]['annotations']['fullness'][...])
                    if 'obstructed' in hf[frame]['annotations'].keys():
                        self.parent.obstructed_box.obstructed.set(str(hf[frame]['annotations']['obstructed'][...]))  

        image = decode_binary_image(image_bytes)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (0, 0), fx=self.parent.image_scale, fy=self.parent.image_scale)
        self.image = image
        self.image_size = self.image.shape[:2]
    
    def display_image(self, image):
        # TODO: Display annotations on the image if loaded
        self.canvas.delete('all')

        # Define events for canvas mouse clicks
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        # Show image
        self.new_image = ImageTk.PhotoImage(Image.fromarray(image))
        self.canvas.create_image(self.image_size[1]/2, self.image_size[0]/2, image=self.new_image)
        
        # Initialize bounding box parameters
        if not self.chute_annotated:
            self.rect = None
            self.start_x = None
            self.start_y = None
        else:
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.curX, self.curY, outline='green', width=2)
            self.current_rect = self.rect
            
        if not self.straw_annotated:
            self.rect2 = None
            self.start_x2 = None
            self.start_y2 = None
        else:
            self.rect2 = self.canvas.create_rectangle(self.start_x2, self.start_y2, self.curX2, self.curY2, outline='red', width=2)
            self.current_rect = self.rect2
        
    
    def on_button_press(self, event):
        if not self.rect:
            self.start_x = event.x
            self.start_y = event.y
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='green', width=2)
            self.current_rect = self.rect
        elif not self.rect2:
            self.start_x2 = event.x
            self.start_y2 = event.y
            self.rect2 = self.canvas.create_rectangle(self.start_x2, self.start_y2, self.start_x2, self.start_y2, outline='red', width=2)
            self.current_rect = self.rect2
        else:
            self.start_x2 = event.x
            self.start_y2 = event.y
            
    def on_move_press(self, event):
        if not self.rect2:
            self.curX = self.canvas.canvasx(event.x)
            self.curY = self.canvas.canvasy(event.y)
        elif self.rect:
            self.curX2 = self.canvas.canvasx(event.x)
            self.curY2 = self.canvas.canvasy(event.y)
        
        if self.current_rect == self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, self.curX, self.curY)
        elif self.current_rect == self.rect2:
            self.canvas.coords(self.rect2, self.start_x2, self.start_y2, self.curX2, self.curY2)
        
    
    def on_button_release(self, event):
        if self.start_x < 0:
            self.start_x = 0
        elif self.start_x > self.image_size[1]:
            self.start_x = self.image_size[1]
        
        if self.start_y < 5:
            self.start_y = 5
        elif self.start_y > self.image_size[0]:
            self.start_y = self.image_size[0]

        if self.curX < 0:
            self.curX = 0
        elif self.curX > self.image_size[1]:
            self.curX = self.image_size[1]
        
        if self.curY < 5:
            self.curY = 5
        elif self.curY > self.image_size[0]:
            self.curY = self.image_size[0]
        
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.curX, self.curY)
    
    def reset(self):
        self.rect = None
        self.rect2 = None
        self.canvas.delete('all')
        self.set_image(self.current_image_group)
        self.display_image(self.image)
    
    
    

class FullnessBox(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        self.full_amount = tk.DoubleVar(value=-1)
        
        empty = ttk.Radiobutton(self, text='0%', value=0, variable=self.full_amount)
        five = ttk.Radiobutton(self, text='5%', value=0.05, variable=self.full_amount)
        ten = ttk.Radiobutton(self, text='10%', value=0.10, variable=self.full_amount)
        fifteen = ttk.Radiobutton(self, text='15%', value=0.15, variable=self.full_amount)
        twenty = ttk.Radiobutton(self, text='20%', value=0.20, variable=self.full_amount)
        twenty_five = ttk.Radiobutton(self, text='25%', value=0.25, variable=self.full_amount)
        thirty = ttk.Radiobutton(self, text='30%', value=0.30, variable=self.full_amount)
        thirty_five = ttk.Radiobutton(self, text='35%', value=0.35, variable=self.full_amount)
        forty = ttk.Radiobutton(self, text='40%', value=0.40, variable=self.full_amount)
        forty_five = ttk.Radiobutton(self, text='45%', value=0.45, variable=self.full_amount)
        fifty = ttk.Radiobutton(self, text='50%', value=0.50, variable=self.full_amount)
        fifty_five = ttk.Radiobutton(self, text='55%', value=0.55, variable=self.full_amount)
        sixty = ttk.Radiobutton(self, text='60%', value=0.60, variable=self.full_amount)
        sixty_five = ttk.Radiobutton(self, text='65%', value=0.65, variable=self.full_amount)
        seventy = ttk.Radiobutton(self, text='70%', value=0.70, variable=self.full_amount)
        seventy_five = ttk.Radiobutton(self, text='75%', value=0.75, variable=self.full_amount)
        eighty = ttk.Radiobutton(self, text='80%', value=0.80, variable=self.full_amount)
        eighty_five = ttk.Radiobutton(self, text='85%', value=0.85, variable=self.full_amount)
        ninety = ttk.Radiobutton(self, text='90%', value=0.90, variable=self.full_amount)
        ninety_five = ttk.Radiobutton(self, text='95%', value=0.95, variable=self.full_amount)
        full = ttk.Radiobutton(self, text='100%', value=1.0, variable=self.full_amount)
        
        full.pack()
        ninety_five.pack()
        ninety.pack()
        eighty_five.pack()
        eighty.pack()
        seventy_five.pack()
        seventy.pack()
        sixty_five.pack()
        sixty.pack()
        fifty_five.pack()
        fifty.pack()
        forty_five.pack()
        forty.pack()
        thirty_five.pack()
        thirty.pack()
        twenty_five.pack()
        twenty.pack()
        fifteen.pack()
        ten.pack()
        five.pack()
        empty.pack()

class ObstructedBox(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        self.obstructed = tk.BooleanVar()
        
        checkbox = ttk.Checkbutton(self, text='Obstructed', variable=self.obstructed)
        
        checkbox.pack()


class MainApplication(ttk.Frame):
    def __init__(self, parent, images_hdf5='data/raw/images/images.hdf5', *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        
        self.parent.title("AnnotateGUI")

        self.current_image = 0
        self.image_scale = 0.5
        self.images_hdf5 = images_hdf5
        
        self.fullness_box = FullnessBox(self)
        self.obstructed_box = ObstructedBox(self)
        self.image_box = ImageBox(self)
        
        self.reset_button = ttk.Button(self, text="Reset bboxes", command=self.reset)
        self.back_button = ttk.Button(self, text="Back", command=self.back)
        self.select_image_button = ttk.Button(self, text="Select Image", command=self.select_image)
        self.next_button = ttk.Button(self, text="Next", command=self.next)
        self.progress_label = ttk.Label(self, text="0/5000000")
        
        
        self.progress_bar = ttk.Progressbar(self, orient = 'horizontal', length=150, mode='determinate')
        self.progress_bar['value'] = 50
        
        self.image_box.grid(row=0, column=0, sticky="NW", padx=5, pady=5, rowspan=4, columnspan=4)
        self.reset_button.grid(row=0, column=4, stick = "W", padx=5, pady=5)
        self.fullness_box.grid(row=1, column=4, sticky="W", padx=5, pady=5)
        self.obstructed_box.grid(row=2, column=4, sticky="W", padx=5, pady=5)
        self.back_button.grid(row=4, column=0, sticky='NW', padx=5, pady=(0, 5))
        self.select_image_button.grid(row=4, column=1, sticky="NW", padx=5, pady=(0, 5))
        self.progress_bar.grid(row=4, column=2, sticky='NW', padx=5, pady=(0, 5))
        self.progress_label.grid(row=4, column=3, sticky="W", padx=5, pady=5)
        self.next_button.grid(row=4, column=4, sticky="W", padx=5, pady=(0, 5))
        
        self.load_image_list()

        
    def load_image_list(self):
        if not os.path.exists(self.images_hdf5):
            raise FileNotFoundError(f"The file {self.images_hdf5} does not exist.")
        
        with h5py.File(self.images_hdf5, 'r') as hf:
            image_list = list(hf.keys())
            image_list = self.sort_image_list(image_list)
  
        self.image_list = image_list
        self.update_progress_bar()

    def sort_image_list(self, image_list):
        return sorted(image_list, key=lambda x: int(x.split('_')[1]))
    
    def update_progress_bar(self):
        self.progress_bar['value'] = self.current_image/len(self.image_list)*100
        self.progress_label['text'] = f"{self.current_image+1}/{len(self.image_list)}"
    
    def next(self):
        self.save_current_frame(printing=False)
        
        if self.current_image == len(self.image_list)-1:
            self.change_image(0)
        else:
            self.change_image(self.current_image+1)
    
    def reset(self):
        self.image_box.reset()
        
    def back(self):
        self.save_current_frame(printing=False)
        
        if self.current_image == 0:
            self.change_image(len(self.image_list)-1)
        else:
            self.change_image(self.current_image-1)

    # TODO: Implement select image functionality
    def select_image(self):
        pass
    
    def change_image(self, new_image_index):
        self.fullness_box.full_amount.set(-1)
        self.obstructed_box.obstructed.set(False)
        self.image_box.set_image(image_group=self.image_list[new_image_index])
        self.image_box.display_image(self.image_box.image)
        self.current_image = new_image_index
        self.update_progress_bar()
    
    def save_current_frame(self, 
             new_hdf5_file='data/processed/annotated_images.hdf5',
             printing=False):
        
        new_hf = h5py.File(new_hdf5_file, 'w') # Open the HDF5 file in write mode
        old_hf = h5py.File(self.images_hdf5, 'r') # Open the original HDF5 file in read mode
        
        # Copy the attributes from the original HDF5 file to the new HDF5 file
        new_hf.attrs['dataset_name'] = old_hf.attrs['dataset_name']
        new_hf.attrs['description'] = old_hf.attrs['description']
        new_hf.attrs['date_created'] = np.bytes_(str(datetime.datetime.now()))
        
        
        # Create a new group for the current frame
        frame = f'frame_{self.current_image}'
        
        # Overwrite the frame if it already exists
        if frame in new_hf.keys():
            del new_hf[frame]
        
        group = new_hf.create_group(frame)
        
        # Copy the original image and image_diff to the new HDF5 file
        old_hf.copy(old_hf[frame]['image'], group)
        old_hf.copy(old_hf[frame]['image_diff'], group) 
        group.attrs['video ID'] = old_hf[frame].attrs['video ID']
        
        # Create a new group for the annotations
        annotation_group = group.create_group('annotations')
        
        # Save the bounding box coordinates (scaled to correct size)
        img_box = self.image_box
        if img_box.start_x != None:
            chute_bbox = [img_box.start_x*2, img_box.start_y*2, img_box.curX*2, img_box.curY*2]
            annotation_group.create_dataset('bbox_chute', data=chute_bbox)
        if img_box.start_x2 != None:
            straw_bbox = [img_box.start_x2*2, img_box.start_y2*2, img_box.curX2*2, img_box.curY2*2]
            annotation_group.create_dataset('bbox_straw', data=straw_bbox)
        
        # Save the fullness and obstructed values
        fullness = self.fullness_box.full_amount.get()
        obstructed = self.obstructed_box.obstructed.get()
        if fullness > 0: # If the fullness is not set, don't save it
            annotation_group.create_dataset('fullness', data=fullness)
        annotation_group.create_dataset('obstructed', data=obstructed)
        
        if printing:
            print(f'Saved annotations for frame {frame}')
            print(new_hf.keys())
            print(new_hf.attrs.keys())
            print(new_hf[frame].keys())
            print(new_hf[frame].attrs.keys())
            print(new_hf[frame]['annotations'].keys())
            if 'bbox_chute' in new_hf[frame]['annotations'].keys():
                print('bbox_chute:')
                print(new_hf[frame]['annotations']['bbox_chute'][...])
            if 'bbox_straw' in new_hf[frame]['annotations'].keys():
                print('bbox_straw:')
                print(new_hf[frame]['annotations']['bbox_straw'][...])
            if 'fullness' in new_hf[frame]['annotations'].keys():
                print('fullness:')
                print(new_hf[frame]['annotations']['fullness'][...])
            if 'obstructed' in new_hf[frame]['annotations'].keys():
                print('obstructed:')
                print(new_hf[frame]['annotations']['obstructed'][...])

        old_hf.close()  # close the original hdf5 file
        new_hf.close()  # close the hdf5 file
        
if __name__ == '__main__':
    root = tk.Tk()
    root.resizable(False, False)
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
    
