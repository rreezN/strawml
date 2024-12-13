from __init__ import *
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import datetime
import h5py
import os
import random
from argparse import ArgumentParser

from strawml.data.make_dataset import decode_binary_image

class MainFrame(ttk.Frame):
    def __init__(self, parent, data_path, data_folder, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.data_path = data_path
        self.data_folder = data_folder
        self.parent.title('Compare Predictions')
        self.current_image_index = 0
        self.image_scale = 0.5
        self.image_list = []
        self.canvas = None
        self.displayed_image = None
        self.scada_line = None
        self.scada_percent = None
        self.yolo_line = None
        self.yolo_percent = None
        self.id = os.environ.get('USERNAME')
        self.already_ranked = False
        
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta']
        random.shuffle(colors)
        
        self.scada_color = colors.pop(0)
        self.yolo_color = colors.pop(0)
        
        # self.parent.bind('<Key>', self.key_pressed)
        
        if data_folder == '':
            self.load_image_list_from_hdf5()
        else:
            self.load_image_list_from_folder()
        
        self.set_image(self.current_image_index)
        self.canvas = tk.Canvas(self, cursor='cross', width=self.image_size[1], height=self.image_size[0])
        self.canvas.pack(side='top', fill='both', expand=True)
        self.display_image(self.image)
    
    def load_image_list_from_hdf5(self) -> None:
        """Loads the image list from the images HDF5 file and sorts it.

        Raises:
            FileNotFoundError: If the images HDF5 file does not exist.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"The file {self.data_path} does not exist.")
        
        with h5py.File(self.data_path, 'r') as hf:
            image_list = list(hf.keys())
            image_list = sorted(image_list)
            self.image_list = image_list
    
    def load_image_list_from_folder(self) -> None:
        """Loads the image list from the data folder and sorts it."""
        image_list = os.listdir(self.data_folder)
        if len(image_list) == 0:
            raise FileNotFoundError(f"The folder {self.data_folder} does not contain any images.")
        # TODO: Change the sorting to fit the data folder
        image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))
        self.image_list = image_list
    
    def set_image(self, new_image_index: int = 0) -> None:
        if self.canvas != None: self.canvas.delete('all')
        
        self.current_image_index = new_image_index
        
        image = None
        if self.data_folder == '':
            with h5py.File(self.data_path, 'r') as hf:
                image = hf[self.image_list[self.current_image_index]]['image'][:]
                image = decode_binary_image(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if 'scada' in hf[self.image_list[self.current_image_index]].keys():
                    scada_group = hf[self.image_list[self.current_image_index]]['scada']
                    self.scada_percent = scada_group['percent'][...].item()
                    self.scada_line = scada_group['pixel'][...]
                else:
                    self.scada_percent = None
                    self.scada_line = None
                    
                if 'yolo' in hf[self.image_list[self.current_image_index]].keys():
                    yolo_group = hf[self.image_list[self.current_image_index]]['yolo']
                    self.yolo_percent = yolo_group['percent'][...].item()
                    self.yolo_line = yolo_group['pixel'][...]
                else:
                    self.yolo_percent = None
                    self.yolo_line = None
                
                if 'rankings' in hf[self.image_list[self.current_image_index]].keys():
                    rankings_group = hf[self.image_list[self.current_image_index]]['rankings']
                    # print(f'Image: {self.current_image_index}, Keys: {rankings_group.keys()}')
                    self.already_ranked = self.id in rankings_group.keys()
                else:
                    self.already_ranked = False
        else:
            raise NotImplementedError("Loading images from a folder is not implemented yet.")
            pass

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (0, 0), fx=self.image_scale, fy=self.image_scale)
        self.image = image
        self.image_size = self.image.shape[:2]
        # print(f"Set image with shape {self.image.shape}")
        
    def display_image(self, image: np.ndarray) -> None:
        self.canvas.delete('all')
        
        self.canvas.focus_set()
        self.canvas.bind('<Key>', self.key_pressed)
        self.canvas.bind('<Right>', self.right_pressed)
        self.canvas.bind('<Left>', self.left_pressed)
        
        # cv2.imshow('image', image)
        
        # Show image
        self.displayed_image = ImageTk.PhotoImage(Image.fromarray(image))
        # print("Displayed image")
        scada_coords = (int(35+self.scada_line[0]*self.image_scale), int(self.scada_line[1]*self.image_scale))
        yolo_coords = (int(35+self.yolo_line[0]*self.image_scale), int(self.yolo_line[1]*self.image_scale))
        self.canvas.create_image(self.image_size[1]/2, self.image_size[0]/2, image=self.displayed_image)
        self.canvas.create_line(scada_coords[0], scada_coords[1], scada_coords[0]+100, scada_coords[1], fill=self.scada_color, width=2)
        self.canvas.create_text(scada_coords[0]+100, scada_coords[1], text=f"{self.scada_percent:.2f}%", anchor='w', fill=self.scada_color, font=('Arial', 16, 'bold'))
        self.canvas.create_line(yolo_coords[0], yolo_coords[1], yolo_coords[0]+100, yolo_coords[1], fill=self.yolo_color, width=2)
        self.canvas.create_text(yolo_coords[0]+100, yolo_coords[1], text=f"{self.yolo_percent:.2f}%", anchor='w', fill=self.yolo_color, font=('Arial', 16, 'bold'))
        
        self.canvas.create_text(10, 10, text=f'Press "a" for {self.scada_color}', anchor='nw', fill=self.scada_color, font=('Arial', 16, 'bold'))
        self.canvas.create_text(10, 30, text=f'Press "b" for {self.yolo_color}', anchor='nw', fill=self.yolo_color, font=('Arial', 16, 'bold'))
        
        if self.already_ranked:
            self.canvas.create_text(10, 50, text='Already ranked!', anchor='nw', fill='yellow', font=('Arial', 24, 'bold'))
        
        self.canvas.create_text(self.image_size[1]-10, self.image_size[0]-10, text=f"Image {self.current_image_index+1}/{len(self.image_list)}", anchor='se', 
                                fill='white', font=('Arial', 16, 'bold'))
        
    def key_pressed(self, event) -> None:
        # print(f"Key pressed: {event.char}")
        winner = None
        if event.char == 'a':
            # print(f'{self.scada_color} (SCADA) was better!')
            self.current_image_index += 1
            winner = 'scada'
        elif event.char == 'b':
            # print(f'{self.yolo_color} (YOLO) was better!')
            self.current_image_index += 1
            winner = 'yolo'
        else:
            print("Invalid key pressed. Press A or B")
            return
        
        
        with h5py.File(self.data_path, 'a') as hf:
            if 'rankings' not in hf[self.image_list[self.current_image_index-1]].keys():
                hf[self.image_list[self.current_image_index-1]].create_group('rankings')
            rankings_group = hf[self.image_list[self.current_image_index-1]]['rankings']
            if self.id in rankings_group.keys():
                del rankings_group[self.id]
            rankings_group[self.id] = winner
                
        print(f'Saved ranking for image {self.current_image_index-1}')
        # print(f'Winner: {winner}')
            
        if self.current_image_index >= len(self.image_list):
            self.current_image_index = len(self.image_list) - 1
            print("No more images to show.")
            return
        self.set_image(self.current_image_index)
        self.display_image(self.image)
    
    def right_pressed(self, event) -> None:
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_list):
            self.current_image_index = len(self.image_list) - 1
            print("No more images to show.")
            return
        self.set_image(self.current_image_index)
        self.display_image(self.image)
    
    def left_pressed(self, event) -> None:
        self.current_image_index -= 1
        if self.current_image_index < 0:
            print("No more images to show.")
            self.current_image_index = 0
            return
        self.set_image(self.current_image_index)
        self.display_image(self.image)


def get_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/processed/recording.hdf5', help='Path to the data file')
    parser.add_argument('--data_folder', type=str, default='', help='Path to the data folder')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    root = tk.Tk()
    root.resizable(False, False)
    MainFrame(root, args.data_path, args.data_folder).pack(side="top", fill="both", expand=True)
    root.mainloop()
    
    