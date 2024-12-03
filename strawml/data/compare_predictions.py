from __init__ import *
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import datetime
import h5py
import os
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
            image_list = sorted(image_list, key=lambda x: int(x.split('_')[1]))
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
        else:
            pass
            # TODO: Load the image from the data folder

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (0, 0), fx=self.image_scale, fy=self.image_scale)
        self.image = image
        self.image_size = self.image.shape[:2]
        print(f"Set image with shape {self.image.shape}")
        
    def display_image(self, image: np.ndarray) -> None:
        self.canvas.delete('all')
        
        self.canvas.focus_set()
        self.canvas.bind('<Key>', self.key_pressed)
        
        # cv2.imshow('image', image)
        
        # Show image
        self.displayed_image = ImageTk.PhotoImage(Image.fromarray(image))
        print("Displayed image")
        self.canvas.create_image(self.image_size[1]/2, self.image_size[0]/2, image=self.displayed_image)
        self.canvas.create_text(10, 10, text=f"Image {self.current_image_index+1}/{len(self.image_list)}", anchor='nw', fill='white')
        
    def key_pressed(self, event) -> None:
        print(f"Key pressed: {event.char}")
        if event.char == 'a':
            print('ML was better!')
        elif event.char == 'b':
            print("Sensor was better!")
        else:
            return
        
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_list):
            print("No more images to show.")
            return
        self.set_image(self.current_image_index)
        self.display_image(self.image)
    


def get_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/processed/sensors.hdf5', help='Path to the data file')
    parser.add_argument('--data_folder', type=str, default='', help='Path to the data folder')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    root = tk.Tk()
    root.resizable(False, False)
    MainFrame(root, args.data_path, args.data_folder).pack(side="top", fill="both", expand=True)
    root.mainloop()
    
    