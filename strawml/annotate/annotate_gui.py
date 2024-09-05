# Code inspired from: https://github.com/Arka-Bhowmik/bounding_box_gui/tree/main



import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np



class ImageBox(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        # Load image
        img = cv2.imread('data/raw/images/frame_test.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.image_size = img.shape[:2]
        
        self.canvas = tk.Canvas(self, cursor="cross", width=self.image_size[1], height=self.image_size[0])
        self.canvas.pack(side="top", fill="both", expand=True)

        self.display_image(img)
        
    def display_image(self, image=None):
        self.canvas.delete('all')
        if image is None:
            image = self.edited_image.copy()
        else:
            image = image

        # Define events for canvas mouse clicks
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        # Initialize bounding box parameters
        self.rect = None
        self.start_x = None
        self.start_y = None
        
        self.rect2 = None
        self.start_x2 = None
        self.start_y2 = None
        
        self.new_image = ImageTk.PhotoImage(Image.fromarray(image))
        self.canvas.create_image(400/2, 525/2, image=self.new_image)
    
    def on_button_press(self, event):
        if not self.rect:
            self.start_x = event.x
            self.start_y = event.y
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='green', width=2)
            self.current_rect = self.rect
        else:
            self.start_x2 = event.x
            self.start_y2 = event.y
            self.rect2 = self.canvas.create_rectangle(self.start_x2, self.start_y2, self.start_x2, self.start_y2, outline='red', width=2)
            self.current_rect = self.rect2

    def on_move_press(self, event):
        if not self.rect2:
            self.curX = self.canvas.canvasx(event.x)
            self.curY = self.canvas.canvasy(event.y)
        elif self.rect:
            self.curX2 = self.canvas.canvasx(event.x)
            self.curY2 = self.canvas.canvasy(event.y)
        
        # TODO: Fix second rectangle not getting overwritten
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
        self.save_bbox()
            
    
    def save_bbox(self):
        path = 'data/processed'
        np.savetxt(f'{path}/bbox.txt', [self.start_x, self.start_y, self.curX, self.curY], delimiter=',')
    
    

class FullnessBox(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        self.full_amount = tk.DoubleVar()
        
        empty = ttk.Radiobutton(self, text='0%', value=0)
        five = ttk.Radiobutton(self, text='5%', value=0.05)
        ten = ttk.Radiobutton(self, text='10%', value=0.10)
        fifteen = ttk.Radiobutton(self, text='15%', value=0.15)
        twenty = ttk.Radiobutton(self, text='20%', value=0.20)
        twenty_five = ttk.Radiobutton(self, text='25%', value=0.25)
        thirty = ttk.Radiobutton(self, text='30%', value=0.30)
        thirty_five = ttk.Radiobutton(self, text='35%', value=0.35)
        forty = ttk.Radiobutton(self, text='40%', value=0.40)
        forty_five = ttk.Radiobutton(self, text='45%', value=0.45)
        fifty = ttk.Radiobutton(self, text='50%', value=0.50)
        fifty_five = ttk.Radiobutton(self, text='55%', value=0.55)
        sixty = ttk.Radiobutton(self, text='60%', value=0.60)
        sixty_five = ttk.Radiobutton(self, text='65%', value=0.65)
        seventy = ttk.Radiobutton(self, text='70%', value=0.70)
        seventy_five = ttk.Radiobutton(self, text='75%', value=0.75)
        eighty = ttk.Radiobutton(self, text='80%', value=0.80)
        eighty_five = ttk.Radiobutton(self, text='85%', value=0.85)
        ninety = ttk.Radiobutton(self, text='90%', value=0.90)
        ninety_five = ttk.Radiobutton(self, text='95%', value=0.95)
        full = ttk.Radiobutton(self, text='100%', value=1.0)
        
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
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent


        self.image_box = ImageBox(self)
        self.fullness_box = FullnessBox(self)
        self.obstructed_box = ObstructedBox(self)
        
        self.next_button = ttk.Button(self, text="Next", command=self.next)
        
        
        self.image_box.grid(row=0, column=0, sticky="NW", padx=5, pady=5, rowspan=3)
        self.fullness_box.grid(row=0, column=1, sticky="W", padx=5, pady=5)
        self.obstructed_box.grid(row=1, column=1, sticky="W", padx=5, pady=5)
        self.next_button.grid(row=2, column=1, sticky="W", padx=5, pady=5)
        

    def next(self):
        pass

if __name__ == '__main__':
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
    
