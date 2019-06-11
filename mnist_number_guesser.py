from tkinter import *
from tkinter.colorchooser import askcolor
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
from tensorflow.keras.models import load_model
from matplotlib import image as pltImage
import matplotlib.pyplot as plt
import skimage.io as ski_io
from skimage.transform import resize
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import io


def load_MNIST_model(filepath):
    return load_model(filepath)

class MNISTPain(object):
    def __init__(self):
        self.prediction_txt = 'Prediction:         '
        self.root = Tk()
        self.CANVAS_WIDTH = 280
        self.CANVAS_HEIGHT = 280
        self.model = load_MNIST_model('mnist_cnn_model.h5')

        # Clear drawing button
        self.clear_button = Button(self.root, text='Clear', command=self.clear)
        self.clear_button.grid(row=0, column=0)

        # Predicted num
        self.predicted_num_label =  Label(self.root, text=self.prediction_txt, anchor='w')
        self.predicted_num_label.grid(row=0, column=1, columnspan=4,  sticky='nsew')

        self.canvas = Canvas(self.root, bg='white', width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT)
        self.canvas.grid(row=1, columnspan=5)
        self.setup()
        self.root.mainloop()

    def canvas_to_ndarray(self):
        # This is pretty bad... requires ghostscript and just hacked together
        try:
            os.remove('tmp_canvas.eps')
        except:
            pass
        self.canvas.postscript(file='tmp_canvas.eps',
                                colormode='color',
                                width=self.CANVAS_WIDTH,
                                height=self.CANVAS_HEIGHT,
                                pagewidth=self.CANVAS_WIDTH-1,
                                pageheight=self.CANVAS_HEIGHT-1)
        img = Image.open('tmp_canvas.eps').convert('L')
        img = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        arr = np.array(img)
        arr = np.array([(255 - x)/255. for x in arr])
        return arr


    def clear(self):
        self.canvas.delete('all')
        self.predicted_num_label['text'] = self.prediction_txt
        self.old_x = None
        self.old_y = None

    def reset(self, event):
        self.old_x = None
        self.old_y = None
        arr = self.canvas_to_ndarray()
        pred = self.model.predict(arr.reshape(-1, 28, 28, 1))
        prediction = np.argmax(pred, axis=1)[0]
        percent = np.sort(pred, kind='mergesort')[0,-1] * 100
        txt = '{} {} ({}%)'.format(self.prediction_txt.strip(), prediction, round(percent, 2))
        self.predicted_num_label['text'] = txt
        

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)


    def paint(self, event):
        self.line_width = 20
        paint_color = 'black'
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=self.line_width, fill=paint_color,
                                    capstyle=ROUND, smooth=TRUE, splinesteps=36)
        
        self.old_x = event.x
        self.old_y = event.y

if __name__ == "__main__":
    MNISTPain()