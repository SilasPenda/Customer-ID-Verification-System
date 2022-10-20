import pandas as pd
import numpy as np
from PIL import Image
import easyocr
import cv2
import matplotlib.pyplot as plt
import os

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Set up YOLO Layers
def get_output_layers(net):
    # Get the names of the layers in the network
    layer_names = net.getLayerNames()
    
    # Get the names of the output layers i.e. the unconnected layers
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Read Image
def read_image(image_path):
    image = cv2.imread(image_path)
    return image


