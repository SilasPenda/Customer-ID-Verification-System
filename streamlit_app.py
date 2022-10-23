import pandas as pd
import numpy as np
from PIL import Image
import cv2
import streamlit as st
from functions import *

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read classes
with open('classes.txt', 'r') as f:
    classes  = [line.strip() for line in f.readlines()]


st.title('Customer ID Verification Sysytem')
st.image('header.jpg')

img_file = st.file_uploader('Choose an Image', type=['jpg', 'png'])
if img_file is not None:
    img = Image.open(img_file).resize((250,250))
    st.image(img, use_column_width=False)
    save_image_path = './upload_images/'+img_file.name
    with open(save_image_path, 'wb') as f:
        f.write(img_file.getbuffer())

    image = read_image(save_image_path)

    net = cv2.dnn.readNet('model/yolov4-obj_last.weights', 'cfg/yolov4-obj.cfg')
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416,416), mean=(0,0,0), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer
    output_layers = net.forward(get_output_layers(net))

    Height = image.shape[0]
    Width = image.shape[1]

    # Go through all the bounding boxes output from the network and keep only the ones with high confidence score
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for output in output_layers:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([round(x), round(y), round(w), round(h)])
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    classes_ = []

    for ids in class_ids:
        class_ = classes[ids]
        classes_.append(class_)
        
    for indice in indices:
        class_indice = classes_[indice]
        
    # Crop the detected objects
    text = []

    for i in indices:
        indice = i
        box = boxes[i]
        (x, y, w, h) = box
        crop_img = image[y:y + h, x:x + w]
        
        # Save the cropped image
        cv2.imwrite('crop/' + classes_[indice] + '.jpg', crop_img)
        
        # Convert image to grayscale
        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # Resize the image to be three times the original size for better readability
        gray_img = cv2.resize(gray_img, (0,0), fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
        
        # Perform Gaussian Blur to smoothen image
        blur = cv2.GaussianBlur(gray_img, (5,5), 0)
        
        # Use Otsu's thresholding to turn write up white and background black
        ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)
        
        contours, hierachy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        sorted_contours = sorted(contours, key = lambda x: cv2.boundingRect(x)[0])
        
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
        roi = closing[y - 5:y + h + 5, x - 5:x + w + 5]
        # Use bitwise to flip image to black and background to white which is better for tesseract
        roi = cv2.bitwise_not(closing)

        # Save the Region of Interest (roi)
        cv2.imwrite('roi/' + classes_[indice] + '.jpg', roi)
        
        
    img = cv2.imread('roi/Passport_No.jpg')
    p_num = pytesseract.image_to_string(img, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/'' --psm 8 --oem 3').strip()


    def cust_rec(passport_number):
        df_data = pd.read_csv('database.csv')
        df = df_data[df_data['Passport_No'] == p_num]
        
        return df

if img_file is not None:
    result = cust_rec(p_num)
    print(result)
    st.write(result)