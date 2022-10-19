# Customer-ID-Verification-System
This is a system that takes in customer international passport as ID, extracts information from it and verifies it from a record.

# Guide
* Gather data set of Nigerian passports by scraping from different news blogs.
  
* Annotate images in Yolo format. Classes: Passport Type, Passport Number, Surname, Other Names, Nationality, DOB, Sex, Location.

* Train a YOLO v4 object Detection model using the dataset and an accuracy of 86% was achieved (Can do better if trained longer).

* Use the trained model to predict the bounding boxes which is extracted using Open CV.

* Pass image through processing steps like grayscale, Gaussian blur, Otsu's thresholding, etc

* Extract information particularly Passport Number from the image using Tesseract -OCR.

* Checks database for details of the owner of the ID. 
