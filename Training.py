import cv2 as cv
import os
import numpy as np

artistes = ['Davido', 'Olamide', 'Tiwa Savage', 'Wizkid']
DIR = r'C:\Users\SHOPINVERSE\Desktop\cv_project\training_set'

haar_cascade = cv.CascadeClassifier('ha_faceclas.xml')

Features = []
Labels = []
def  create_training():
    for artiste in artistes:
        path = os.path.join(DIR, artiste)
        Label = artistes.index(artiste)
        
        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            image_array = cv.imread(image_path)
            gray_image = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
            face_detect= haar_cascade.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 4)
            
            for (x,y,w,h) in face_detect:
                faces_roi = gray_image[y:y+h, x:x+w]
                Features.append(faces_roi)
                Labels.append(Label)
create_training()
print("Training completed successfully!")
Features = np.array(Features, dtype='object')
Labels = np.array(Labels)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(Features, Labels)

face_recognizer.save("face_trained.yml")
np.save("features.npy", Features)
np.save("labels.npy", Labels)
                
            
            
    