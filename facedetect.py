import cv2 as cv
img = cv.imread('Images/Sch.jpg')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
haar_cascade = cv.CascadeClassifier('ha_faceclas.xml')
face_detect = haar_cascade.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 3)
for (x, y, w, h) in face_detect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0,255,225), 2)
cv.imshow("Detected Faces", img)
print("Number of faces detected is {}".format(len(face_detect)))
cv.waitKey(0)
    
