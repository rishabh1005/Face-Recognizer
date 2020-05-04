import cv2
import os


# to Draw a rectangle over a deteted face
def draw_boundary(img,classifier, scaleFactor, minNeighbours, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords= []
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        id ,_ = clf.predict(gray_img[y:y+h,x:x+w])
        if id==1:
            cv2.putText(img, "Rohit", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        elif id==2:
            cv2.putText(img, "dinesh", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        elif id==2:
            cv2.putText(img, "Poonam Maam", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x,y,w,h]
    
    return coords

def recognize(img, clf, faceCascade):
    color = {"blue" : (255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img,faceCascade,1.1,10,color["blue"],"You were Looking Good",clf)
    return img


# inserting the haar cascade file for front face
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")
#Starting the camera
video_capture = cv2.VideoCapture(0)

img_id = 0

while True:
    _, img = video_capture.read()
    img = recognize(img, clf, faceCascade)
    cv2.imshow("Face Capture",img)
    img_id += 1
    if cv2.waitKey(1) & 0xff == ord('\r') :
        break

video_capture.release()
cv2.destroyAllWindows()


