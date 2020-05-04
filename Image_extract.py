import cv2
import os

# to making the data directory
if not os.path.exists('data'):
    os.makedirs('data')

# to write multiple image in a data folder
def generate_data(img, id, img_id):
    cv2.imwrite("data/Dinesh_sir"+str(id)+"."+str(img_id)+".jpg",img)

# to Draw a rectangle over a deteted face
def draw_boundary(img,classifier, scaleFactor, minNeighbours, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords= []
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x,y,w,h]
    
    return coords

def detect(img,faceCascade,img_id):
    color = {"blue" : (255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img,faceCascade,1.1,10,color["blue"],"You were Looking Good")

    if(len(coords)==4):
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]

        #for First user I am setting user id as 1 
        user_id = 2
        generate_data(roi_img, user_id, img_id)

    return img

# inserting the haar cascade file for front face
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Starting the camera
video_capture = cv2.VideoCapture(-1)

img_id = 0

while True:
    _, img = video_capture.read()
    img = detect(img, faceCascade, img_id)
    cv2.imshow("Face Capture",img)
    img_id += 1
    if cv2.waitKey(1) & 0xff == ord('\r') or img_id == 200:
        break

video_capture.release()
cv2.destroyAllWindows()


