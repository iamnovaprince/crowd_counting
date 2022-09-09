import numpy as np
import cv2 as cv

CLASS_FILE = "D:\Production\Projects\ML\yolo\darknet\data\coco.names"
WEIGHT_FILE = "D:\Production\Projects\ML\yolo\yolov3.weights"
CONFIG_FILE = "D:\Production\Projects\ML\yolo\darknet\cfg\yolov3.cfg"
CONFIDENCE_THRESOLD = 0.5
IMAGE_FILE = "D:\Wallpaper\object-detection.jpg"
LABELS = open(CLASS_FILE).read().strip().splitlines()

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
yolo = cv.dnn.readNet(WEIGHT_FILE, CONFIG_FILE)

img = cv.imread(IMAGE_FILE)
# img = img.reshape([512,512],3)
img = cv.resize(img, (512, 512),3)
(H,W) = img.shape[:2]

# blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
# yolo.setInput(blob)

boxes = []
confidences = []
classIDs = []


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def detectionFilter(output,H,W):
    # confidences = []
    # classIDs = []
    # boxes = []
    for out in output:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE_THRESOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    return (confidence,classIDs,boxes)





def countPeople():
    idxs = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESOLD,CONFIDENCE_THRESOLD)
    counter = 0
    if len(idxs) > 0:
        for i in idxs.flatten():
            if "person" in LABELS[classIDs[i]]:
                counter+=1
           
    return counter
                
# # detectionFilter()
# # countPeople()

# # cv.imshow("img",img)
# # cv.waitKey(10000)

# vid = cv.VideoCapture(0)

# while True:
#     res,frame = vid.read()
#     cv.imshow("img",frame)

vid = cv.VideoCapture(1)
last = 0  
while(True):
      
    ret, frame = vid.read()
  
    # (H,W) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo.setInput(blob)
    output = yolo.forward(get_output_layers(yolo))
    
    confidences, classIDs, boxes = detectionFilter(output,H,W)
    people = countPeople()
    if people != last:
        last = people
        print(people)
        
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()

