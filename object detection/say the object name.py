import cv2
import pyttsx3
from ultralytics import YOLO


engine = pyttsx3.init()
engine.setProperty('rate', 150)  


model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)
   
    
    detected_objects = set()
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])      
            obj_name = model.names[cls]
            detected_objects.add(obj_name)

    
    if detected_objects:
        text = "Detected " + ", ".join(detected_objects)
        print(text)
        engine.say(text)
        engine.runAndWait()

    
    cv2.imshow("YOLOv8 Object Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
