import cv2
import numpy as np
import torch
from urllib.request import urlopen
from io import BytesIO
import pyttsx3

model = torch.hub.load("ultralytics/yolov5", "yolov5x6")

def get_detected_objects_image(image):
    results = model(image)
    annotated_image = results.render()[0]

    engine = pyttsx3.init()  # Initialize TTS engine

    # Extract object class
    objects = results.pandas().xyxy[0]  # Assuming results.pandas() provides object data
    for index, row in objects.iterrows():
        object_class = row['name']

        # Announce detected object
        engine.say("Detected: " + object_class)
        engine.runAndWait()

    return annotated_image

def run():
    cv2.namedWindow("Live Transmission", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_img = get_detected_objects_image(frame)
        cv2.imshow('Live Transmission', annotated_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Started")
    run()
