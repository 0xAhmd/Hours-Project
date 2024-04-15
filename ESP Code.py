import cv2
import numpy as np
import urllib.request
import torch
from io import BytesIO
import pyttsx3  # Import pyttsx3 for text-to-speech functionality

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5x6")  # or yolov5n - yolov5x6, custom

url = 'http://192.168.68.165/cam-hi.jpg'

def get_detected_objects_image(image):
    # Perform inference to detect objects
    results = model(image)
    # Get detected object labels
    labels = results.names
    # Draw bounding boxes and labels on the image
    annotated_image = results.render()[0]
    return annotated_image, labels

def speak_detected_objects(labels):
    # Convert list of labels to a sentence
    sentence = ", ".join(labels)
    # Speak out the detected objects
    engine.say(f"I detect {sentence}")
    engine.runAndWait()

def run1():
    cv2.namedWindow("Live Transmission", cv2.WINDOW_NORMAL)
    while True:
        try:
            # Download image from URL
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgnp, -1)
            
            # Get annotated image with detected objects and labels
            annotated_img, labels = get_detected_objects_image(img)
            
            # Show live transmission window with annotated image
            cv2.imshow('Live Transmission', annotated_img)
            
            # Speak out detected objects
            speak_detected_objects(labels)
            
        except Exception as e:
            print(f"Error downloading image: {e}")
        
        # Exit loop if 'q' key is pressed
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Started")
    run1()
