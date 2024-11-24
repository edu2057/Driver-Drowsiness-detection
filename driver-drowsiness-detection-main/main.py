'''
2024-05-23 Author: Jang Chang-ho 
webcam_drowsiness_detection.py (v.3) 

1. Implemented on Chang-ho's laptop environment.
2. Hyun-min's model name (best.pt).
3. Modified requirements.txt (added pygame package).
4. Created separate queues for drowsiness, yawning, and head movements. Bounding box (BB) colors: (Red, Yellow, Yellow).
5. Merged "strong drowsiness" and "mild drowsiness" into one "drowsiness" category (800 ms threshold). Adjusted human average yawn time (6000 ms -> testing with 1000 ms). For head nodding or tilting (800 ms threshold).
6. Added a warning message display on the screen when specific conditions for drowsiness, yawning, or head movement are detected.
7. Added an alarm file `alarm.wav`.
8. Alarm for drowsiness detection: Repeats until the drowsy state ends (3 seconds duration).
9. Alarm for yawning or head movement detection: Plays once (1 second duration) and then resets.
'''

import cv2
import torch
import pygame
import numpy as np
from ultralytics import YOLO
from collections import deque
import datetime

# Define constants
FPS = 30  # Frames per second
WARNING_DURATION = 2  # Duration of warning display (seconds)
QUEUE_DURATION = 2  # Time to store in the queue (seconds)
YAWN_THRESHOLD_FRAMES = int(FPS * 1)  # Frame threshold for yawning -> Changed to 1 second for testing 
DROWSY_THRESHOLD_FRAMES = int(FPS * 0.8)  # Frame threshold for strong drowsiness -> All sleep events classified as "drowsy"
HEAD_THRESHOLD_FRAMES = int(FPS * 0.8)  # Frame threshold for head movement events

def play_alarm(sound_file, duration):
    # Initialize Pygame mixer
    pygame.mixer.init()
    # Load sound file
    alarm_sound = pygame.mixer.Sound(sound_file)
    # Play sound (for the specified duration)
    alarm_sound.play(loops=0, maxtime=duration)  # Use loop for partial replay of 8-second alarm

def trigger_alarm(trigger, sound_file, duration):
    if trigger:
        print("Alarm is ringing!")
        play_alarm(sound_file, duration)
    else:
        print("Alarm is not ringing.")

def get_webcam_fps():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Cannot access the webcam.")
        return None
    
    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30

def load_model(model_path):
    # Load YOLOv8 model
    model = YOLO(model_path)
    return model

def webcam_detection(model, fps):
    queue_length = int(fps * QUEUE_DURATION)
    # Define again
    drowsy_threshold_frames = int(fps * 0.8)  # Strong drowsiness -> Classified as "drowsy"
    yawn_threshold_frames = int(fps * 1)
    head_threshold_frames = int(fps * 0.8)
    
    eye_closed_queue = deque(maxlen=queue_length)
    yawn_queue = deque(maxlen=queue_length)
    head_queue = deque(maxlen=queue_length)
    head_warning_time = None
    yawn_warning_time = None
    drowsy_warning_time = None
    alarm_end_time = None

    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Cannot access the webcam.")
        return

    while True:  # Process each frame
        ret, frame = cap.read()  # Read frame
        if not ret:
            print("Cannot retrieve frame.")
            break

        # Preprocess the image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
        results = model.predict(source=[img], save=False)[0]  # Model prediction (first result)

        # Visualize results and output object information
        detected_event_list = []  # Initialize empty list to store detected events
        current_eye_closed = False
        current_yawn = False
        current_head_event = False

        for result in results:  # Process each detected object
            boxes = result.boxes  # Extract bounding boxes
            xyxy = boxes.xyxy.cpu().numpy()  # Convert bounding box coordinates to numpy array
            confs = boxes.conf.cpu().numpy()  # Convert confidence scores to numpy array
            classes = boxes.cls.cpu().numpy()  # Convert class IDs to numpy array

            for i in range(len(xyxy)):  # Iterate over detected objects
                xmin, ymin, xmax, ymax = map(int, xyxy[i])  # Extract bounding box coordinates
                confidence = confs[i]  # Confidence score for object
                label = int(classes[i])  # Class ID for object
                
                # Print object information
                print(f"Detected {model.names[label]} with confidence {confidence:.2f} at [{xmin}, {ymin}, {xmax}, {ymax}]")

                if confidence > 0.5:  # Display objects with confidence > 0.5
                    label_text = f"{model.names[label]} {confidence:.2f}"

                    # Default color (green)
                    color = (0, 255, 0)

                    # Check for eye-closed state (assume labels 0, 1, 2 indicate closed eyes)
                    if label in [0, 1, 2]:
                        current_eye_closed = True

                    # Check for head tilting/nodding state (assume labels 4, 5 indicate head movement)
                    if label in [4, 5]:
                        color = (0, 255, 255)  # Change to yellow
                        current_head_event = True

                    # Check for yawning state (assume label 8 indicates yawning)
                    if label == 8:
                        color = (0, 255, 255)  # Change to yellow
                        current_yawn = True

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)  # Draw bounding box
                    cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Process and detect events...

if __name__ == "__main__":
    fps = get_webcam_fps()
    print(f"Webcam FPS: {fps}")
    model_path = 'best.pt'  # Set model file path
    model = load_model(model_path)
    webcam_detection(model, fps)
