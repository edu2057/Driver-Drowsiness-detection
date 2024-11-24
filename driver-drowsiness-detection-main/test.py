import cv2
import torch
import pygame
import numpy as np
from ultralytics import YOLO
from collections import deque
import datetime

# Constants definition
FPS = 30  # Frames per second
WARNING_DURATION = 2  # Warning duration (seconds)
QUEUE_DURATION = 2  # Queue duration (seconds)
YAWN_THRESHOLD_FRAMES = int(FPS * 1)  # Yawning threshold in frames -> changed to around 1 second for demonstration purposes
# DROWSY_THRESHOLD_FRAMES = int(FPS * 0.4)  # Weak drowsiness threshold in frames -> combined as a 'drowsy' state
DROWSY_THRESHOLD_FRAMES = int(FPS * 0.8)  # Strong drowsiness threshold -> combined all sleep -> drowsy
HEAD_THRESHOLD_FRAMES = int(FPS * 0.8)  # Head movement threshold in frames

def play_alarm(sound_file, duration):
    # Initialize Pygame mixer
    pygame.mixer.init()
    # Load sound file
    alarm_sound = pygame.mixer.Sound(sound_file)
    # Play sound for the specified duration
    alarm_sound.play(loops=0, maxtime=duration)  # Repeat the alarm sound for 3 seconds

def trigger_alarm(trigger, sound_file, duration):
    if trigger:
        print("Alarm is ringing!")
        play_alarm(sound_file, duration)
    else:
        print("Alarm is not ringing.")

def get_webcam_fps():
    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("Unable to access the webcam.")
        return None
    
    
    
    # Get the FPS (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30

def load_model(model_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)
    return model

def webcam_detection(model, fps):
    queue_length = int(fps * QUEUE_DURATION)
    
    drowsy_threshold_frames = int(fps * 0.8)  # Strong drowsiness threshold
    yawn_threshold_frames = int(fps * 1)
    head_threshold_frames = int(fps * 0.8)
    
    eye_closed_queue = deque(maxlen=queue_length)
    yawn_queue = deque(maxlen=queue_length)
    head_queue = deque(maxlen=queue_length)
    head_warning_time = None
    yawn_warning_time = None
    drowsy_warning_time = None
    alarm_end_time = None

    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("Unable to access the webcam.")
        return

    while True:  # Process frames continuously
        ret, frame = cap.read()  # Read a frame
        if not ret:
            print("Failed to grab frame.")
            break

        # Preprocess the image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
        results = model.predict(source=[img], save=False)[0]  # Get model predictions -> [0] for the first result

        # Visualize the results and print object information
        detected_event_list = []  # Initialize an empty list to store detected events
        current_eye_closed = False
        current_yawn = False
        current_head_event = False

        for result in results:  # Loop through each detected object
            boxes = result.boxes  # Extract bounding boxes from the results
            xyxy = boxes.xyxy.cpu().numpy()  # Convert bounding box coordinates to numpy arrays
            confs = boxes.conf.cpu().numpy()  # Convert confidence scores to numpy arrays
            classes = boxes.cls.cpu().numpy()  # Convert class IDs to numpy arrays

            for i in range(len(xyxy)):  # Loop through each detected object
                xmin, ymin, xmax, ymax = map(int, xyxy[i])  # Extract bounding box coordinates
                confidence = confs[i]  # Get confidence score
                label = int(classes[i])  # Get class ID

                # Print object information
                print(f"Detected {model.names[label]} with confidence {confidence:.2f} at [{xmin}, {ymin}, {xmax}, {ymax}]")

                if confidence > 0.5:  # Only display objects with confidence > 0.5
                    label_text = f"{model.names[label]} {confidence:.2f}"

                    # Default color (green)
                    color = (0, 255, 0)

                    # Check for eye closed state (assumed labels 0, 1, 2 are eye closed)
                    if label in [0, 1, 2]:
                        current_eye_closed = True

                    # Check for head down or up state (assumed labels 4, 5 are head states)
                    if label in [4, 5]:
                        color = (0, 255, 255)  # Yellow color
                        current_head_event = True

                    # Check for yawning state (assumed label 8 is yawn)
                    if label == 8:
                        color = (0, 255, 255)  # Yellow color
                        current_yawn = True

                    # Draw bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add the current eye closed state to the queue
        eye_closed_queue.append(current_eye_closed)

        # Add yawning event to the queue
        yawn_queue.append(current_yawn)

        # Add head movement event to the queue
        head_queue.append(current_head_event)

        # Determine drowsiness and sleep states based on eye closed status
        eye_closed_count = sum(eye_closed_queue)
        if eye_closed_count >= drowsy_threshold_frames:
            detected_event_list.append('drowsy')
            drowsy_warning_time = datetime.datetime.now()
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 3000)  # Trigger alarm for 3 seconds
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=3)

        # Check for yawning state
        yawn_count = sum(yawn_queue)
        if yawn_count >= yawn_threshold_frames:
            detected_event_list.append('yawn')
            yawn_warning_time = datetime.datetime.now()
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 1000)  # Trigger alarm for 1 second
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
            yawn_queue.clear()  # Reset yawning state

        # Check for head movement state
        head_event_count = sum(head_queue)
        if head_event_count >= head_threshold_frames:
            detected_event_list.append('head_movement')
            head_warning_time = datetime.datetime.now()
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 1000)  # Trigger alarm for 1 second
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
            head_queue.clear()  # Reset head movement state

        if eye_closed_count < drowsy_threshold_frames and yawn_count < yawn_threshold_frames and head_event_count < head_threshold_frames:
            alarm_end_time = None

        # Get current time
        current_time = datetime.datetime.now()

        # Change color based on eye closed state
        for result in results:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                xmin, ymin, xmax, ymax = map(int, xyxy[i])
                label = int(classes[i])
                if label in [0, 1, 2]:  # Change color only for eye closed state
                    if 'drowsy' in detected_event_list:
                        color = (0, 0, 255)  # Red color
                    else:
                        color = (0, 255, 0)  # Green color

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, model.names[label], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display warning messages
        font_scale = 0.75  # Reduce font size
        font_thickness = 2  # Reduce font thickness

        if drowsy_warning_time and (current_time - drowsy_warning_time).total_seconds() < WARNING_DURATION:
            cv2.putText(frame, 'Warning: Drowsy Detected!', (50, 150), cv2.FONT_ITALIC, font_scale, (0, 0, 255), font_thickness)
        if yawn_warning_time and (current_time - yawn_warning_time).total_seconds() < WARNING_DURATION:
            cv2.putText(frame, 'Warning: Yawning Detected!', (50, 50), cv2.FONT_ITALIC, font_scale, (0, 255, 255), font_thickness)
        if head_warning_time and (current_time - head_warning_time).total_seconds() < WARNING_DURATION:
            cv2.putText(frame, 'Warning: Head Movement Detected!', (50, 100), cv2.FONT_ITALIC, font_scale, (0, 255, 255), font_thickness)

        cv2.imshow("Driver Drowsiness Detection System", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Initialize the system
fps = get_webcam_fps() or FPS  # Get webcam FPS, default to 30 if unavailable
model_path = "yolov8_model.pt"  # Specify the correct path to your YOLOv8 model
model = load_model(model_path)

# Start webcam detection
webcam_detection(model, fps)
