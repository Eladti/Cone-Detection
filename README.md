# Cone-Detection
This project showcases object detection using the YOLO (You Only Look Once) deep learning model integrated with OpenCV.
It processes both images and videos to detect specific objects such as traffic cones and cars, annotating them with bounding boxes and confidence scores.
The project is implemented on Windows OS using PyCharm IDE.

# Features
1. Supports image and video file inputs.
2. detected objects with bounding boxes and confidence scores.
3. Saves annotated images to disk.
4. Displays annotated frames in real-time for videos.

# Prerequisites
Operating System: Windows 10 or later.
Development Environment: PyCharm IDE.
Python Version: >= 3.7
Required Python Libraries:
ultralytics==8.1.26
opencv-python==4.10.0.84

# YOLO Model Weights:
Ensure you have a YOLO model file (best.pt) trained with the desired dataset.
The model must include classes such as "traffic_cone" for this script.

# Installation
1. Clone or download this repository to your local machine.
2. Open the project in PyCharm IDE.
3. Install the required Python libraries by running the following the requirements.txt file.
4. Place your YOLO model weights file (best.pt) in the desired directory. Update the path in the script:
  model = YOLO("path_to_your_model/best.pt")
5. Ensure you have an image or video file ready for testing.
6. Run Program

When prompted, enter the full path without the "" to the image or video file:

For images, supported formats: .jpg, .jpeg, .png.
For videos, supported formats: .mp4, .avi, etc.

# Image:
The program processes the image, annotates detected objects, and saves the result as output_image.jpg.
# Video:
The program processes each video frame and displays the annotated frames in real-time.
Press q to exit video playback.

# How It Works
The project employs a deep learning-based approach to object detection using the YOLO framework:
The YOLO model is pre-trained or fine-tuned with a custom dataset to detect objects of interest, but for us to detect annotations of cones i used CVAT to make annotations by hand.
Then the results of the data to be used by the YOLO model to the rest of the process.

the is a line of code under # for the reason of training the model in 50 epochs - after training the code will not need that line and could work on detecting cones.
The ultralytics library simplifies model loading and usage.

# Input Handling:
The program checks the file extension to determine whether the input is an image or a video.
Images are read using OpenCV’s cv2.imread(), while videos are processed frame by frame using cv2.VideoCapture().

# Object Detection:
Each frame (or image) is passed to the YOLO model for detection.
The model returns bounding box coordinates, confidence scores, and class IDs for detected objects.
Annotation:

For objects matching traffic_cone, bounding boxes are drawn on the frame.
The confidence score and class name are displayed above the bounding box.
For images, the annotated result is saved as output_image.jpg.
For videos, annotated frames are displayed in real-time.

Input: An image containing a car and traffic cone.
Output: Annotated image with bounding boxes and confidence scores.
Video Detection: The video is displayed frame by frame with annotations, showing bounding boxes around detected objects in real-time.

This project is licensed under the MIT License. Feel free to use and modify it for personal or educational purposes.
