from ultralytics import YOLO
import cv2
import os

model = YOLO("C:\\Coding Porfolio For Study\\Yolo learning\\runs\detect\\train2\\weights\\best.pt")

# model.train(data='C:/Coding Porfolio For Study/Yolo learning/data.yaml', epochs=50, imgsz=640)

file_path = input("Please enter the path to the video or image file: ")

if os.path.splitext(file_path)[1] in ['.jpg', '.jpeg', '.png']:
    image_mode = True
    frame = cv2.imread(file_path)
    if frame is None:
        print("Error: Image not found. Check the file path.")
else:
    image_mode = False
    cap = cv2.VideoCapture(file_path)


def process_frame(frame):
    results = model(frame)
    if len(results) == 0:
        print("No objects detected.")
    else:
        print(f"Objects detected: {len(results)}")

    print("Class names in the model:", model.names)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
            conf = box.conf[0].cpu().numpy()  # Confidence score
            cls = int(box.cls[0].cpu().numpy())  # Class ID

            if model.names[cls] in ['traffic_cone', 'car']:
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)

    return frame


if image_mode:
    annotated_frame = process_frame(frame)
    cv2.imwrite("output_image.jpg", annotated_frame)
    print("Annotated image saved as output_image.jpg")

    cv2.waitKey(0)
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = process_frame(frame)
        cv2.imshow('Video Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()
