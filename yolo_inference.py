from ultralytics import YOLO # Import YOLO class from ultralytics

model = YOLO('models/best (1).pt') # Load the best yolo model, if doesnt work- try yolov5s, yolov5m, yolov5l, yolov5x
results = model.predict('input_videos/08fd33_4.mp4',save=True) # Predict on a video and save the output

print(results[0]) # Print the results
print('===================================')
for box in results[0].boxes:
    print(box) # Print the boxes