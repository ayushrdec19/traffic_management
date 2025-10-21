import cv2
from ultralytics import YOLO

yolo = YOLO('yolov8s.pt')

videoCap = cv2.VideoCapture(0)

if not videoCap.isOpened():
    print("Error: Camera could not be accessed.")
else:
    print("Camera opened successfully.")


def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
             ((cls_num // len(base_colors)) % 5) for i in range(3)]
    return tuple(color)


specific_vehicles = ['car', 'motorbike']

while True:
    ret, frame = videoCap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    results = yolo(frame)

    specific_vehicle_count = {vehicle: 0 for vehicle in specific_vehicles}

    for result in results:
        classes_names = result.names

        for box in result.boxes:
            if box.conf[0] > 0.4:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                class_name = classes_names[cls]

                if class_name in specific_vehicles:
                    specific_vehicle_count[class_name] += 1

                    colour = getColours(cls)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

    for vehicle, count in specific_vehicle_count.items():
        cv2.putText(frame, f'{vehicle.capitalize()}s: {count}',
                    (10, 50 + 50 * specific_vehicles.index(vehicle)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting loop...")
        break
videoCap.release()
cv2.destroyAllWindows()
# traffic_management
