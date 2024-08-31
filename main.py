from ultralytics import YOLO
import cv2

class Detection:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    def predict(self, img, conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = self.model.predict(img, conf=conf)
        for result in results:
            for box in result.boxes:
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
        return img, results

