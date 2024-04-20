import cv2 as cv
from ultralytics import YOLO

class Object_Detection():

    def __init__(self):
        self.model = YOLO("best.pt")
        

    def detect(self,result):
        boxes  = self.get_bbox(result)
        names, cls, score = self.get_label(result)
        classes = zip(names, cls)

        return boxes, score, classes

    def draw_bbox(self, frame, result):
        box = self.get_bbox(result)
        names, cls, score = self.get_label(result)

        for i, (x, y, w, h) in enumerate(box):
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            start_point = (x1, y1)
            end_point = (x2, y2)
            
            label = f"{names[int(cls[i])]} {score[i]:.2f}"
            
            cv.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            
            text_size = cv.getTextSize(label, cv.FONT_HERSHEY_COMPLEX, 0.5, 1)[0]
            cv.rectangle(frame, (int(x1), int(y1) - text_size[1] - 5), (int(x1) + text_size[0], int(y1) - 5), (0, 0, 225), -1)
            
            img = cv.putText(frame, label, (int(x1), int(y1) - 5), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

        return img

    def get_label(self, result):
        names, cls = self.get_class(result)
        score = self.get_score(result)

        return names, cls, score

    def get_bbox(self, result):
        boxes = result[0].boxes.xywh
        return boxes

    def get_score(self, result):
        score = result[0].boxes.conf.tolist()
        return score

    def get_class(self, result):
        names = result[0].names
        cls = result[0].boxes.cls

        return names, cls
