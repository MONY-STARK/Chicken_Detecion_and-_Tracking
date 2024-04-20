
from ultralytics import YOLO
import easyocr
import numpy as np
import cv2 as cv
import os

class EasyOCR():
    
    reader = easyocr.Reader(["en"])

    def __init__(self, ):
        self.reader = self.initialize_ocr_reader()
        self.model = YOLO("best.pt")
        #self.img_path = r"C:\Users\lenovo\Projects\Project - DS and AI\Project - 2 Object Detection and Weight Tracking\Code\Data\test\images\data393_jpg.rf.e72cb8f7566426d9a7bd599e1240d896.jpg"
        #self.img = cv.imread(self.img_path)


    def initialize_ocr_reader(self):
        reader = easyocr.Reader(['en'])
        return reader

    
    def get_wm_img(self,frame):
        #img = cv.resize(frame, (512, 640))
            results = self.model.predict(frame)
            wm = [3]
            weight_machine_coor = []
            cropped_wm = np.zeros((1, 1, 3), dtype=np.uint8)
            cropped_num = np.zeros((1, 1, 3), dtype=np.uint8) 
            binary = np.zeros((1, 1, 3), dtype=np.uint8)


            for result in results[0].boxes.data.tolist():
                x1, y1, x2, y2, score, cls_id = result

                if int(cls_id) in wm :
                    weight_machine_coor.append((x1, y1, x2, y2, score))

                    cropped_wm = frame[int(y1):int(y2), int(x1):int(x2), :]
                    grey = cv.cvtColor(cropped_wm, cv.COLOR_BGR2GRAY)
                    _, binary = cv.threshold(grey, 179, 255, cv.THRESH_BINARY_INV)

                    #cropped_num = cropped_wm[35:100, 15:120]
                    # grey = cv.cvtColor(cropped_num, cv.COLOR_BGR2GRAY)
                    # _, binary_image = cv.threshold(grey, 180, 255, cv.THRESH_BINARY_INV)
                    # cv.imshow("cropped_wm" ,cropped_wm)
                    # cv.imshow("cropped_num",cropped_num )
                    cv.imshow("binary img", binary)
                    # cv.waitKey(0)
            
            return weight_machine_coor, binary

    
    def get_text(self, frame):

        result = self.reader.readtext(frame)

        for (bbox, text, prob) in result:
            return text
        

        

# if __name__ == "__main__" :

#     ocr = OCR()

#     path = r"C:\Users\lenovo\Projects\Project - DS and AI\Project - 2 Object Detection and Weight Tracking\Code\Data\test\images\data848_jpg.rf.beb9153a89d01b8eb2220226953ccaa7.jpg"

#     img = cv.imread(path)

#     _, imgs = ocr.get_wm_img(img)
#     text = ocr.get_text(imgs)

    # output_dir = r"C:\Users\lenovo\Projects\Project - DS and AI\Project - 2 Object Detection and Weight Tracking\Code" 
    # output_path = output_dir + "imgs.jpg"


    # cv.imwrite(output_path, imgs)
    # print(text)