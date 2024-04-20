import cv2 as cv
from ultralytics import YOLO
from Object_Detection import Object_Detection
from OCR import EasyOCR

file_path = r"IMG_2659.MOV"

model = YOLO("best.pt")
od = Object_Detection()
ocr = EasyOCR()
cap = cv.VideoCapture(file_path)
fps = cap.get(cv.CAP_PROP_FPS)

while True:

    ret, frame = cap.read()

    if ret is not True:
        break

    result = model.predict(frame)

    img = od.draw_bbox(frame, result)
    _, imgs = ocr.get_wm_img(frame) 
    text = ocr.get_text(imgs)
    print(f"text {text} ")

    resized_frame = cv.resize(img, (800, 600))  # Adjust the dimensions as needed

    cv.imshow("img", resized_frame)
    
        
    delay = int(1000 / fps)  

    if cv.waitKey(delay) & 0xFF == ord('q'):  
        break

cap.release()
cv.destroyAllWindows()