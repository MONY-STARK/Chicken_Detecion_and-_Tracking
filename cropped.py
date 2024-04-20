import cv2
from OCR import EasyOCR
# Load the image
image = cv2.imread(r"Data\test\images\data75_jpg.rf.be6dca1fbb6dc49dfbdf6953329cd536.jpg")

ocr = EasyOCR()


#img = image[35:100, 15:120]

grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(grey, 179, 255, cv2.THRESH_BINARY)

text1 = ocr.get_text(image)
text = ocr.get_text(binary)

print(f"image text {text1}")
print(f"binary text {text}")

cv2.imshow("BinaryImage", binary)
cv2.imshow("grey", grey)
cv2.imshow("img" ,image)
cv2.waitKey(0)

"""# Display the original and cropped images (optional)
cv2.imshow('Original Image', image)
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the cropped image to a file
cv2.imwrite('cropped_image.jpg', cropped_image)"""
