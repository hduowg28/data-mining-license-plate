from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import cv2
import pytesseract
import re
from paddleocr import PaddleOCR

# ocr = PaddleOCR(use_angle_cls=True, lang='en')

def write_file(file_name, data, img_name, conf):
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write(f"{img_name},{data[0]},{data[1]},{data[2]},{data[3]},{conf}\n")


def main(): 
    model = YOLO("C:/data mining/best.pt")
    results=model.predict(source="dataset_license_plate/test2")
    # results=model.predict(source="test1.png")
    for result in results:
        img_name=os.path.basename(result.path)
        for i, box in enumerate(result.boxes):
            # print(box)
            conf=float(box.conf[0])
            
            data = list(map(int, box.xyxy[0]))
            # print(data,img_name)
            write_file("test.txt", data, f"{img_name}_{i}", conf)


def crop_and_process():
    input_file = "test.txt"
    image_folder = "dataset_license_plate/test2"
    output_folder = "plates test"

    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 5:
            continue

        img_full = parts[0]
        x1, y1, x2, y2 = map(int, parts[1:5])

        img_name, index = img_full.rsplit("_", 1)
        img_path = os.path.join(image_folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        plate = img[y1:y2, x1:x2]

        plate = cv2.resize(plate, None, fx=2, fy=2)

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        save_name = os.path.join(output_folder, f"{img_name}_{index}.jpg")
        cv2.imwrite(save_name, thresh)

        print("Saved:", save_name)


def ocr_from_plates_tesOCR():
    folder="plates test"
    for file in os.listdir(folder):
        img_path=os.path.join(folder, file)
        img = cv2.imread(img_path)

        text = pytesseract.image_to_string(
            img, 
            config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        text=text.strip()
        print(file, "->", text)

# def ocr_from_plates_paddle():
#     folder = "plates test"

#     for file in os.listdir(folder):
#         img_path = os.path.join(folder, file)
#         img = cv2.imread(img_path)

#         if img is None:
#             continue

#         result = ocr.ocr(img, cls=True)

#         text = ""
#         for line in result:
#             for word in line:
#                 text += word[1][0] + " "

#         text = text.strip()

#         print(file, "->", text)

if __name__ == "__main__": 
    # main()
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    crop_and_process()

    ocr_from_plates_tesOCR()
    # ocr_from_plates_paddle()
    print("thanh cong")

