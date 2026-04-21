from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import cv2
import pytesseract
import re

# ...existing code...

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
        save_name = os.path.join(output_folder, f"{img_name}_{index}.jpg")
        cv2.imwrite(save_name, plate)

        print("Saved:", save_name)
def ocr_from_plates():
    folder = "plates test"

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)

        img = cv2.imread(img_path)
        text = pytesseract.image_to_string(
            img,
            config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        texts.append(text.strip())
        final_text = "".join(texts)
        final_text = re.sub(r"[^A-Z0-9]", "", final_text.upper())

        if not final_text:
            final_text = "None"

        print(f"{file} -> {final_text}")


# def ocr_from_plates():
#     folder="plates test"
#     for file in os.listdir(folder):
#         img_path=os.path.join(folder, file)
#         img = cv2.imread(img_path)

#         text = pytesseract.image_to_string(
#             img, 
#             config="--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#         )
#         text=text.strip()
#         print(file, "->", text)


if __name__ == "__main__":
    # main()
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    crop_and_process()

    ocr_from_plates()
    print("thanh cong")

