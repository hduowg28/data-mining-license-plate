from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import cv2
import pytesseract

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

    with open(input_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(",")
        img_full = parts[0]  
        x1, y1, x2, y2 = map(int, parts[1:5])
        conf = float(parts[5])

        img_name, index = img_full.rsplit("_", 1)
        print(index, img_name)
    
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        h, w = img.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        plate = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (300, 120))
        gray = gray[:, 10:-10]
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        gray = cv2.equalizeHist(gray)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            gray = gray[y:y+h, x:x+w]
        kernel = np.ones((3,3), np.uint8)
        # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
        save_name = f"{output_folder}/{img_name}_{index}.jpg"
        cv2.imwrite(save_name, gray)

        print("Đã lưu:", save_name)

# def crop_and_process():
#     input_file = "test.txt"
#     image_folder = "dataset_license_plate/test2"
#     output_folder = "plates test"

#     os.makedirs(output_folder, exist_ok=True)

#     with open(input_file, "r") as f:
#         lines = f.readlines()

#     for line in lines:
#         parts = line.strip().split(",")
#         img_full = parts[0]
#         x1, y1, x2, y2 = map(int, parts[1:5])
#         conf = float(parts[5])

#         img_name, index = img_full.rsplit("_", 1)
#         print(index, img_name)

#         img_path = os.path.join(image_folder, img_name)
#         img = cv2.imread(img_path)

#         h, w = img.shape[:2]

#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(w, x2)
#         y2 = min(h, y2)

#         plate = img[y1:y2, x1:x2]

#         gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
#         gray = cv2.resize(gray, (320, 140))
#         gray = gray[:, 10:-10]
#         gray = cv2.bilateralFilter(gray, 11, 17, 17)
#         gray = cv2.equalizeHist(gray)

#         thresh = cv2.adaptiveThreshold(gray, 255,
#                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY_INV,
#                                        15, 5)

#         kernel = np.ones((3, 3), np.uint8)
#         thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         boxes = []
#         for c in contours:
#             x, y, w, h = cv2.boundingRect(c)
#             area = w * h
#             if area > 300 and h > 25:
#                 boxes.append((x, y, w, h))

#         if boxes:
#             x_min = min([b[0] for b in boxes])
#             y_min = min([b[1] for b in boxes])
#             x_max = max([b[0] + b[2] for b in boxes])
#             y_max = max([b[1] + b[3] for b in boxes])
#             thresh = thresh[y_min:y_max, x_min:x_max]
#             gray = gray[y_min:y_max, x_min:x_max]

#         coords = np.column_stack(np.where(thresh > 0))
#         if len(coords) > 0:
#             rect = cv2.minAreaRect(coords)
#             angle = rect[-1]

#             if angle < -45:
#                 angle = -(90 + angle)
#             else:
#                 angle = -angle

#             (h, w) = thresh.shape[:2]
#             center = (w // 2, h // 2)

#             M = cv2.getRotationMatrix2D(center, angle, 1.0)
#             thresh = cv2.warpAffine(thresh, M, (w, h),
#                                     flags=cv2.INTER_CUBIC,
#                                     borderMode=cv2.BORDER_REPLICATE)

#         coords = np.column_stack(np.where(thresh > 0))
#         if len(coords) > 0:
#             rect = cv2.minAreaRect(coords)
#             box = cv2.boxPoints(rect)
#             box = box.astype("float32")

#             w = int(rect[1][0])
#             h = int(rect[1][1])

#             if w > 0 and h > 0:
#                 dst_pts = np.array([
#                     [0, h - 1],
#                     [0, 0],
#                     [w - 1, 0],
#                     [w - 1, h - 1]
#                 ], dtype="float32")

#                 M = cv2.getPerspectiveTransform(box, dst_pts)
#                 thresh = cv2.warpPerspective(thresh, M, (w, h))

#         kernel = np.ones((2, 2), np.uint8)
#         thresh = cv2.dilate(thresh, kernel, iterations=1)

#         thresh = cv2.bitwise_not(thresh)

#         save_name = f"{output_folder}/{img_name}_{index}.jpg"
#         cv2.imwrite(save_name, thresh)

#         print("Đã lưu:", save_name)
        
def ocr_from_plates():
    folder="plates test"
    for file in os.listdir(folder):
        img_path=os.path.join(folder, file)
        img = cv2.imread(img_path)

        text = pytesseract.image_to_string(
            img, 
            config="--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        text=text.strip()
        print(file, "->", text)


if __name__ == "__main__":
    # main()
    crop_and_process()
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ocr_from_plates()
    print("thanh cong")

