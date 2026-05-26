from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import cv2
import pytesseract
import re
from paddleocr import PaddleOCR
import easyocr

# ocr = PaddleOCR(use_angle_cls=True, lang='en')
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

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

def luutenbienso():

    folder = r"C:\data mining\test-20260517T152223Z-3-001\test\images"

    output_file = "fileRealPlate.txt"

    open(output_file, "w").close()

    for file in os.listdir(folder):

        plate_name = os.path.splitext(file)[0]

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{plate_name}\n")

def ocr_from_plates_padOCR():
    folder = r"C:\data mining\test-20260517T152223Z-3-001\test\images"
    output_file = "answerOCR.txt"

    model = YOLO("C:/data mining/best.pt")

    open(output_file, "w").close()

    for file in os.listdir(folder):

        img_path = os.path.join(folder, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        results = model.predict(img)

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                h, w = img.shape[:2]

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                plate = img[y1:y2, x1:x2]

                plate = cv2.resize(
                    plate,
                    None,
                    fx=2.5,
                    fy=2.5,
                    interpolation=cv2.INTER_CUBIC
                )

                gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

                _, thresh = cv2.threshold(
                    gray,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                cv2.imshow("anh thresh dua vao OCR", thresh)

                result_ocr = ocr.ocr(thresh, cls=True)

                text = ""

                if result_ocr and result_ocr[0]:

                    for line in result_ocr[0]:
                        text += line[1][0]

                text = re.sub(r'[^A-Z0-9]', '', text)

                text = post_process(text)

                print(file, "->", text)

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"{file},{text}\n")

    print(f"\nĐã lưu vào file: {output_file}")

def video_ocr():
    model = YOLO("C:/data mining/best.pt")
    cap = cv2.VideoCapture("videotest.mp4")
    # webcam:
    # cap = cv2.VideoCapture(0)

    detected_plates = set()
    frame_count = 0
    
    last_text = "" 

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Không đọc được frame")
            break
            
        frame_count += 1
        frame = cv2.resize(frame, (640, 480))

        # Đã set conf=0.34 ở đây thì kết quả trả về chắc chắn >= 0.34
        results = model.predict(frame, verbose=False, conf=0.34)

        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if frame_count % 5 == 0:
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    plate = frame[y1:y2, x1:x2]

                    plate = cv2.resize(plate, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    cv2.imshow("Anh Thresh Dua Vao OCR", thresh)

                    result_ocr = ocr.ocr(thresh, cls=True)
                    text = ""
                    if result_ocr and result_ocr[0]:
                        for line in result_ocr[0]:
                            text += line[1][0]

                    print(f"Frame {frame_count} - OCR raw: '{text}'")

                    # Làm sạch
                    text = re.sub(r'[^A-Z0-9]', '', text)

                    if len(text) >= 4: 
                        text = post_process(text)
                        last_text = text 
                        
                        if text not in detected_plates:
                            print("=> DETECTED OK:", text)
                            detected_plates.add(text)

                if last_text != "":
                    cv2.putText(
                        frame,
                        last_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

        cv2.imshow("License Plate Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def ocr_from_plates_EASY():
    folder = r"C:\data mining\\test-20260517T152223Z-3-001\\test\\images"
    output_file = "answerOCR_EasyOCR.txt"

    model = YOLO("C:/data mining/best.pt")
    reader = easyocr.Reader(['en'], gpu=True) 

    # Làm trống file output trước khi ghi mới
    open(output_file, "w").close()

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        results = model.predict(img, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                plate = img[y1:y2, x1:x2]

                # Tiền xử lý ảnh giống hệt hàm PaddleOCR
                plate = cv2.resize(
                    plate,
                    None,
                    fx=2.5,
                    fy=2.5,
                    interpolation=cv2.INTER_CUBIC
                )

                gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

                _, thresh = cv2.threshold(
                    gray,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                cv2.imshow("anh thresh dua vao OCR", thresh)
                result_ocr = reader.readtext(thresh)
                text = ""
                if result_ocr:
                    for line in result_ocr:
                        text += line[1]  

                text = re.sub(r'[^A-Z0-9]', '', text.upper())
                text = post_process(text)
                print(file, "->", text)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"{file},{text}\n")
    print(f"\nĐã lưu vào file: {output_file}")

def post_process(st):
    st = re.sub(r'[^A-Z0-9]', '', st.upper())
    
    if len(st) <= 7:
        return st
        
    char_to_num = {'A': '4', 'B': '8', 'D': '0', 'G': '6', 'I': '1', 'O': '0', 'S': '5', 'Z': '2', 'Q': '0'}
    num_to_char = {'0': 'D', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
    
    st_list = list(st)
    
    for i in range(2):
        if st_list[i] in char_to_num:
            st_list[i] = char_to_num[st_list[i]]
            
    if st_list[2] in num_to_char:
        st_list[2] = num_to_char[st_list[2]]
        
    for i in range(len(st_list) - 4, len(st_list)):
        if st_list[i] in char_to_num:
            st_list[i] = char_to_num[st_list[i]]
            
    processed_st = "".join(st_list)
    return processed_st


if __name__ == "__main__": 
    # main()
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # crop_and_process()
    # luutenbienso()
    # ocr_from_plates_tesOCR()
    # video_ocr()
    # ocr_from_plates_padOCR()
    # ocr_from_plates_paddle()
    ocr_from_plates_EASY()
    print("thanh cong")

