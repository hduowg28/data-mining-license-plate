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
    model = YOLO(r"C:\data mining\results\runs\yolov10s_custom\weights\best.pt")
    results=model.predict(source="dataset_license_plate/test/images")
    # results=model.predict(source="test1.png")
    for result in results:
        img_name=os.path.basename(result.path)
        for i, box in enumerate(result.boxes):
            # print(box)
            conf=float(box.conf[0])
            
            data = list(map(int, box.xyxy[0]))
            # print(data,img_name)
            write_file("test.txt", data, f"{img_name}_{i}", conf)


# def ocr_from_plates_tesOCR():
#     folder = "plates test"
#     outputfile = "ans_tesOCR.txt"

#     with open(outputfile, "w", encoding="utf-8") as f:
#         for file in os.listdir(folder):
#             img_path = os.path.join(folder, file)

#             img = cv2.imread(img_path)

#             text = pytesseract.image_to_string(
#                 img,
#                 config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#             )

#             text = text.strip().replace("\n", "").replace(" ", "")

#             # lấy tên file không có đuôi .jpg/.png
#             plate_name = os.path.splitext(file)[0]

#             # bỏ _0, _1, _2... ở cuối tên
#             plate_name = re.sub(r'_\d+$', '', plate_name)

#             # lưu theo format: tenbienso,bienso
#             line = f"{plate_name},{text}\n"

#             f.write(line)

#             print(line.strip())

#     print(f"\nĐã lưu kết quả vào file: {outputfile}")

def deskew_image(image):
    # 1. Tìm các đường viền sắc nét trong ảnh bằng Canny
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # 2. Đặt điều kiện: Đường thẳng phải dài ít nhất bằng 1/4 chiều rộng ảnh.
    # Điều này giúp bỏ qua các nét chữ ngắn, chỉ bắt vào đường viền dài của biển số.
    min_length = max(20, image.shape[1] // 4)

    # 3. Sử dụng Hough Transform để trích xuất các đường thẳng
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=min_length, maxLineGap=10)

    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Tính toán góc của đường thẳng (hàm arctan2 tự động bù trừ hướng nghiêng)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Chỉ lấy các đường kẻ gần nằm ngang (nghiêng từ -30 đến 30 độ)
        # Bỏ qua các mép dọc hai bên của biển số
        if -30 < angle < 30:
            angles.append(angle)

    if not angles:
        return image

    # 4. Lấy góc trung vị (median) để không bị nhiễu bởi một đường chéo ngẫu nhiên nào đó
    median_angle = np.median(angles)

    # 5. Xoay ảnh nếu phát hiện độ nghiêng
    if abs(median_angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        # QUAN TRỌNG: Dùng BORDER_REPLICATE để khi xoay, mép ảnh tự kéo giãn màu ra,
        # không tạo thành các tam giác đen/trắng ở góc khiến OCR bị nhầm lẫn.
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    return image
def clear_border(thresh_img):
    # Copy ảnh để không làm hỏng ảnh gốc
    clean = thresh_img.copy()
    h, w = clean.shape[:2]

    # Tạo mặt nạ cho hàm floodFill (yêu cầu của OpenCV là mask phải to hơn ảnh 2 pixel)
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Quét mép trên và mép dưới
    for x in range(w):
        if clean[0, x] == 0:  # Nếu mép trên có màu đen
            cv2.floodFill(clean, mask, (x, 0), 255) # Đổ màu trắng (255) vào
        if clean[h-1, x] == 0: # Nếu mép dưới có màu đen
            cv2.floodFill(clean, mask, (x, h-1), 255)

    # Quét mép trái và mép phải
    for y in range(h):
        if clean[y, 0] == 0:
            cv2.floodFill(clean, mask, (0, y), 255)
        if clean[y, w-1] == 0:
            cv2.floodFill(clean, mask, (w-1, y), 255)

    return clean


def crop_and_process():
    input_file = "test.txt"
    image_folder = "dataset_license_plate/test/images"
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
            print("Không đọc được:", img_path)
            continue

        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # 1. Crop và Resize (Dùng INTER_CUBIC để nét hơn)
        plate = img[y1:y2, x1:x2]
        
        print("crop\n")
        # 2. Chuyển xám và Khử nhiễu
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        print("converGray\n")
        rotated_gray = deskew_image(gray)
        resized_gray = cv2.resize(rotated_gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.bilateralFilter(resized_gray, 11, 75, 75)

        # 3. Adaptive Threshold - QUAN TRỌNG: Dùng 'denoised'
        # Tăng blockSize lên 41 để chữ đặc hơn, C=10 để sạch nền
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 41, 10
        )

        thresh = clear_border(thresh)
        # 4. Xử lý hình thái học
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # Làm dày nét để nối các điểm đứt gãy
        # thresh = cv2.dilate(thresh, kernel, iterations=1)
        # Làm mịn bề mặt chữ
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 5. Thêm lề trắng (Padding) để OCR không bị rối ở mép
        thresh = cv2.copyMakeBorder(thresh, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

        # 6. Lưu ảnh
        save_name = os.path.join(output_folder, f"{img_name}")
        cv2.imwrite(save_name, thresh)

        # print("Saved:", save_name)
        # save_name = os.path.join(output_folder, f"{img_name}.jpg")
        # cv2.imwrite(save_name, thresh)

        # print("Saved:", save_name)

def ocr_from_plates_tesOCR():
    folder = "plates test"
    file_name="results2.txt"

    with open(file_name, 'w', encoding='utf-8') as f:
      f.write("")

    with open(file_name, 'a', encoding='utf-8') as filew:
      for file in os.listdir(folder):
          img_path = os.path.join(folder, file)
          img = cv2.imread(img_path)

          if img is None:
              continue

          text = pytesseract.image_to_string(
              img,
              config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
          )

          text = text.strip()
          text=text.replace("\n", "")
          print(file, "->", text)
          filew.write(f"{file},{text}\n")

def luutenbienso():

    folder = r"C:\data mining\test-20260517T152223Z-3-001\test\images"

    output_file = "fileRealPlate.txt"

    open(output_file, "w").close()

    for file in os.listdir(folder):

        plate_name = os.path.splitext(file)[0]

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{plate_name}\n")

def ocr_from_plates_padOCR():
    folder = r"C:\data mining\dataset_license_plate\test\images"
    output_file = "answerOCR.txt"

    model = YOLO("results/runs/yolov10s_custom/weights/best.pt")

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

                # cv2.imshow("anh thresh dua vao OCR", thresh)

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
    folder = "plates test"
    output_file = "answerOCR_EasyOCR.txt"

    # Khởi tạo mô hình EasyOCR (chỉ chạy 1 lần bên ngoài vòng lặp)
    reader = easyocr.Reader(['en'], gpu=True) 

    # Làm trống file output trước khi ghi mới
    open(output_file, "w").close()

    # Duyệt trực tiếp qua từng ảnh đã cắt trong thư mục "plates test"
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Lấy kích thước ảnh trực tiếp từ file đã cắt
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            continue
        aspect_ratio = w / h

        # Nhận diện với allowlist để ép mô hình bỏ qua ký tự lạ (vết xước, đinh vít...)
        result_ocr = reader.readtext(img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        text = ""
        if result_ocr:
            # THUẬT TOÁN SẮP XẾP LẠI THỨ TỰ BOX CHỮ CHUẨN THEO BIỂN SỐ VIỆT NAM
            if aspect_ratio > 2.5:
                # KIỂU 1: Biển số dài (1 dòng) -> Sắp xếp từ TRÁI sang PHẢI theo trục X
                result_ocr = sorted(result_ocr, key=lambda x: sum([p[0] for p in x[0]]) / 4)
            else:
                # KIỂU 2: Biển số vuông (2 dòng) -> Chia đôi ảnh để tách dòng Trên và dòng Dưới
                mid_y = h / 2
                row1 = []
                row2 = []
                for item in result_ocr:
                    cy = sum([p[1] for p in item[0]]) / 4  # Lấy tâm chữ theo trục Y
                    if cy < mid_y:
                        row1.append(item)
                    else:
                        row2.append(item)
                
                # Sắp xếp từng dòng độc lập từ Trái sang Phải
                row1 = sorted(row1, key=lambda x: sum([p[0] for p in x[0]]) / 4)
                row2 = sorted(row2, key=lambda x: sum([p[0] for p in x[0]]) / 4)
                
                # Gộp dòng 1 trước, dòng 2 sau
                result_ocr = row1 + row2

            # Nối các chuỗi chữ sau khi đã xếp đúng thứ tự đọc chuẩn
            for line in result_ocr:
                text += line[1]  

        # Hậu xử lý chuẩn hóa chuỗi bằng hàm post_process của bạn
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        text = post_process(text)
        
        print(file, "->", text)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{file},{text}\n")
            
    print(f"\nĐã lưu toàn bộ kết quả EasyOCR vào file: {output_file}")

def post_process(st):
    # Chỉ giữ chữ và số
    st = re.sub(r'[^a-zA-Z0-9]', '', st)

    if len(st) <= 7:
        return st

    char_to_num = {
        'A': '4', 'B': '8', 'D': '0',
        'G': '6', 'I': '1', 'O': '0',
        'S': '5', 'Z': '2', 'Q': '0',

        'a': '4', 'b': '8', 'd': '0',
        'g': '6', 'i': '1', 'o': '0',
        's': '5', 'z': '2', 'q': '0'
    }

    num_to_char = {
        '0': 'D', '1': 'I', '2': 'Z',
        '4': 'A', '5': 'S', '6': 'G',
        '8': 'B'
    }

    st_list = list(st)

    # 2 ký tự đầu là số
    for i in range(min(2, len(st_list))):
        if st_list[i] in char_to_num:
            st_list[i] = char_to_num[st_list[i]]

    # ký tự thứ 3 là chữ
    if len(st_list) > 2 and st_list[2] in num_to_char:
        st_list[2] = num_to_char[st_list[2]]

    # 4 ký tự cuối là số
    start = max(0, len(st_list) - 4)

    for i in range(start, len(st_list)):
        if st_list[i] in char_to_num:
            st_list[i] = char_to_num[st_list[i]]

    return "".join(st_list)


def process_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    processed_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # format: image1.jpg,50A101231
        if "," not in line:
            processed_lines.append(line)
            continue

        filename, plate = line.split(",", 1)

        processed_plate = post_process(plate)

        # giữ nguyên tên file và dấu ,
        processed_lines.append(f"{filename},{processed_plate}")

    with open(output_file, "w", encoding="utf-8") as f:
        for line in processed_lines:
            f.write(line + "\n")

    print("Đã xử lý xong:", output_file)

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
    # process_file("answerOCR_EasyOCR.txt", "results3.txt")
    print("thanh cong")

