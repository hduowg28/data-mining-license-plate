from ultralytics import YOLO
from PIL import Image

model = YOLO("c:/data mining/best.pt")
results = model.predict(source="c:/data mining/dataset_license_plate/test/images/AQUA5_66266_checkoutex_2020-10-26-13-296niXCvE7pq_jpg.rf.133f4d36c19a466572b57eb0a0f76ad5.jpg", save=True, conf=0.25)

for result in results:
    print (result.boxes)
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print("tọa độ:", x1, y1, x2, y2)
# #display results
# for r in results:
#     im_array = r.plot() #returns an OpenCV image array with the predictions plotted on it
#     im = Image.fromarray(im_array[..., ::-1]) #convert to PIL image
#     im.show()
#     im.save("c:/data mining/predicted_test.png")