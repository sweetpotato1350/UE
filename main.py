import cv2
import torch

# YOLOv5 모델을 불러옵니다.
model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)

# 웹캠 캡처 객체를 생성합니다.
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    # 이미지를 YOLOv5 모델에 입력할 형식으로 변환합니다.
    results = model(img)

    # 탐지된 객체 정보를 얻습니다.
    detections = results.xyxy[0]

    # 탐지된 객체 주변에 경계 상자와 레이블을 표시합니다.
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        label = f"{results.names[int(cls)]}: {conf:.2f}"
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        img = cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 영상을 좌우로 반전시킵니다.
    img = cv2.flip(img, 1)

    # 영상을 보여줍니다.
    cv2.imshow("Object Detection", img)

    # 'q' 키를 누르면 루프를 종료합니다.
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()