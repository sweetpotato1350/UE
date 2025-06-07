import cv2
import numpy as np
import math

# 이미지 불러오기
img1 = cv2.imread(r'C:\Users\82102\PycharmProjects\robot\Depth-Anything\assets\dataset\test_grayscaled\point1.jpg',
                  cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r'C:\Users\82102\PycharmProjects\robot\Depth-Anything\assets\dataset\test_grayscaled\point2.jpg',
                  cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError("이미지를 제대로 불러오지 못했습니다.")

# ORB 초기화
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 매칭
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 시각화용 이미지
img_matches = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

# 회전각 및 거리 계산을 위한 누적 변수
angle_sum = 0
magnitude_sum = 0
count = 0

for m in matches[:50]:
    pt1 = np.array(kp1[m.queryIdx].pt)
    pt2 = np.array(kp2[m.trainIdx].pt)

    dx, dy = pt2 - pt1
    magnitude = math.sqrt(dx ** 2 + dy ** 2)
    angle_rad = math.atan2(dy, dx)  # 방향: 라디안
    angle_deg = math.degrees(angle_rad)  # °로 변환

    # 시각화
    cv2.arrowedLine(img_matches, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0, 255, 0), 1, tipLength=0.3)

    # 누적합
    angle_sum += angle_deg
    magnitude_sum += magnitude
    count += 1

# 평균 회전각도 및 거리 추정
if count > 0:
    avg_angle = angle_sum / count  # 평균 회전 각도 (deg)
    avg_magnitude = magnitude_sum / count  # 평균 이동 거리 (px)

    # 거리 추정: 크기가 클수록 가까움 (간이 역비례)
    approx_distance = 1 / (avg_magnitude + 1e-6)  # 0으로 나누지 않도록 작은 값 추가
else:
    avg_angle = 0
    approx_distance = float('inf')

# 정보 출력
print(f"평균 회전 각도: {avg_angle:.2f}도")
print(f"평균 이동 거리(픽셀): {avg_magnitude:.2f}")
print(f"추정 상대 거리: {approx_distance:.4f} (크면 멀고, 작으면 가까움)")

# 결과 저장
save_path = r'C:\Users\82102\PycharmProjects\robot\Depth-Anything\assets\dataset\test_grayscaled\orb_matches_result.png'
cv2.imwrite(save_path, img_matches)
print(f"결과 이미지 저장 완료: {save_path}")

cv2.imshow("ORB Match + Vectors", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
