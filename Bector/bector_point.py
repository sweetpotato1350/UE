import cv2
import numpy as np

# 이미지 두 장 불러오기
img1 = cv2.imread(r'C:\Users\82102\PycharmProjects\robot\Depth-Anything\assets\dataset\test_grayscaled\point1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r'C:\Users\82102\PycharmProjects\robot\Depth-Anything\assets\dataset\test_grayscaled\point2.jpg', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError("이미지를 제대로 불러오지 못했습니다.")

# ORB 특징점 추출기 생성
orb = cv2.ORB_create()

# 특징점 검출 및 디스크립터 계산
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 매칭 (Brute Force + 해밍 거리)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 매칭 결과 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 결과 시각화
img_matches = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

for m in matches[:50]:  # 상위 50개만 시각화
    pt1 = tuple(map(int, kp1[m.queryIdx].pt))
    pt2 = tuple(map(int, kp2[m.trainIdx].pt))
    cv2.arrowedLine(img_matches, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

# 결과 출력
cv2.imshow("Feature Point Matches (ORB)", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과 이미지 저장 (여기 경로 수정 가능)
save_path = r'C:\Users\82102\PycharmProjects\robot\Depth-Anything\assets\dataset\test_grayscaled\orb_matches_result.png'
cv2.imwrite(save_path, img_matches)
print(f"결과 이미지가 저장되었습니다: {save_path}")
