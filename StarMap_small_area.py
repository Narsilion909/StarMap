import cv2
import numpy as np

img = cv2.imread("StarMap.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template1 = cv2.imread("Small_area.png", 0)

threshold = 0.9

w1, h1 = template1.shape[::-1]

result1 = cv2.matchTemplate(img_gray, template1, cv2.TM_CCOEFF_NORMED)
loc1 = np.where(result1 >= threshold)

cv2.circle(img, loc1[::-1], 3, (0,0,255), -1)
cv2.circle(img, (loc1[::-1][0] + w1, loc1[::-1][1]), 3, (0,0,255), -1)
cv2.circle(img, (loc1[::-1][0], loc1[::-1][1] + h1), 3, (0,0,255), -1)
cv2.circle(img, (loc1[::-1][0] + w1, loc1[::-1][1] + h1), 3, (0,0,255), -1)



cv2.imshow("Star Map", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

