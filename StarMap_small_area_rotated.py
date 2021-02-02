import cv2
import numpy as np
import imutils

img = cv2.imread("StarMap.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template2 = cv2.imread("Small_area_rotated.png", 0)

threshold = 0.9

for angle in np.arange(0,360, 90):
    rotated = imutils.rotate(template2, angle)
    w2, h2 = rotated.shape[::-1]

    result2 = cv2.matchTemplate(img_gray, rotated, cv2.TM_CCOEFF_NORMED)
    loc2 = np.where(result2 >= threshold)

    if loc2:
        cv2.circle(img, loc2[::-1], 3, (0, 0, 255), -1)
        cv2.circle(img, (loc2[::-1][0] + w2, loc2[::-1][1]), 3, (0, 0, 255), -1)
        cv2.circle(img, (loc2[::-1][0], loc2[::-1][1] + h2), 3, (0, 0, 255), -1)
        cv2.circle(img, (loc2[::-1][0] + w2, loc2[::-1][1] + h2), 3, (0, 0, 255), -1)



cv2.imshow("Star Map", img)
cv2.waitKey(0)
cv2.destroyAllWindows()