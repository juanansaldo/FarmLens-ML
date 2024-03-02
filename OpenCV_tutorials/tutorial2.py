import cv2
import random
import numpy as np

img = cv2.imread("assets/strawberry.jpeg", -1)

print(img)
print(type(img))
print(img.shape)
print(img[257][45:400])
print(img[257][400])

for i in range(100):
    for j in range(img.shape[1]):
        img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

cv2.imshow("Image", img)
cv2.waitKey(0)

tag = img[100:300, 200:500]
img[200:400, 100:400] = tag

cv2.imshow("Image", img)
cv2.waitKey(0)

cv2.destroyAllWindows()