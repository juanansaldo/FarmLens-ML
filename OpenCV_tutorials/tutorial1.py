import cv2
import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(x + y)

img1 = cv2.imread("assets/strawberry.jpeg", cv2.IMREAD_COLOR)     # -1
img2 = cv2.imread("assets/strawberry.jpeg", cv2.IMREAD_GRAYSCALE) # 0
img3 = cv2.imread("assets/strawberry.jpeg", cv2.IMREAD_UNCHANGED) # 1

cv2.imshow("Strawberry", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img1_resized_v1 = cv2.resize(img1, (600, 400))
cv2.imshow("Strawberry (resized)", img1_resized_v1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img1_resized_v2 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
cv2.imshow("Strawberry (resized v2)", img1_resized_v2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img1_resized_v3 = cv2.resize(img1, (0,0), fx=2, fy=2)
cv2.imshow("Strawberry (resized v3)", img1_resized_v3)
cv2.waitKey(0)
cv2.destroyAllWindows()

img1_rotated_v1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("Strawberry (rotated v1)", img1_rotated_v1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img1_rotated_v2 = cv2.rotate(img1, cv2.ROTATE_180)
cv2.imshow("Strawberry (rotated v2)", img1_rotated_v2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img1_rotated_v3 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("Strawberry (rotated v3)", img1_rotated_v3)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("assets/resized_strawberry.jpeg", img1_resized_v2)
cv2.imwrite("assets/rotated_strawberry.jpeg", img1_rotated_v3)