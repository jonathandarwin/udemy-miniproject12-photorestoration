import cv2
import numpy as np


# load gambar yang rusak
image = cv2.imread('abraham.jpg')
cv2.imshow('Original Damaged Photo', image)
cv2.waitKey(0)

# load gambar yang sudah di mark bagian yang rusak
marked_damages = cv2.imread('mask.jpg', 0)
cv2.imshow('Marked Damages', marked_damages)
cv2.waitKey(0)

# kasih threshold biar keliatan jelas bagian yang rusak
ret, thresh1 = cv2.threshold(marked_damages, 254, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', thresh1)
cv2.waitKey(0)

# gambar di dilate biar bagian yang rusak terlihat makin lebar
kernel = np.ones((7,7), np.uint8)
mask = cv2.dilate(thresh1, kernel, iterations = 1)
cv2.imshow('Dilated Mask', mask)
cv2.imwrite('images/abraham_mask.png', mask)

cv2.waitKey(0)
# fungsi utama yang digunakan untuk photo restoration
restored = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

cv2.imshow('Restored', restored)
cv2.waitKey(0)
cv2.destroyAllWindows()