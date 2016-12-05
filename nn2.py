import cv2
import re
import numpy as np
import random
from pprint import pprint


charFindCount = 41
chars = []
count = 0
dataset = open('newdataset.txt', 'r').read().split(",")
for char in range(charFindCount):
	chars.append(np.zeros((28,28), dtype=np.uint8))
	for pxCol in range(28):
		for pxRow in range(28):
			chars[char][pxCol][pxRow] = dataset[count]
			count += 1

for i in range(30):
	cv2.imshow("frame"+str(i),chars[i])

cv2.waitKey(0)
cv2.destroyAllWindows()