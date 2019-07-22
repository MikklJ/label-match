"""
Label applier for images
Author: Michael Ji
July 16, 2019
"""

import cv2
import os
import json

# Root folder of the image files
root = "/home/michael/datasets/michael_processed/"
totalCounter = 1
doneCounter = 1

outputLabels = {}


class GetOutOfLoop(Exception):
    pass


try:
    for dir in os.listdir(root)[::-1]:
        for image in os.listdir(root + dir):
            print("\nIMAGES LABELED:", doneCounter)
            print("\nImage", totalCounter, ":", dir + "/" + image)
            totalCounter += 1
            label = ""
            img = cv2.imread(root + dir + "/" + image)
            # img_scaled = cv2.resize(img, (1000, 1000))
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 300, 300)

            cv2.imshow('image', img)
            k = cv2.waitKey(1)

            label = input("Label: ")
            if label == "ESC":
                raise GetOutOfLoop
            elif label != "":
                print("Final Label =", label)
                outputLabels.update({root + dir + "/" + image: label})
                doneCounter += 1
            else:
                print("SKIPPED", root + dir + "/" + image)
except GetOutOfLoop:
    pass

print("\nDict values:")
for key, value in outputLabels.items():
    print(key, "with Label:", value)

with open("labels2.json", 'w') as json_file:
    json.dump(outputLabels, json_file)
