"""
This file takes labeled image directories from labels_corrected and deletes their corresponding image directories
in a clone of michael_processed (michael_processed_left)

NOTE: The counting system used is faulty, and directories need to be adjusted
"""

import os
import json
import shutil

rmcounter = 1
failedtoremove = 1
root = "/home/michael/datasets/michael_processed_left/"
labeled_set = {}

removed_list = []
left_list = []

with open("labels_corrected.json", 'r') as json_file:
    labeled_set = json.load(json_file)
    
print("Already labeled", len(labeled_set.keys()), "images")

for key in labeled_set:
    path = root + key.rsplit("/", 2)[1]
    #print(os.path.exists(root + key.rsplit("/", 2)[1] ))
    try:
        shutil.rmtree(path)
        rmcounter += 1
        removed_list.append(path)
    except FileNotFoundError:
        failedtoremove += 1
        left_list.append(path)
            
    
print(rmcounter, "image directories removed from", root)
for e in removed_list:
    print(e)
print("failed to remove", failedtoremove)
for e in left_list:
    print(e)