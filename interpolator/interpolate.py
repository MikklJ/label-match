"""
Interpolator program for sequential bounding-box image annotation
Author: Michael Ji
July 15, 2019

This program takes .json files with image labeling information (bounding boxes) for two images in a set of
 sequential video frames and creates bounding box information for all images between the two specified images

Currently using SSIM ratings to match bounding boxes, as SNN approach has some issues

Uses PIL to process images from box information
"""

import json
import sys
import os.path
from bounding_box import bounding_box
import PIL.Image as Image
import numpy as np
import random
import torch
import models
import pytorch_msssim
import SSIM_PIL

# Check command line arguments for two .json label files
if len(sys.argv) == 3:
    json0 = sys.argv[1]
    jsonN = sys.argv[2]
else:
    print("Usage: $python interpolate.py [starting json] [ending json]")
    exit(1)

# Retrieve labels from first image (stored in the file specified by path json0)
with open(json0) as json_file:
    json0data = json.load(json_file)
    json0shapes = json0data["shapes"]

# Retrieve labels from last image (stored in the file specified by path jsonN)
with open(jsonN) as json_file:
    jsonNdata = json.load(json_file)
    jsonNshapes = jsonNdata["shapes"]

# Create and fill lists of bounding_box objects (read from .json)
json0BBox = []
jsonNBBox = []
for box in json0shapes:
    json0BBox.append(bounding_box(box["label"], box["points"]))

for box in jsonNshapes:
    jsonNBBox.append(bounding_box(box["label"], box["points"]))


# Print coordinates of all read boxes
print("Unordered first frame boxes. Number of boxes:", len(json0BBox))
for box in json0BBox:
    print(box)

print("\nUnordered last frame boxes. Number of boxes:", len(jsonNBBox))
for box in jsonNBBox:
    print(box)


# Read Images and initialize 2D probability matrices
json0BBoxProbs = [] # 2D probability matrix if first json file contains more boxes than the last json file
jsonNBBoxProbs = [] # 2D probability matrix if last json file contains more boxes than the first json file
firstFrameImage = Image.open(json0.replace(".json", ".png"))
lastFrameImage = Image.open(jsonN.replace(".json", ".png"))

# Create a (currently nonfunctional) siamese neural network
model = models.SiameseNet(models.SalakhNet)
model.load_state_dict(torch.load("outputs/train/2019-07-30-19-26/label_match_0.0001/models/final.pt", map_location='cpu'))
model.eval()

if len(json0BBox) <= len(jsonNBBox):
    """
    If there are more boxes in frame 0, construct a 2d list of probabilities such that
    json0BBoxProbs = [
        (box0-0) [p_bboxN-0, p_bboxN-1, ... p_bboxN-B]
        (box0-1) [...]
        ...
        (box0-A) [...]
    ]
    where A <= B, A = len(json0BBox), B = len(jsonNBBox)
    """
    print("\nFrame", json0, "has fewer boxes")
    for bbox0 in json0BBox:
        p_list = []
        bounded_image = firstFrameImage.crop(
            (bbox0.pointMajor[0], bbox0.pointMajor[1], bbox0.pointMinor[0], bbox0.pointMinor[1])
        )
        # Alternate box processing if SNN is used
        """
        width_1, height_1 = bounded_image.size
        bounded_image = bounded_image.resize((int(width_1 * (160 / height_1)), 160), resample=Image.NEAREST, box=None)
        width_1, height_1 = bounded_image.size
        bounded_image = bounded_image.crop((0, 0, width_1 - width_1 % 32, height_1))
        image_tensor = np.array(bounded_image).astype(np.float32)
        image_tensor = np.transpose(image_tensor, (2, 0, 1))
        image_tensor = image_tensor/255.0
        image_tensor = np.minimum(image_tensor, 1.0)
        image_tensor = np.array(bounded_image).astype(np.int)
        image_tensor = torch.from_numpy(image_tensor).unsqueeze(0)
        """
        bounded_image = bounded_image.resize((160, 160), resample=Image.NEAREST, box=None)

        for bboxN in jsonNBBox:
            compared_image = lastFrameImage.crop(
                (bboxN.pointMajor[0], bboxN.pointMajor[1], bboxN.pointMinor[0], bboxN.pointMinor[1])
            )
            # Alternate box processing if SNN is used
            """
            width_1, height_1 = compared_image.size
            compared_image = compared_image.resize((int(width_1 * (160 / height_1)), 160), resample=Image.NEAREST,
                                                 box=None)
            width_1, height_1 = compared_image.size
            compared_image = compared_image.crop((0, 0, width_1 - width_1 % 32, height_1))
            compared_tensor = np.array(compared_image).astype(np.float32)
            compared_tensor = np.transpose(compared_tensor, (2, 0, 1))
            compared_tensor = compared_tensor / 255.0
            compared_tensor = np.minimum(compared_tensor, 1.0)
            compared_tensor = np.array(compared_image).astype(np.int)
            compared_tensor = torch.from_numpy(compared_tensor).unsqueeze(0)
            
            p = random.random()  # SiameseNet(image_tensor, compared_tensor)
            p = model(image_tensor, compared_tensor)
            print(image_tensor.shape, compared_tensor.shape)
            p = pytorch_msssim.ms_ssim(image_tensor, compared_tensor)
            """
            compared_image = compared_image.resize((160, 160), resample=Image.NEAREST, box=None)

            p = SSIM_PIL.compare_ssim(bounded_image, compared_image)
            p_list.append(p)

        json0BBoxProbs.append(p_list)

    # Prints 2d probability matrix
    print('\n'.join([''.join(['{:6}'.format(round(item, 2)) for item in row])
                     for row in json0BBoxProbs]))
else:
    """
    If there are more boxes in frame N, construct a 2d list of probabilities such that
    jsonNBBoxProbs = [
        (boxN-0) [p_bbox0-0, p_bbox0-1... p_bbox0-A]
        (boxN-1) [...]
        ...
        (boxN-B) [...]
    ]
    where B > A, with A = len(json0BBox), B = len(jsonNBBox)
    """
    print("\nFrame", jsonN, "has fewer boxes")
    for bboxN in jsonNBBox:
        p_list = []
        bounded_image = lastFrameImage.crop(
            (bboxN.pointMajor[0], bboxN.pointMajor[1], bboxN.pointMinor[0], bboxN.pointMinor[1])
        )
        bounded_image = bounded_image.resize((160, 160), resample=Image.NEAREST, box=None)
        
        # Alternate box processing if SNN is used
        """
        image_tensor = np.array(bounded_image).astype(np.float32)
        image_tensor = np.transpose(image_tensor, (2, 0, 1))
        image_tensor = torch.from_numpy(image_tensor).float().unsqueeze(0)
        """"
        for bbox0 in json0BBox:
            compared_image = firstFrameImage.crop(
                (bbox0.pointMajor[0], bbox0.pointMajor[1], bbox0.pointMinor[0], bbox0.pointMinor[1])
            )
            compared_image = compared_image.resize((160, 160), resample=Image.NEAREST, box=None)
            
            # Alternate box processing if SNN is used
            """
            compared_tensor = np.array(compared_image).astype(np.float32)
            compared_tensor = np.transpose(compared_tensor, (2, 0, 1))
            compared_tensor = torch.from_numpy(compared_tensor).float().unsqueeze(0)
            
            p = random.random()  # SiameseNet(image_tensor, compared_tensor)
            p = model(image_tensor.float(), compared_tensor.float())
            """
            
            p = SSIM_PIL.compare_ssim(bounded_image, compared_image)
            p_list.append(p)

        jsonNBBoxProbs.append(p_list)
    
    # Prints 2d probability matrix
    print('\n'.join([''.join(['{:6}'.format(round(item, 2)) for item in row])
                     for row in jsonNBBoxProbs]))

"""
Here, the program reads through the probability matrix with elements generated by the siamese neural network
It does the "matching" of similar image labels and stores them in 2 "matched" arrays containing bounding boxes
from the two image frames, with the two elements at each index being the matched pair of boxes

e.g. json0BBoxMatched[5] and jsonNBBoxMatched[5] would represent the 6th pair of matched bounding box labels
"""
json0BBoxMatched = []
jsonNBBoxMatched = []

if len(json0BBoxProbs) > 0:
    """Assuming more boxes in frame 0"""

    for index0, bbox0list in enumerate(json0BBoxProbs):
        index_match = np.argmax(bbox0list)

        json0BBoxMatched.append(json0BBox[index0])
        jsonNBBoxMatched.append(jsonNBBox[index_match])

        for i, bbox0list in enumerate(json0BBoxProbs):
            bbox0list[index_match] = -1
            # print("unmatched indices left", len(bbox0list))
            json0BBoxProbs[i] = bbox0list

else:
    """Assuming more boxes in frame N"""

    for indexN, bboxNlist in enumerate(jsonNBBoxProbs):
        index_match = np.argmax(bboxNlist)

        jsonNBBoxMatched.append(jsonNBBox[indexN])
        json0BBoxMatched.append(json0BBox[index_match])

        for i, bboxNlist in enumerate(jsonNBBoxProbs):
            bboxNlist[index_match] = -1
            # print("unmatched indices left", len(bboxNlist))
            jsonNBBoxProbs[i] = bboxNlist

# Assign matched lists of bboxes bacl to original bbox lists (e.g. json0BBox)
json0BBox = json0BBoxMatched
jsonNBBox = jsonNBBoxMatched

# Print coordinates of all boxes after matching
print("\nOrdered first frame boxes. Number of boxes:", len(json0BBox))
for box in json0BBox:
    print(box)

print("\nOrdered last frame boxes. Number of boxes:", len(jsonNBBox))
for box in jsonNBBox:
    print(box)

# Get file names of the first and last .json file (e.g. 0.json to 7.json)
firstJson = os.path.basename(json0)
lastJson = os.path.basename(jsonN)

# Gets number of intermediate frames
numberOfFrames = int(lastJson.split('.')[0]) - int(firstJson.split('.')[0])

# Write to .json files storing labeling information for intermediate frames
# Assumes that box json0BBox[n] and box jsonNBBox[n] describes the same object
for fileNumber in range(int(firstJson.split('.')[0]) + 1, int(lastJson.split('.')[0])):
    # Generate relative path to target .json labels file
    fileName = os.path.dirname(json0) + "/" + str(fileNumber) + ".json"

    # Declare bounding box list to be written to file
    BBoxList = []

    """
    For each bounding box label in frame 0 and its counterpart in frame N, create a bounding box that has box points
    that are somewhere between the points in frame 0 and points in frame N.

    Each point moves a fraction of the distance between the frame 0 point and the frame N point
    (e.g. if there are 6 frames between frame 0 and frame 7, frame 1 has boxes with points moved 1/7 of the distance
    between points in frame 0 and frame 7

    This is a naive linear algorithm that does not take acceleration, minor camera shaking, etc into account.
    """
    for i in enumerate(json0BBox):
        bbox0 = json0BBox[i[0]]
        bboxN = jsonNBBox[i[0]]

        # Calculate bounding box's top-left/major point's coords
        pointMajorXChange = ((fileNumber - int(firstJson.split('.')[0])) / numberOfFrames) * (
                    bboxN.pointMajor[0] - bbox0.pointMajor[0])
        pointMajorYChange = ((fileNumber - int(firstJson.split('.')[0])) / numberOfFrames) * (
                    bboxN.pointMajor[1] - bbox0.pointMajor[1])
        pointMajor = [bbox0.pointMajor[0] + pointMajorXChange, bbox0.pointMajor[1] + pointMajorYChange]

        # Calculate bounding box's bottom_right/minor point's coords
        pointMinorXChange = ((fileNumber - int(firstJson.split('.')[0])) / numberOfFrames) * (
                    bboxN.pointMinor[0] - bbox0.pointMinor[0])
        pointMinorYChange = ((fileNumber - int(firstJson.split('.')[0])) / numberOfFrames) * (
                    bboxN.pointMinor[1] - bbox0.pointMinor[1])
        pointMinor = [bbox0.pointMinor[0] + pointMinorXChange, bbox0.pointMinor[1] + pointMinorYChange]

        newBBox = bounding_box(bbox0.label, [pointMajor, pointMinor])
        BBoxList.append(newBBox.dictVersion())

    # Write bounding box labels to .json files storing labels for the intermediate frame
    try:
        outdata = {}
        with open(fileName, 'r') as outfile:
            outdata = json.load(outfile)
            outdata["shapes"] = BBoxList
            print("Accessed", fileName)
        with open(fileName, 'w') as outfile:
            json.dump(outdata, outfile)
    except FileNotFoundError:
        print("Missing .json:", fileName)
        exit(1)
