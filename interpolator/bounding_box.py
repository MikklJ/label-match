"""
Bounding Box class for the interpolator program (interpolator.py)
Author: Michael Ji
July 15, 2019

Initialization requires a label and a 2-element list with two floats in each element
These floats represent the upper-left (major) point and bottom-right (minor) point
Coordinate system follows PIL format --- (0, 0) is the top-left corner of an image/box
"""

class bounding_box(object):
    def __init__ (self, label, points):
        self.label = label
        self.line_color = None
        self.fill_color = None
        self.pointMajor = points[0] # The upper-left point should be represented by floats [x, y]
        self.pointMinor = points[1] # The bottom-right point should be represented by floats [x, y]
        self.shape_type = 'rectangle'
        self.flags = {}

    @property
    def width(self):
        return abs(self.pointMajor[0] - self.pointMinor[0])

    @property
    def height(self):
        return abs(self.pointMinor[1] - self.pointMajor[1])

    # Get the linear center of the bounding box
    @property
    def center(self):
        return [self.pointMajor[0] + self.width/2, self.pointMajor[1] + self.height/2]

    # Create dictionary version of bounding box for export to json files
    def dictVersion(self):
        BBox = {'label': self.label, 'line_color': self.line_color, 'fill_color': self.fill_color,
                'points': [self.pointMajor, self.pointMinor], 'shape_type': self.shape_type, 'flags': self.flags}
        return BBox

    def __str__(self):
        return "Bounding Box " + str(self.pointMajor) + " " + str(self.pointMinor)