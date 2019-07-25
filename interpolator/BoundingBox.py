"""
Bounding Box class for the interpolator program
Author: Michael Ji
July 15, 2019
"""

class BoundingBox(object):
    def __init__ (self, label, points):
        self.label = label
        self.line_color = None
        self.fill_color = None
        self.pointMajor = points[0]
        self.pointMinor = points[1]
        self.shape_type = 'rectangle'
        self.flags = {}

    @property
    def width(self):
        return abs(self.pointMajor[0] - self.pointMinor[0])

    @property
    def height(self):
        return abs(self.pointMinor[1] - self.pointMajor[1])

    @property
    def center(self):
        return [self.pointMajor[0] + self.width/2, self.pointMajor[1] + self.height/2]

    # Create dictionary version of bounding box for export to json
    def dictVersion(self):
        BBox = {'label': self.label, 'line_color': self.line_color, 'fill_color': self.fill_color,
                'points': [self.pointMajor, self.pointMinor], 'shape_type': self.shape_type, 'flags': self.flags}
        return BBox

    def __str__(self):
        return "Bounding Box " + str(self.pointMajor) + " " + str(self.pointMinor)