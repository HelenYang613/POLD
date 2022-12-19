import numpy as np
from shapely.geometry import Polygon

def intersection(data1, data2):

    poly1 = Polygon(data1).convex_hull
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  
    else:
        inter_area = poly1.intersection(poly2).area  
    return inter_area
    