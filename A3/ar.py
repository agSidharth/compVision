import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import sys


if __name__=='__main__':
    img = cv.imread(sys.argv[1])
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6),None)
    print(corners)

