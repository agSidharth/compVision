import cv2
import sys

def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')

img = cv2.imread(sys.argv[1])

cv2.namedWindow('Coordi')

cv2.setMouseCallback('Coordi', click_event)

while True:
   cv2.imshow('Coordi',img)
   k = cv2.waitKey(1) & 0xFF
   if k == 27:
      break
cv2.destroyAllWindows()