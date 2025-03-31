import cv2

camera1 = cv2.VideoCapture(1)
while True:
    ret1, frame1 = camera1.read()
    cv2.imshow('frame1', frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera1.release()

cv2.destroyAllWindows()