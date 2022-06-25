import cv2

def UpdateAdaptive(num):
    blockSize = cv2.getTrackbarPos('Thresh', 'Threshold')
    type = cv2.getTrackbarPos('1:Mean 2:Gaussian', 'Threshold')
    if type==0:
        tval=cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        tval = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

    if blockSize<=0:
        blockSize=1
    blockSize=blockSize*2+1

    outImage = cv2.adaptiveThreshold(img, 255, tval, cv2.THRESH_BINARY,
                                     blockSize, 2)
    cv2.imshow('Threshold', outImage)


if __name__ == '__main__':
    cv2.namedWindow('Threshold',1)
    cv2.createTrackbar('Thresh', 'Threshold', 1, 255, UpdateAdaptive)
    cv2.createTrackbar('1:Mean 2:Gaussian', 'Threshold', 0, 1, UpdateAdaptive)

    img = cv2.imread('test_images/gaussianBlur_rice.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    UpdateAdaptive(0)
    cv2.waitKey(0)


cv2.destroyAllWindows()