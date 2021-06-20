def mini_batch(X, mini_batch_size=8):
    for i in range(0, len(X), mini_batch_size):
        yield X[i:i+mini_batch_size]

#XY_batch_ = list(zip(mini_batch(my_dataset.data), mini_batch(my_dataset.exp_out)))

'''
TEST TEXT READ
import cv2 as cv

resim = cv.imread('SS2.png', 0)
resim = 255 - resim
#_, resim = cv.threshold(resim,125,255,cv.THRESH_BINARY)
resim = cv.adaptiveThreshold(resim, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -10)
cv.imshow('denem', resim)
cv.waitKey(-1)
#resim = cv.cvtColor(resim, cv.COLOR_BGR2GRAY)
'''
