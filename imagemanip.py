from numpy import zeros, amax, amin
from skimage import img_as_float
import skimage, skimage.io
from copy import deepcopy
import os, mathystuff

def grad2rgb(image):
    outImage = zeros((image.shape[0], image.shape[1], 3), dtype=float)
    imageMax = amax(image)
    imageMin = amin(image)
    if (imageMax+imageMin < 0):
        imageMax = imageMin*(-1)
    for row in range(0, image.shape[0]):
        for col in range(0, image.shape[1]):
            if (image[row][col] > 0):
                outImage[row][col] = [0.0, image[row][col]/imageMax, 0.0]
            elif (image[row][col] < 0):
                outImage[row][col] = [(-1)*image[row][col]/imageMax, 0.0, 0.0]
            else:
                outImage[row][col] = [0.0, 0.0, 0.0]
    return outImage

def addCorners(image, points, windowSize):
    imageOut = deepcopy(image)
    for point in points:
        row = point[1]
        col = point[2]
        for i in range(row, row + windowSize):
            if ((i == row) or (i == row + windowSize - 1)):
                for j in range(col, col + windowSize):
                    imageOut[i][j] = 1.0
            if ((i == row + 1) or (i == row + windowSize - 2)):
                imageOut[i][col] = 1.0
                imageOut[i][col + windowSize - 1] = 1.0
                for j in range(col + 1, col + windowSize -1):
                    imageOut[i][j] = 0.0
            else:
                imageOut[i][col] = 1.0
                imageOut[i][col + windowSize - 1] = 1.0
                imageOut[i][col + 1] = 0.0
                imageOut[i][col + windowSize - 2] = 0.0
    return imageOut

def procAndWrite(filename, sigma, kernelWindow, t_high, t_low, cornerWindow):
    image = skimage.img_as_float(skimage.io.imread(filename))
    folder = os.path.splitext(filename)[0]
    try:
        os.stat(folder)
    except:
        os.mkdir(folder)
    folder = folder + '/'
    skimage.io.imsave(folder + 'normal.jpg', image)
    greyimage = skimage.color.rgb2grey(image)
    skimage.io.imsave(folder + 'grey.jpg', greyimage)
    gradient = mathystuff.convolvedGradient(greyimage, sigma, kernelWindow)
    xGrad = grad2rgb(gradient[0])
    yGrad = grad2rgb(gradient[1])
    skimage.io.imsave(folder + 'xGrad.jpg', xGrad)
    skimage.io.imsave(folder + 'yGrad.jpg', yGrad)
    edgeStrength = mathystuff.edgeStrength(gradient)
    mathystuff.normalize(edgeStrength)
    skimage.io.imsave(folder + 'edgeStrength.jpg', edgeStrength)
    orientation = mathystuff.normalOrientation(gradient)
    dStar = mathystuff.squareDirections(orientation)
    suppressed = mathystuff.edgeSuppress(edgeStrength, dStar)
    skimage.io.imsave(folder + 'suppressed.jpg', suppressed)
    thresholded = mathystuff.edgeThreshold(suppressed, dStar, 
                                           t_high, t_low)
    skimage.io.imsave(folder + 'threshold.jpg', thresholded)
    eiganImage = mathystuff.eiganImage(gradient, cornerWindow)
    mathystuff.normalize(eiganImage)
    skimage.io.imsave(folder + 'eigan.jpg', eiganImage)
    cornerList = mathystuff.eigan2Points(eiganImage, cornerWindow)
    suppressedList = mathystuff.cornerSuppress(cornerList, cornerWindow, 
                                               greyimage.shape)
    imagecornered = addCorners(image, suppressedList, cornerWindow)
    skimage.io.imsave(folder + 'corners.jpg', imagecornered)
    return
