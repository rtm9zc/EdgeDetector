import skimage
import skimage.io, skimage.transform
import matplotlib.pyplot as pyplot

def readImage(filepath):
    return skimage.img_as_float(skimage.io.imread(filepath))

def displayImage(image):
    skimage.io.imshow(image)
    skimage.io.show()
    return

def square(image):
    dimensions = widthHeight(image)
    if dimensions[0] > dimensions[1]: # width higher, scale down width
        return skimage.transform.resize(image, (dimensions[1], dimensions[1]))
    else:
        return skimage.transform.resize(image, (dimensions[0], dimensions[0]))

def grey(image):
    return skimage.color.rgb2grey(image)

def grey2float(greyImage):
    return skimage.color.gray2rgb(image)

def displayFloatGrey(floatImage):
    displayImage(grey(floatImage))
    return
    
def plotIntensity(image, row):
    greyImage = grey(image)
    intensities = []
    for i in range(0, widthHeight(greyImage)[0]):
        intensities.append(greyImage[row][i])
    pyplot.plot(intensities)
    pyplot.show()
    return

def widthHeight(image):
    return(image.shape[1], image.shape[0])

def zeroPix(image):
    dims = widthHeight(image)
    greyImage = grey(image)
    for x in range(10, dims[1], 10):
        for y in range(20, dims[0], 20):
            greyImage[x][y] = 0
    return greyImage

def maxrow(image, row):
    maxbright = 0
    maxDex = -1
    greyImage = grey(image)
    for i in range(0, widthHeight(image)[0]):
        if (image[row][i] > maxbright).all():
            maxbright = image[row][i]
            maxDex = i
    return maxDex

def write(image, filepath):
    skimage.io.imsave(filepath, image)
    return

image = readImage('Termina.png')
displayImage(image)
displayImage(square(image))
greyImage = grey(image)
displayImage(greyImage)
displayImage(grey2float(greyImage))
displayFloatGrey(image)
plotIntensity(image, 67)
print widthHeight(image)
displayImage(zeroPix(image))
print maxrow(image, 12)
write(square(image), 'termina2.png')
