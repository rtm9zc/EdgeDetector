import skimage, skimage.io
import mathystuff

image = skimage.img_as_float(skimage.io.imread('imagetest/lines.png'))
skimage.io.imshow(image)
skimage.io.show()
greyimage = skimage.color.rgb2grey(image)
skimage.io.imshow(greyimage)
skimage.io.show()
gradient = mathystuff.convolvedGradient(greyimage, 2.0, 9)
skimage.io.imshow(skimage.color.gray2rgb(gradient[0]))
skimage.io.show()
skimage.io.imshow(gradient[1])
skimage.io.show()
print gradient[0]
