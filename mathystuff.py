from math import exp, pi, sqrt
from numpy import array, atleast_2d
from scipy.signal import convolve2d

def getGaussian(sigma, kernelSize, deriv, dimension): #deriv is a bool, 
    if not(dimension == 'x' or dimension == 'y'): #dimension is char x or y
        print "Invalid dimension spec for getGaussian!" 
        return
    if type(deriv) is not bool:
        print "deriv must be a boolean value in getGaussian!"
        return
    if dimension is 'x':
        return buildXGauss(sigma, kernelSize, deriv)
    else:
        return buildYGauss(sigma, kernelSize, deriv)
    
def buildXGauss(sigma, kernelSize, deriv):
    out = []
    mean = (kernelSize-1)/2.0
    for i in range(0, kernelSize):
        if deriv:
            out.append(derivGaussian(i-mean, sigma))
        else:
            out.append(gaussian(i-mean, sigma))
    return atleast_2d(array(out))

def buildYGauss(sigma, kernelSize, deriv):
    out = []
    mean = (kernelSize-1)/2.0
    for i in range(0, kernelSize):
        if deriv:
            out.append([derivGaussian(i-mean, sigma)])
        else:
            out.append([gaussian(i-mean, sigma)])
    return atleast_2d(array(out))

def gaussian(x, sigma):
    return((exp(-1*((x*x)/(2*sigma*sigma))))/(sigma*(sqrt(2*pi))))

def derivGaussian(x, sigma):
    return (-1*x*gaussian(x, sigma))/(sigma*sigma)

def convolvedGradient(image, sigma, kernelSize):
    gX = getGaussian(sigma, kernelSize, False, 'x')
    print gX
    gXprime = getGaussian(sigma, kernelSize, True, 'x')
    print gXprime
    gY = getGaussian(sigma, kernelSize, False, 'y')
    print gY
    gYprime = getGaussian(sigma, kernelSize, True, 'y')
    print gYprime
    return(convolve2d(convolve2d(image, gXprime, 'same', 'symm')
                      , gY, 'same', 'symm'), 
           convolve2d(convolve2d(image, gX, 'same', 'symm')
                      , gYprime, 'same', 'symm'))
