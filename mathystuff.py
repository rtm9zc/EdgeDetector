from math import exp, pi, sqrt, atan2
from numpy import array, zeros, atleast_2d, amax
from numpy.linalg import eig
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
    gXprime = getGaussian(sigma, kernelSize, True, 'x')
    gY = getGaussian(sigma, kernelSize, False, 'y')
    gYprime = getGaussian(sigma, kernelSize, True, 'y')
    return(convolve2d(convolve2d(image, gXprime, 'same', 'symm')
                      , gY, 'same', 'symm'), 
           convolve2d(convolve2d(image, gX, 'same', 'symm')
                      , gYprime, 'same', 'symm'))

def edgeStrength(gradient):
    outArray = zeros(gradient[0].shape)
    for row in range(0, outArray.shape[0]):
        for col in range(0, outArray.shape[1]):
            xGrad = gradient[0][row][col]
            yGrad = gradient[1][row][col]
            outArray[row][col] = sqrt(xGrad*xGrad + yGrad*yGrad)
    return outArray

def normalOrientation(gradient):
    outArray = zeros(gradient[0].shape)
    for row in range(0, outArray.shape[0]):
        for col in range(0, outArray.shape[1]):
            xGrad = gradient[0][row][col]
            yGrad = gradient[1][row][col]
            outArray[row][col] = atan2(yGrad, xGrad)
    return outArray
    
def squareDirections(orientation):
    outArray = zeros(orientation.shape)
    for row in range(0, outArray.shape[0]):
        for col in range(0, outArray.shape[1]):
            outArray[row][col] = squareAngle(orientation[row][col])
    return outArray

def squareAngle(angle):
    if (angle < -7*pi/8.0): 
        return 0
    if (angle < -5*pi/8.0):
        return 45
    if (angle < -3*pi/8.0):
        return 90
    if (angle < -1*pi/8.0):
        return 45
    if (angle < pi/8.0):
        return 0
    if (angle < 3*pi/8.0):
        return 45
    if (angle < 5*pi/8.0):
        return 90
    if (angle < 7*pi/8.0):
        return 135
    return 0

def edgeSuppress(edges, directions):
    outArray = zeros(edges.shape)
    for row in range(0, outArray.shape[0]):
        for col in range(0, outArray.shape[1]):
            angle = directions[row][col]
            outArray[row][col] = check4Edge(edges, angle, row, col)
    return outArray

def check4Edge(edges, angle, row, col):
    if (angle == 0):
        if (col == 0):
            if (edges[row][col] < edges[row][col+1]):
                return 0
            return edges[row][col]
        if (col == edges.shape[1]-1):
            if (edges[row][col] < edges[row][col-1]):
                return 0
            return edges[row][col]
        if ((edges[row][col] < edges[row][col+1]) or (edges[row][col] < edges[row][col-1])):
            return 0
        return edges[row][col]
    elif (angle == 45):
        if ((row == 0 and col == 0) or 
            (row == edges.shape[0]-1 and col == edges.shape[1]-1)):
            return 0
        if (col == edges.shape[1]-1 or row == 0):
            if (edges[row][col] < edges[row+1][col-1]):
                return 0
            return edges[row][col]
        if (col == 0 or row == edges.shape[0]-1):
            if (edges[row][col] < edges[row-1][col+1]):
                return 0
            return edges[row][col]
        if ((edges[row][col] < edges[row-1][col+1]) or (edges[row][col] < edges[row+1][col-1])):
                return 0
        return edges[row][col]
    elif (angle == 90):
        if (row == 0):
            if (edges[row][col] < edges[row+1][col]):
                return 0
            return edges[row][col]
        if (row == edges.shape[0]-1):
            if (edges[row][col] < edges[row-1][col]):
                return 0
            return edges[row][col]
        if ((edges[row][col] < edges [row-1][col]) or (edges[row][col] < edges[row+1][col])):
            return 0
        return edges[row][col]
    elif (angle == 135):
        if ((row == 0 and col == edges.shape[1]-1) or 
            (row == edges.shape[0]-1 and col == 0)):
            return 0
        if (row == 0 or col == 0):
            if (edges[row][col] < edges[row+1][col+1]):
                return 0
            return edges[row][col]
        if (row == edges.shape[0]-1 or col == edges.shape[1]-1):
            if (edges[row][col] < edges[row-1][col-1]):
                return 0
            return edges[row][col]
        if ((edges[row][col] < edges[row-1][col-1]) or (edges[row][col] < edges[row+1][col+1])):
            return 0
        return edges[row][col]
    else:
        print "ooooops"
        return

def edgeThreshold(edges, directions, t_high, t_low):
    visited = zeros(edges.shape, dtype=bool)
    outArray = zeros(edges.shape)
    for row in range(0, visited.shape[0]):
        for col in range(0, visited.shape[1]):
            if (visited[row][col]):
                continue
            if (edges[row][col] > t_high):
                outArray[row][col] = edges[row][col]
                chain(outArray, edges, row, col, 
                      directions, t_low, visited)
            visited[row][col] = True
    return outArray

def chain(outArray, edges, row, col, directions, t_low, visited):
    if ((outArray[row][col] != 0) and visited[row][col]):
        return True
    return False
    visited[row][col] = True
    if (edges[row][col] < t_low):
        return
    outArray[row][col] = edges[row][col]
    direction = directions[row][col]
    if (direction == 0):
        if (col > 0):
            chain(outArray, edges, row, col-1, directions, t_low, visited)
        if (col < edges.shape[1] -1):
            chain(outArray, edges, row, col+1, directions, t_low, visited)
    if (direction == 45):
        if (row > 0 and col < edges.shape[1]-1):
            chain(outArray, edges, row-1, col+1, directions, t_low, visited)
        if (row < edges.shape[0]-1 and col > 0):
            chain(outArray, edges, row+1, col-1, directions, t_low, visited)
    if (direction == 90):
        if (row > 0):
            chain(outArray, edges, row-1, col, directions, t_low, visited)
        if (row < edges.shape[0]-1):
            chain(outArray, edges, row+1, col, directions, t_low, visited)
    if (direction == 135):
        if (row > 0 and col > 0):
            chain(outArray, edges, row-1, col-1, directions, t_low, visited)
        if (row < edges.shape[0]-1 and col < edges.shape[1]-1):
            chain(outArray, edges, row-1, col-1, directions, t_low, visited)
    else:
        print "oh no"
    return

def normalize(array):
    maxval = amax(array)
    for row in range(0, array.shape[0]):
        for col in range(0, array.shape[1]):
            array[row][col] = array[row][col]/maxval
    return

def eiganImage(gradient, windowSize):
    out = zeros(gradient[0].shape, dtype=float)
    dimensions = gradient[0].shape
    for row in range(0, dimensions[0]-windowSize):
        for col in range(0, dimensions[1]-windowSize):
            neighborhood = covarianceMat(gradient, windowSize, row, col)
            eigenvals, eigenvectors = eig(neighborhood)
            minEigan = min(eigenvals)
            out[row+(windowSize-1)/2][col+(windowSize-1)/2] = minEigan
    return out

def eigan2Points(eiganImage, windowSize):
    out = []
    offset = (windowSize-1)/2
    eiganThresh = amax(eiganImage)/4.0
    for row in range(0, eiganImage.shape[0]):
        for col in range(0, eiganImage.shape[1]):
            if (eiganImage[row][col] > eiganThresh):
                out.append((eiganImage[row][col], row-offset, col-offset))
    return out

def covarianceMat(gradient, windowSize, row, col):
    xsquared = 0
    xy = 0
    ysquared = 0
    for i in range(row, row+windowSize):
        for j in range(col, col+windowSize):
            xsquared = xsquared + gradient[0][i][j]**2
            xy = xy + gradient[0][i][j]*gradient[1][i][j]
            ysquared = ysquared + gradient[1][i][j]**2
    return array([[xsquared, xy] , [xy, ysquared]])

def cornerSuppress(cornerList, windowSize, imgShape):
    cornerList.sort(reverse=True)
    covered = zeros(imgShape, dtype=bool)
    out = []
    threshold = cornerList[0][0]/4.0
    for point in cornerList:
        if (point[0] < threshold):
            continue
        row = point[1]
        col = point[2]
        if not covered[row][col]:
            out.append(point)
            for i in range(row - windowSize, row + windowSize):
                for j in range(col - windowSize, col + windowSize):
                    if ((i >= 0) and (j >= 0) and (i < covered.shape[0]) and (j < covered.shape[1])):
                        covered[i][j] = True
    return out

