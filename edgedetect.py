def cannyEdgeDetector(image):
    return hysteresisThresh(nonmaxSuppression(cannyEnhancer(image)))

#def hysteresisThresh(image):

#def nonmaxSuppression(image):

#def cannyEnhancer(image):
