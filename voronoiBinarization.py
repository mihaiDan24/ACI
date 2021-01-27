import numpy as np
import cv2 as cv2

def main():  
    print('Reading image...')
    # Read in the image.
    img = cv2.imread('Doc.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Generate SURF image
    surfSubdiv2 = getSurfSubdiv2(gray)
    surfVoronoi = drawVoronoi(gray, surfSubdiv2)
    
    print('Finished drawing voronoi...')
    cv2.imwrite('gray.jpg',gray)
    cv2.imwrite('SURF.jpg',surfVoronoi)

def drawVoronoi(img, subdiv): 
    voronoi = np.zeros(img.shape, dtype = img.dtype)
    binPoly = np.zeros(img.shape, dtype = img.dtype)
    
    #Use subdiv.getVoronoiFacetList to get the list of Voronoi facets
    ( facets, centers) = subdiv.getVoronoiFacetList([])
 
    for facetsIndex in range(0,len(facets)):
        # Generate array of polygon corners
        facetArray = []
        for facet in facets[facetsIndex] :
            
            facetArray.append(facet)
        
        # Get average color of polygon from original image
        mask = np.zeros(img.shape[:2], np.uint8)
        
        polygon = cv2.fillPoly(mask, np.int32([facetArray]), (255,255,255));
        cv2.imwrite('polygon.jpg',polygon)
        
        #Copy initial info to the created polygons
        newPoly = cv2.bitwise_and(img, polygon)
        
        #Apply binarization for each polygon
        ret, ad_th1 = cv2.threshold(newPoly,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #Add each computed polygon to the final image
        binPoly = cv2.add(binPoly,ad_th1)
        cv2.imwrite('ad_th1.jpg',ad_th1)
        # Fill polygon with average color
        intFacet = np.array(facetArray, np.int)
        
        # Draw lines around polygon
        polyFacets = np.array([intFacet])
        cv2.polylines(voronoi, polyFacets, True, (255, 255, 255), 1, cv2.LINE_AA, 5) 
    
    #Final image
    cv2.imwrite('binPoly.jpg',binPoly)
    return voronoi

def getSurfSubdiv2(img):
    # Set Hessian Threshold value
    hessianThreshold = 8000
    
    # Create SURF object
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold)
    
    # Set the Upright flag to TRUE => no need to find the orientation
    surf.setUpright(True)
    
    # Find keypoints and descriptors directly
    kp, des = surf.detectAndCompute(img,None)
    
    #This method converts vector of keypoints to vector of points or the reverse, where each keypoint is assigned the same size and the same orientation.
    keyPoints = cv2.KeyPoint_convert(kp)
    points = []
    for keyPoint in keyPoints: 
        points.append((keyPoint[0], keyPoint[1]))
    
    size = img.shape
    #Define the space you want to partition using a rectangle (rect). 
    #If the points you have defined in the previous step are defined on an image, this rectangle can be ( 0, 0, width, height ). 
    #Otherwise, you can choose a rectangle that bounds all the points
    subdiv2DShape = (0, 0, size[1], size[0])
    
    #Create an instance of Subdiv2D with the rectangle obtained in the previous step
    subdiv = cv2.Subdiv2D(subdiv2DShape);
    for p in points :
        subdiv.insert(p)
        
    return subdiv

if __name__ == '__main__':
    main()
