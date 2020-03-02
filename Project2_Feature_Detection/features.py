import math
import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
import transformations
import pdb

def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True

## Keypoint detectors ##########################################################
class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pindex_xel satisfies this meaningless criterion,
                    # make it a feature.
                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10
                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        # Dimensions of image
        height, width = srcImage.shape[:2]
         # Numpy array containing zeroes of shape of image, to be filled with Harris scores at each pindex_xel
        harrisImage = np.zeros(srcImage.shape[:2])
        # Numpy array containing zeroes of shape of image, to be filled with orientation of gradient at each pindex_xel
        orientationImage = np.zeros(srcImage.shape[:2])
        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN
        # Calculation of sobel image
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html
        index_x = scipy.ndimage.sobel(srcImage, axis=1, output=None, mode='reflect', cval=0.0)
        index_y = scipy.ndimage.sobel(srcImage, axis=0, output=None, mode='reflect', cval=0.0)
        # Implementation of Gaussian mask
        # https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.gaussian_filter.html
        # The elements will be used to derive the determinant, trace, and Harris image
        A = scipy.ndimage.filters.gaussian_filter(index_x**2, sigma=.5)
        B = scipy.ndimage.filters.gaussian_filter(index_y*index_x, sigma=.5)
        C = scipy.ndimage.filters.gaussian_filter(index_y**2, sigma=.5)
        # Derive determinant of Gaussian filtered matrix
        det = A*C-B**2
        # Derive trace of Gaussian filtered matrix
        trace = A + C
        # Derive Harris image
        harrisImage = det - 0.1*(trace**2)
        # Derive orientation of image
        orientationImage = np.arctan2(index_y,index_x)*180/np.pi
        return harrisImage, orientationImage
        # TODO-BLOCK-END
        
    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pindex_xel.
        Output:
            destImage -- numpy array containing True/False at
                         each pindex_xel, depending on whether
                         the pindex_xel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        # Creates an array of zeroes same shape as Harris image of type 'a'
        destImage = np.zeros_like(harrisImage, np.bool)
        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        # Filters the input image wth the maximim fulter to find local maxima
        # 7x7 size specified in prompt
        local_max = scipy.ndimage.filters.maximum_filter(harrisImage, size=(7,7))
        destImage = (harrisImage == local_max)
        # TODO-BLOCK-END
        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        # Convert image matrix to float
        image = image.astype(np.float32)
        # Convert to BGR scale
        image /= 255.
        # Get dim parameters
        height, width = image.shape[:2]
        features = []
        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # computeHarrisValues() computes the harris score at each pindex_xel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)
        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pindex_xels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)
        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue
                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                f = cv2.KeyPoint()
                f.size = 10
                f.pt = (x,y)
                f.angle = orientationImage[y][x]
                f.response = harrisImage[y][x]
                # TODO-BLOCK-END
                features.append(f)
        return features

class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        # import pdb; pdb.set_trace()
        pdb.set_trace()
        return detector.detect(image)

## Feature descriptors #########################################################
class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError

class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        # Cast to float
        image = image.astype(np.float32)
        # Convert to BGR scale
        image /= 255.
        # Convert to gray scale
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        for i, f in enumerate(keypoints):
            x, y = int(f.pt[0]), int(f.pt[1])
            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pindex_xels outside the image as zero.
            # TODO-BLOCK-BEGIN
            x = int(x)
            y = int(y)
            for rows in range (y-2, y+3):
            	for columns in range (x-2, x+3):
            		if (columns>-1 and columns<grayImage.shape[1] and rows>-1 and rows<grayImage.shape[0]):
            			pindex_xel = grayImage[rows][columns]
            		else:
            			pindex_xel = 0
            		desc[i][j] = pindex_xel
            # TODO-BLOCK-END
        return desc

class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        # Cast to float
        image = image.astype(np.float32)
        # Convert to BGR scale
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pindex_xel in the 40x40 rotated window surrounding
            # the feature to the appropriate pindex_xels in the 8x8 feature
            # descriptor image.
            transMx = np.zeros((2, 3))

            # TODO-BLOCK-BEGIN
            vector = np.array([-1*f.pt[0], -1*f.pt[1], 0])
            # Inputs translation vector represented by 1D numpy array wand outputs 4x4 numpy array with 3D translation
            translation_1 = transformations.get_trans_mx(vector)
            # Inputs angles in x,y,z orientations and ouputs 4x4 numpy array with 3D rotation
            rotation = transformations.get_rot_mx(0, 0, -f.angle/180*np.pi)
            # Inputs scaling in x,y,z, directions and puts 4x4 numpy array with 3D scaling
            scale = transformations.get_scale_mx(0.2, 0.2, 1) 
            translation_2 = transformations.get_trans_mx(np.array([4,4,0]))
            temp = np.dot(translation_2, np.dot(scale, np.dot(rotation, translation_1)))
            transMx = np.array([[temp[0][0], temp[0][1], temp[0][3]],
            				    [temp[1][0], temp[1][1], temp[1][3]]])

            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrindex_x
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)
            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is negligibly small (which we 
            # define as less than 1e-10) then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
            if destImage.std() < 1e-10:
            	destImage = np.zeros((windowSize,windowSize))
            else: 
            	destImage = (destImage- destImage.mean())/destImage.std()

            destImage = destImage.flatten()
            desc[i] = destImage
            # TODO-BLOCK-END
        return desc

class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc

# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError('NOT IMPLEMENTED')

## Feature matchers ############################################################

class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)
            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        # Computes the distances between each pair of inputs
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        dist = scipy.spatial.distance.cdist(desc1, desc2,'euclidean')
        indexes=dist.argmin(1)
        for i in range(0, desc1.shape[0]):
            # https://stackoverflow.com/questions/31690265/matching-features-with-orb-python-opencv
        	matcher = cv2.DMatch()
        	matcher.queryIdx = i
        	matcher.trainIdx = indexes[i]
        	matcher.distance = dist[i][matcher.trainIdx]
        	matches.append(matcher)
        # TODO-BLOCK-END
        return matches

class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []
        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        # TODO-BLOCK-BEGIN 
        # Computes the distances between each pair of inputs
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        dist = scipy.spatial.distance.cdist(desc1, desc2,'euclidean')

        # Getting the first set of minimum distance arrays
        # Setting up first minumum distance arrays
        min_indexes_1 = dist.argmin(1)
        dist_1 = np.zeros(desc1.shape[0])
        train = np.empty_like(min_indexes_1)

        # Getting the first set distance values and the array for training
        for i in range(0,desc1.shape[0]):
        	train[i] = min_indexes_1[i]
        	dist_1[i] = dist[i][train[i]]
        	dist[i][train[i]] = 1000

		#Setting up second minimum distance matrices
        second_min_indexes=dist.argmin(1)
        second_dist = np.zeros(desc1.shape[0])

		#Getting the second distance values
        for i in range(0,desc1.shape[0]):
        	second_min_index = second_min_indexes[i]
        	second_dist[i] = dist[i][second_min_index]

		#Creating the matches
        for i in range(0,desc1.shape[0]):
        	featurematch = cv2.DMatch()
        	featurematch.queryIdx = i
        	featurematch.trainIdx = train[i]
        	if second_dist[i] != 0:
        		featurematch.distance = dist_1[i]/second_dist[i]
        	else:
        		featurematch.distance = 1

        	matches.append(featurematch)

        # TODO-BLOCK-END

        return matches