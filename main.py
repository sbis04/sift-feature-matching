import cv2
import matplotlib.pyplot as plt

# Reading the two raw RGB images
img1 = cv2.imread('eiffel_2.png')
img2 = cv2.imread('eiffel_1.png')

# Generating the grayscale version of the images
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Applying SIFT
sift = cv2.xfeatures2d.SIFT_create()

firstImagKeypoints, firstImageDescriptors = sift.detectAndCompute(img1, None)
secondImagKeypoints, secondImageDescriptors = sift.detectAndCompute(img2, None)

# Perform feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(firstImageDescriptors, secondImageDescriptors)
matches = sorted(matches, key=lambda x: x.distance)

# Show the matching
img3 = cv2.drawMatches(img1, firstImagKeypoints, img2, secondImagKeypoints, matches[:50], img2, flags=2)
plt.imshow(img3), plt.show()
