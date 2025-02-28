import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt

def waterShed(image_path):
    img = cv.imread(image_path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)


    # Noise removal using morphology (opening)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.09*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    
    
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    cv.watershed(img, markers)
    
    visual_markers = markers.copy()
    visual_markers[visual_markers == -1] = 0

    processed_image_dir = './static/processed_images'
    os.makedirs(processed_image_dir, exist_ok=True)  # Ensure the directory exists

    colored_markers = plt.get_cmap('nipy_spectral')(visual_markers / visual_markers.max())  # Normalize to 0-1
    colored_markers = (colored_markers[:, :, :3] * 255).astype(np.uint8)
    
    # Construct the full processed image path
    processed_image_path = os.path.join(processed_image_dir, os.path.basename(image_path).replace(".jpg", "_watershed.png"))

    # Save the watershed result image
    cv.imwrite(processed_image_path, cv.cvtColor(colored_markers, cv.COLOR_RGB2BGR))
    # Return the relative path for Flask to access the image
    return processed_image_path


def kmeans(image_path):
    img = cv.imread(image_path)
    img2 = img.reshape(-1,3)
    img2 = np.float32(img2)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    k = 4
    attemps = 10
    ret,label,center=cv.kmeans(img2,k,None,criteria,attemps,cv.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
     # Save the results to disk in the same way as 'sam' function
    processed_image_dir = './static/processed_images'
    os.makedirs(processed_image_dir, exist_ok=True)  # Ensure the directory exists

    # Construct the full processed image path
    processed_image_path = os.path.join(processed_image_dir, os.path.basename(image_path).replace(".jpg", "_kmeans.png"))

    # Save the K-means result image
    cv.imwrite(processed_image_path, res2)

    # Return the relative path for Flask to access the image
    return processed_image_path


def canny_edge(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    edges = cv.Canny(img, 100, 200)
    
    processed_image_dir = './static/processed_images'
    os.makedirs(processed_image_dir, exist_ok=True)
    
    processed_image_path = os.path.join(processed_image_dir, os.path.basename(image_path).replace(".jpg", "_canny.png"))
    cv.imwrite(processed_image_path, edges)
    
    return processed_image_path