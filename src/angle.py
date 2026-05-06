import cv2
import numpy as np
import math
from sklearn.linear_model import RANSACRegressor, LinearRegression


def _get_baseline_points(binary_text):
    points = []
    h, w = binary_text.shape
    
    # column-wise scan to find bottom-most ink pixels
    for x in range(w):
        column = binary_text[:, x]
        ink_pixels = np.where(column > 0)[0]
        
        if len(ink_pixels) == 0:
            continue
            
        ink_span = ink_pixels.max() - ink_pixels.min()
        
        if ink_span > 10: 
            y_bottom = ink_pixels.max()
            points.append([x, y_bottom])
    
    return np.array(points)

def _fit_baseline_ransac(points):
    if len(points) == 0:
        return 0.0

    X = points[:, 0].reshape(-1, 1)  
    y = points[:, 1]                

    # RANSAC finding the best fit line
    model = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=0.1,        
        residual_threshold=5.0, 
        max_trials=1000,
        random_state=42
    )
    
    model.fit(X, y)
    slope = model.estimator_.coef_[0]
    return slope

def get_baseline_angle(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_img = cv2.bitwise_not(binary) # Invert
    
    # morphological opening to clean noise
    kernel = np.ones((3,3), np.uint8)
    text_opened = cv2.morphologyEx(text_img, cv2.MORPH_OPEN, kernel)

    points = _get_baseline_points(text_opened)
    if len(points) < 10:
        return 0.0 # there is not enough data to compute a reliable baseline

    slope = _fit_baseline_ransac(points)
    
    angle_degrees = np.degrees(np.arctan(slope))
    
    return float(angle_degrees)

def get_slant_angle(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = cv2.bitwise_not(binary)
    h, w = binary_image.shape

    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)    
    # Hough transform to detect lines
    min_len = h / 3 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=min_len, maxLineGap=10)
    
    if lines is None:
        return 0.0

    slants = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # angle calculation and normalization
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)
        
        if angle_deg < 0:
            angle_deg += 180
            
        # final filtering of verticalish lines (45 to 135 degrees)
        if 45 <= angle_deg <= 135:
            slants.append(angle_deg)

    if not slants:
        return 0.0

    slant_angle = np.median(slants)
    offset_angle = slant_angle - 90 
    
    return float(offset_angle)
