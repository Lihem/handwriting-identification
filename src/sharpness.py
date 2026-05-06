import cv2
import numpy as np

def calculate_sharpness_with_smoothing(image_path):
    def get_strong_keypts(keypoints, min_distance=5):
        # sorting keypoints by response (desc)
        sorted_kps = sorted(keypoints, key=lambda x: x.response, reverse=True)
        kept_kps = []
        
        for current_kps in sorted_kps:
            is_redundant = False
            for kept_kp in kept_kps:
                dist = np.linalg.norm(np.array(current_kps.pt)-np.array(kept_kp.pt))
                if dist < min_distance:
                    is_redundant = True
                    break
            if not is_redundant:
                kept_kps.append(current_kps)
        return kept_kps
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    # blurring to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # morphological opening to remove small artifacts
    img_morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    img_smoothed = cv2.GaussianBlur(img_morph, (5, 5), 0)

    # skeletonize image to gain thickness invariant representation
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    skeleton = np.zeros(binary.shape, np.uint8)
    temp = binary.copy()
    while True:
        open_temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, element)
        temp_sub_open = cv2.subtract(temp, open_temp)
        eroded = cv2.erode(temp, element)
        skeleton = cv2.bitwise_or(skeleton, temp_sub_open)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    # detecting corners using Harris-Laplace to gain scale invariance
    detector = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(
        numOctaves=6, 
        corn_thresh=0.05, 
        DOG_thresh=0.01
    )
    keypoints = detector.detect(img_smoothed, None)

    cleaned_keypoints = get_strong_keypts(keypoints, min_distance=15)
    valid_keypoints = [kp for kp in cleaned_keypoints if 2 < kp.size < 50]
    corner_count = len(valid_keypoints)
    total_stroke_pixels = cv2.countNonZero(skeleton)
    pointiness_ratio = np.float32(corner_count)/np.float32(total_stroke_pixels)
    #print(f"corners:{corner_count}, pixels:{total_stroke_pixels}, ratio:{pointiness_ratio}")

    scale_factor = 150.0 
    score = (pointiness_ratio*scale_factor)
    return score