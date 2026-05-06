import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class BoVWExtractor:
    def __init__(self, n_clusters=50):
        self.n_clusters = n_clusters
        self.kmeans = None
        
        # using SIFT for keypoint detection and descriptor extraction
        self.detector = cv2.SIFT_create()

    def get_descriptors(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = self.detector.detectAndCompute(img, None)
        return descriptors

    def fit(self, image_paths):
        all_descriptors = []
        print("Extracting descriptors")
        
        valid_images = 0
        for path in image_paths:
            des = self.get_descriptors(path)
            if len(des)>0:
                all_descriptors.append(des)
                valid_images += 1

        stacked_descriptors = np.vstack(all_descriptors)
                
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=1000, n_init='auto')
        self.kmeans.fit(stacked_descriptors)

    def extract(self, image_path):
        des = self.get_descriptors(image_path)        
        histogram = np.zeros(self.n_clusters, dtype=np.float32)
        
        if len(des)>0:
            # predicting visual word (cluster center) each descriptor belongs to
            predictions = self.kmeans.predict(des)
            
            # occurrences of each visual word
            for pred in predictions:
                histogram[pred] += 1
           
            norm = np.linalg.norm(histogram)        
            if norm > 0:
                histogram = histogram / norm
            
        return histogram
