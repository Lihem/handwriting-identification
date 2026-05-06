# Handwriting-Based Person Identification via Combined Texture and Graphological Features
## Abstract:
This project proposes a computer vision pipeline for identifying individuals based on offline handwriting samples. The system integrates texture-based features, specifically Histogram of Oriented Gradients (HOG) and Bag of Visual Words (BoVW), with handcrafted graphological features such as baseline angle, slant angle, curvature, and word spacing. Dimensionality reduction was performed using Principal Component Analysis (PCA) before classification via a Nearest Neighbor search. Experimental results demonstrate that the combination of texture and graphological features yields a Top-1 accuracy of 70% and a Top-10 accuracy of 90% on a subset of 100 users. While accuracy declines as the user pool expands to 352, the high retrieval rates suggest the system is effective for ranking-based identification tasks.

For more details, you can read our [project report](./Report.pdf).

---