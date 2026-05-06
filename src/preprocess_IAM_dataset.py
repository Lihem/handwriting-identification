# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python",
#     "numpy",
# ]
# ///
import shutil
import os
import cv2
import numpy as np
from collections import defaultdict

IMAGES_DIR = "IAM_GITHUB/images/"
OUTPUT_DIR = "IAM_GITHUB/lines_unfiltered/"


def segment_and_save_lines(
    image_path: str,
    filename: str,
    writer_id: str,
    writer_counts: dict[str, int],
    limit: int = 10,
) -> list[str]:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error reading {image_path}")
        return []

    # Binarize to find lines
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to connect text horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours
    lines = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 400 and h in range(20, 100):  # Filter small noise
            lines.append((x, y, w, h, c))

    # Sort by Y position
    lines.sort(key=lambda b: b[1])

    base_name = os.path.splitext(filename)[0]
    saved_files = []

    for idx, (x, y, w, h, c) in enumerate(lines):
        if writer_counts[writer_id] >= limit:
            break

        # Create mask for this line
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [c], -1, 255, -1)

        # Crop with padding
        pad = 10
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)

        line_crop = img[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        # Normalize background and ink
        line_float = line_crop.astype(np.float32)
        bg = cv2.GaussianBlur(line_float, (51, 51), 0)

        norm = line_float / (bg + 1.0) * 255.0
        norm = np.clip(norm, 0, 255).astype(np.uint8)

        # Close holes in ink (dark) using Morphological Opening (Erosion -> Dilation)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        norm = cv2.morphologyEx(norm, cv2.MORPH_OPEN, kernel)

        # Contrast stretching for clean white background and consistent ink (soft threshold)
        otsu_thresh, _ = cv2.threshold(
            norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Soft thresholding
        lower = otsu_thresh * 0.6
        upper = min(255, otsu_thresh + (255 - otsu_thresh) * 0.5)

        norm_float = norm.astype(np.float32)
        if upper > lower + 10:  # Ensure range is valid and not too small
            clean_line = (norm_float - lower) / (upper - lower) * 255.0
            clean_line = np.clip(clean_line, 0, 255).astype(np.uint8)
        else:
            clean_line = norm

        # Apply mask to remove neighbors
        clean_line[mask_crop == 0] = 255

        # Resize to fixed height
        target_height = 64
        h, w = clean_line.shape
        scale = target_height / h
        new_w = int(w * scale)

        if scale > 1:
            clean_line = cv2.resize(
                clean_line, (new_w, target_height), interpolation=cv2.INTER_CUBIC
            )
        else:
            clean_line = cv2.resize(
                clean_line, (new_w, target_height), interpolation=cv2.INTER_AREA
            )

        save_name = f"{base_name}-{idx}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        cv2.imwrite(save_path, clean_line)
        saved_files.append(save_path)
        writer_counts[writer_id] += 1

    return saved_files


def main() -> int:
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR)
    full_text_images = tuple(
        name for name in os.listdir(IMAGES_DIR) if name.endswith(".png")
    )

    writer_counts = defaultdict(int)

    for image_name in full_text_images:
        writer_id = image_name.split("-")[0]
        if writer_counts[writer_id] >= 10:
            continue

        image_path = os.path.join(IMAGES_DIR, image_name)
        print(f"Processing image: {image_path}")
        segment_and_save_lines(image_path, image_name, writer_id, writer_counts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
