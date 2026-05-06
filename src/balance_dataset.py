import os
import shutil
import random
from collections import defaultdict


def balance_dataset():
    source_dir = os.path.abspath("IAM_GITHUB/lines_unfiltered")
    target_dir = os.path.abspath("IAM_GITHUB/lines_balanced")

    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}")
        return

    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")
    else:
        print(f"Target directory already exists: {target_dir}")

    files = [f for f in os.listdir(source_dir) if f.endswith(".png")]

    writer_images = defaultdict(list)

    for filename in files:
        # Filename format: XXX-(Y)+-(Z)+.png
        # Writer ID is the first part
        parts = filename.split("-")
        if len(parts) >= 1:
            writer_id = parts[0]
            writer_images[writer_id].append(filename)

    writers_processed = 0
    images_copied = 0

    print(f"Found {len(writer_images)} writers.")

    for writer_id, images in writer_images.items():
        if len(images) >= 10:
            # Select exactly 10 images
            selected_images = random.sample(images, 10)

            for image in selected_images:
                src_path = os.path.join(source_dir, image)
                dst_path = os.path.join(target_dir, image)
                shutil.copy2(src_path, dst_path)
                images_copied += 1

            writers_processed += 1
        else:
            # print(f"Skipping writer {writer_id}: only {len(images)} images.")
            pass

    print("Finished balancing dataset.")
    print(f"Processed {writers_processed} writers with >= 10 images.")
    print(f"Total images copied: {images_copied}")


if __name__ == "__main__":
    balance_dataset()
