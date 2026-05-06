# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pillow",
#     "scikit-learn",
#     "matplotlib",
#     "bokeh",
# ]
# ///
"""Histogram of Oriented Gradients (HOG) feature extraction implementation."""

from collections.abc import Callable
from typing import cast
from typing import TypeVar
from typing import TypeAlias
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_ScalarType_co = TypeVar("_ScalarType_co", bound=np.generic, covariant=True)
NDArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[_ScalarType_co]]
_1DArray: TypeAlias = np.ndarray[tuple[int], np.dtype[_ScalarType_co]]
_2DArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_ScalarType_co]]
_3DArray: TypeAlias = np.ndarray[tuple[int, int, int], np.dtype[_ScalarType_co]]


_GradientImage = _2DArray[np.int16]


def _compute_central_differences(
    image: _2DArray[np.uint8],
) -> tuple[_GradientImage, _GradientImage]:
    """Compute the central differences of the image in y and x directions."""
    # to prevent overflow, convert to signed integer type
    signed_image = image.astype(np.int16)

    # pad images along each axis to avoid artificial edges
    # handwriting image have white background, so pad with max value
    pad_constant_value = np.iinfo(image.dtype).max
    vertically_padded = np.pad(
        signed_image,
        ((1,), (0,)),
        "constant",
        constant_values=pad_constant_value,
    )
    horizontally_padded = np.pad(
        signed_image,
        ((0,), (1,)),
        "constant",
        constant_values=pad_constant_value,
    )

    y_gradients = cast(
        _GradientImage, vertically_padded[2:, :] - vertically_padded[:-2, :]
    )
    x_gradients = cast(
        _GradientImage,
        horizontally_padded[:, 2:] - horizontally_padded[:, :-2],
    )

    return y_gradients, x_gradients


def compute_hog(
    image: _2DArray[np.uint8],
    cell_size: tuple[int, int] = (8, 8),
    block_size: tuple[int, int] = (2, 2),
    nbins: int = 9,
) -> tuple[_3DArray[np.float32], Callable[[str], None]]:
    """Compute HOG features for a given grayscale image.

    Args:
        image: 2D array representing a grayscale image.
        cell_size: Size of each cell in pixels (height, width).
        block_size: Size of each block in cells (height, width).
        nbins: Number of orientation bins for the histogram.

    Returns:
        HOG features and a visualization function that saves
        an image to a given path.
    """
    gradients = _compute_central_differences(image)

    gradient_magnitutes = np.hypot(*gradients)
    gradient_orientations = np.arctan2(*gradients) * (180 / np.pi) % 180

    # Compute histograms for each cell
    height, width = gradient_magnitutes.shape
    cell_rows = height // cell_size[0]
    cell_cols = width // cell_size[1]

    def get_cells_view(array: NDArray[np.floating]) -> NDArray[np.floating]:
        # Ensure dimensions are divisible by cell size for reshaping
        array_cropped = array[: cell_rows * cell_size[0], : cell_cols * cell_size[1]]
        return (
            array_cropped.reshape(cell_rows, cell_size[0], cell_cols, cell_size[1])
            .swapaxes(1, 2)
            .reshape(cell_rows, cell_cols, -1)
        )

    cells_mag = get_cells_view(gradient_magnitutes)
    cells_ori = get_cells_view(gradient_orientations)

    # Determine bin indices for all pixels at once
    bin_indices = (cells_ori / (180 / nbins)).astype(int) % nbins

    histograms = np.zeros((cell_rows, cell_cols, nbins))
    for b in range(nbins):
        mask = bin_indices == b
        histograms[:, :, b] = np.sum(cells_mag * mask, axis=2)

    block_view = sliding_window_view(histograms, window_shape=(*block_size, 1))

    n_blocks_row, n_blocks_col, *_ = block_view.shape
    flat_blocks = block_view.reshape(n_blocks_row, n_blocks_col, -1)

    # Compute L2 norm for each block
    block_norms = np.linalg.norm(flat_blocks, axis=2, keepdims=True)

    # Normalize
    normalized_blocks = flat_blocks / (block_norms + 1e-5)

    ################ Visualization function for debugging ###########################
    def save_visualization(path: str) -> None:
        """Visualize HOG features by drawing lines for each cell's histogram."""
        # Create a blank image for visualization

        hog_image = np.zeros((height, width), dtype=np.float32)
        radius = min(cell_size) // 2 - 1

        for r in range(cell_rows):
            for c in range(cell_cols):
                cell_grad = histograms[r, c]
                cell_grad /= np.max(cell_grad) + 1e-5  # Normalize for visualization

                center_x = c * cell_size[1] + cell_size[1] // 2
                center_y = r * cell_size[0] + cell_size[0] // 2

                for bin_idx in range(nbins):
                    angle = bin_idx * 180 / nbins + 90
                    angle_rad = angle * np.pi / 180
                    magnitude = cell_grad[bin_idx]

                    if magnitude > 0.1:  # Threshold to avoid clutter
                        x1 = int(center_x - radius * magnitude * np.cos(angle_rad))
                        y1 = int(center_y - radius * magnitude * np.sin(angle_rad))
                        x2 = int(center_x + radius * magnitude * np.cos(angle_rad))
                        y2 = int(center_y + radius * magnitude * np.sin(angle_rad))

                        # Draw line
                        rr, cc = (
                            np.linspace(y1, y2, 10).astype(int),
                            np.linspace(x1, x2, 10).astype(int),
                        )
                        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)

                        hog_image[rr[valid], cc[valid]] = 255

        # Save the visualization image
        import PIL.Image

        PIL.Image.fromarray(hog_image.astype(np.uint8)).convert("L").save(path)

    ###############################################################################

    return normalized_blocks.astype(np.float32), save_visualization


def compute_fixed_size_hog(
    image: _2DArray[np.uint8],
    fixed_height: int = 128,
) -> _1DArray[np.float32]:
    """Compute fixed-size HOG features by resizing to fixed height and aggregating.

    This pipeline:
    1. Resizes the input image to a fixed height while maintaining aspect ratio.
    2. Computes HOG features on the resized image.
    3. Aggregates features across the width (mean and std) to create a
       width-invariant feature vector.

    Args:
        image: Input grayscale image.
        fixed_height: Target height in pixels.

    Returns:
        Fixed-length HOG feature vector (invariant to image width).
    """
    import PIL.Image

    # Convert to PIL Image for easy resizing
    pil_img = PIL.Image.fromarray(image)

    # Calculate new size maintaining aspect ratio.
    # Note: For the IAM dataset, the ratio will always be 1, so this is
    # redundant for that case. But we keep it general for other datasets.
    ratio = fixed_height / pil_img.height
    new_width = int(pil_img.width * ratio)

    # Ensure minimum width for at least one HOG block (needs 2 cells of 8px = 16px)
    # We add a bit of margin to be safe
    new_width = max(new_width, 32)

    new_size = (new_width, fixed_height)

    # Resize (LANCZOS for high quality downsampling)
    pil_img = pil_img.resize(new_size, resample=PIL.Image.Resampling.LANCZOS)

    # Convert back to numpy
    fixed_image = np.array(pil_img)

    # Compute HOG
    features, _ = compute_hog(cast(_2DArray[np.uint8], fixed_image))

    # Aggregate statistics across the horizontal axis (axis 1)
    # This makes the feature vector invariant to the width of the image (length of text)
    mean_features = np.mean(features, axis=1)
    std_features = np.std(features, axis=1)

    # Concatenate mean and std deviation
    combined_features = np.concatenate([mean_features, std_features], axis=1)

    return combined_features.flatten()


def get_hog_features(image_path: str, fixed_height: int = 128) -> _1DArray[np.float32]:
    """Compute HOG features for a single image file.

    Args:
        image_path: Path to the image file.
        fixed_height: Target height in pixels for resizing.

    Returns:
        Fixed-length HOG feature vector.
    """
    import PIL.Image

    # Load image and convert to grayscale
    img = PIL.Image.open(image_path).convert("L")
    img_arr = cast(_2DArray[np.uint8], np.array(img))

    # Compute fixed size features
    return compute_fixed_size_hog(img_arr, fixed_height=fixed_height)


def main() -> int:
    """Example usage of the HOG feature extractor with clustering."""
    import os
    import matplotlib.pyplot as plt
    import matplotlib.cm
    import matplotlib.colors
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import PIL.Image
    import base64
    from io import BytesIO
    import bokeh.plotting
    import bokeh.models

    imgdir = "IAM/image"
    if not os.path.exists(imgdir):
        print(f"Directory {imgdir} not found.")
        return 1

    print(f"Loading images from {imgdir}...")
    features = []
    image_b64s = []
    filenames = []
    valid_images = 0

    image_files = sorted(os.listdir(imgdir))

    for img_path in image_files:
        full_path = os.path.join(imgdir, img_path)
        # Convert to grayscale to ensure 2D array
        img = PIL.Image.open(full_path).convert("L")

        # Prepare thumbnail for interactive plot
        thumb = img.copy()
        thumb.thumbnail((256, 256))
        buffered = BytesIO()
        thumb.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_b64s.append(f"data:image/png;base64,{img_str}")
        filenames.append(img_path)

        img_arr = cast(_2DArray[np.uint8], np.array(img))

        # Compute fixed size features
        feat = compute_fixed_size_hog(img_arr)
        features.append(feat)
        valid_images += 1

        if valid_images % 50 == 0:
            print(f"Processed {valid_images} images...")

    if not features:
        print("No features extracted.")
        return 1

    X = np.array(features)
    print(f"Feature matrix shape: {X.shape}")

    # PCA Dimensionality Reduction
    print("Running PCA...")
    # Reduce to 50 dimensions (or fewer if samples are scarce) as recommended for t-SNE
    n_components = min(50, len(features), X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced shape: {X_pca.shape}")

    # Clustering
    n_clusters = 20
    print(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42, verbose=1)
    labels = kmeans.fit_predict(X_pca)

    # t-SNE Visualization
    print("Running t-SNE...")
    # Perplexity should be less than number of samples. Default is 30.
    perplexity = min(30, len(features) - 1) if len(features) > 1 else 1
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=24,  # Increased from default 12 to make clusters tighter and more separated
        max_iter=3000,  # Increased from default 1000 to ensure convergence
        random_state=42,
        init="pca",
        learning_rate="auto",
        verbose=1,
    )
    X_embedded = tsne.fit_transform(X_pca)

    print("Plotting...")
    plt.figure(figsize=(10, 8))
    cmap_name = "hsv"
    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=labels,
        cmap=cmap_name,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Cluster ID")
    plt.title(f"t-SNE of HOG Features (n={len(features)})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    outfile = "hog_clusters_tsne.png"
    plt.savefig(outfile)
    print(f"Saved plot to {outfile}")

    # Interactive Plotting with Bokeh
    print("Generating interactive plot...")

    # Generate colors from the same colormap
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=labels.min(), vmax=labels.max())
    hex_colors = [matplotlib.colors.rgb2hex(cmap(norm(label))) for label in labels]

    source = bokeh.models.ColumnDataSource(
        data=dict(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            desc=filenames,
            image=image_b64s,
            cluster=labels.astype(str),
            color=hex_colors,
        )
    )

    p = bokeh.plotting.figure(
        title="t-SNE of HOG Features (Hover for Image)",
        width=1200,
        height=900,
    )

    p.scatter(
        "x",
        "y",
        source=source,
        fill_color="color",
        line_color=None,
        size=12,
        fill_alpha=0.8,
    )

    hover = bokeh.models.HoverTool(
        tooltips="""
        <div>
            <div>
                <img
                    src="@image" height="64" alt="@desc"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="1"
                ></img>
            </div>
            <div>
                <span style="font-size: 15px; font-weight: bold;">@desc</span>
            </div>
            <div>
                <span style="font-size: 12px; color: #666;">Cluster: @cluster</span>
            </div>
        </div>
    """
    )
    p.add_tools(hover)

    interactive_outfile = "hog_clusters_interactive.html"
    bokeh.plotting.output_file(interactive_outfile)
    bokeh.plotting.save(p)
    print(f"Saved interactive plot to {interactive_outfile}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
