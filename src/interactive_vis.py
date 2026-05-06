# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "bokeh>=3.8.1",
#     "matplotlib>=3.10.8",
#     "numpy>=2.3.5",
#     "opencv-contrib-python>=4.11.0.86",
#     "opencv-python>=4.11.0.86",
#     "pillow>=12.0.0",
#     "scikit-learn>=1.8.0",
# ]
# ///
import base64
from sklearn.manifold._t_sne import TSNE
from sklearn.preprocessing._label import LabelEncoder
import matplotlib
import os
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm
from collections import defaultdict
import bokeh.plotting
import bokeh.models
import bokeh.layouts

# Import feature extraction functions
# Assuming these files are in the same directory
try:
    from hog import get_hog_features
    from angle import get_baseline_angle, get_slant_angle
    from sharpness import calculate_sharpness_with_smoothing
    from omar.word_detector import get_word_spacing_features
    from bovw import BoVWExtractor
except ImportError as e:
    print(f"Error importing feature extraction modules: {e}")
    exit(1)


def extract_features(image_path, bovw_extractor=None):
    """
    Extracts and concatenates features from an image.
    Features: HOG (vector), Baseline Angle (scalar), Slant Angle (scalar), Sharpness (scalar), BoVW (vector).
    """
    try:
        hog_feat = get_hog_features(image_path)

        baseline_angle = get_baseline_angle(image_path)
        slant_angle = get_slant_angle(image_path)

        sharpness_score = calculate_sharpness_with_smoothing(image_path)

        if isinstance(sharpness_score, tuple):
            sharpness_score = sharpness_score[0]

        spacing_features = get_word_spacing_features(image_path)

        bovw_feat = np.array([])
        if bovw_extractor is not None:
            bovw_feat = bovw_extractor.extract(image_path)

        # Concatenate all features
        features = np.concatenate(
            [
                hog_feat,
                np.array([baseline_angle, slant_angle, sharpness_score]),
                np.array(spacing_features),
                bovw_feat,
            ]
        )

        return features
    except Exception as e:
        print(f"Error extracting features for {image_path}: {e}")
        return None


def load_data(images_dir):
    """
    Loads images from the directory and groups them by user ID.
    User ID is the first 3 digits of the filename.
    """
    user_images = defaultdict(list)

    if not os.path.exists(images_dir):
        print(f"Directory not found: {images_dir}")
        return user_images

    files = os.listdir(images_dir)
    for f in files:
        if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            continue

        # Extract User ID (first 3 chars)
        try:
            user_id = f[:3]
            # Verify it's a number
            int(user_id)

            full_path = os.path.join(images_dir, f)
            user_images[user_id].append(full_path)
        except ValueError:
            print(f"Skipping file with invalid format: {f}")
            continue

    return user_images


def evaluate_pipeline(
    evaluation_name="default_run", num_users_limit=None, bovw_clusters=1000
):
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create output directory
    output_dir = os.path.join("evaluation", evaluation_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    images_dir = "../cv-project/MY_IAM/IAM_GITHUB/images"

    print("1. Loading Data...")
    user_images = load_data(images_dir)

    if not user_images:
        print("No images found.")
        return

    if num_users_limit is not None:
        print(f"Limiting evaluation to {num_users_limit} users.")
        all_users = list(user_images.keys())
        if len(all_users) > num_users_limit:
            selected_users = all_users[:num_users_limit]
            user_images = {u: user_images[u] for u in selected_users}

    print(f"Found {len(user_images)} users.")

    # Prepare datasets
    X_train = []  # Database features
    y_train = []  # Database labels (User IDs)
    X_test = []  # Test features
    y_test = []  # Test labels (User IDs)
    X_final_test = []  # Final Test features
    y_final_test = []  # Final Test labels (User IDs)

    train_paths = []
    test_paths = []
    final_test_paths = []

    print("1.5. Splitting Data (Train/Test)...")
    # Pre-calculate splits to ensure BoVW only sees training data
    user_splits = {}  # user_id -> (test_path, [train_paths])
    all_training_images_for_bovw = []

    for user_id, paths in user_images.items():
        if len(paths) < 2:
            print(f"User {user_id} has less than 2 images. Skipping.")
            continue

        # Select one random image for testing
        test_image, final_test_image = random.sample(paths, 2)
        # The rest are for the database (training)
        train_images = [p for p in paths if p != test_image and p != final_test_image]

        user_splits[user_id] = (final_test_image, test_image, train_images)
        all_training_images_for_bovw.extend(train_images)

    print("1.6. Initializing and Training BoVW (on Training Data only)...")

    bovw = BoVWExtractor(n_clusters=bovw_clusters)

    bovw_train_subset = all_training_images_for_bovw

    if not bovw_train_subset:
        print("Error: No training images available for BoVW.")
        return

    bovw.fit(bovw_train_subset)
    # ---------------------------

    print("2. Extracting Features...")

    # For progress tracking
    total_users = len(user_splits)
    processed_users = 0
    image_b64s = []  # For Bokeh interactive plot

    for user_id, (final_test_image, test_image, train_images) in user_splits.items():
        # Extract features for Test Image
        feat_test = extract_features(test_image, bovw_extractor=bovw)
        if feat_test is not None:
            X_test.append(feat_test)
            y_test.append(user_id)
            test_paths.append(test_image)

        feat_final_test = extract_features(final_test_image, bovw_extractor=bovw)
        if feat_final_test is not None:
            X_final_test.append(feat_final_test)
            y_final_test.append(user_id)
            final_test_paths.append(final_test_image)

        # Extract features for Database Images
        for img_path in train_images:
            feat_train = extract_features(img_path, bovw_extractor=bovw)
            if feat_train is not None:
                X_train.append(feat_train)
                train_paths.append(img_path)
                with open(img_path, "rb") as img_file:
                    img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                    image_b64s.append(f"data:image/png;base64,{img_b64}")
                y_train.append(user_id)

        processed_users += 1
        if processed_users % 10 == 0:
            print(f"Processed {processed_users}/{total_users} users...")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_final_test = np.array(X_final_test)
    y_final_test = np.array(y_final_test)

    print(f"Database size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Final Test set size: {X_final_test.shape}")

    if len(X_train) == 0 or len(X_test) == 0 or len(X_final_test) == 0:
        print("Insufficient data for evaluation.")
        return

    print("3. Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("4. Applying PCA...")
    pca = PCA(n_components=0.90, random_state=42)

    X_train_pca = pca.fit_transform(X_train_scaled)
    print(f"Encoding {len(set(y_train))} unique labels...")
    le = LabelEncoder()
    labels = le.fit_transform(y_train)

    # t-SNE Visualization
    print("Running t-SNE...")
    perplexity = 30
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=24,
        max_iter=3000,
        random_state=42,
        init="pca",
        learning_rate="auto",
        verbose=1,
    )
    X_embedded = tsne.fit_transform(X_train_pca)

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
    plt.colorbar(scatter, label="Class Label")
    plt.title(f"t-SNE of HOG Features (n={len(X_train_pca)})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    outfile = "hog_clusters_tsne.png"
    plt.savefig(outfile)
    print(f"Saved plot to {outfile}")

    # Interactive Plotting with Bokeh
    print("Generating interactive plot...")

    # Generate colors from the same colormap
    cmap = matplotlib.cm.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=labels.min(), vmax=labels.max())
    hex_colors = [matplotlib.colors.rgb2hex(cmap(norm(label))) for label in labels]

    data = dict(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        desc=train_paths,
        image=image_b64s,
        label=y_train,
        color=hex_colors,
        alpha=[0.8] * len(y_train),
    )

    source = bokeh.models.ColumnDataSource(data=data)
    original_source = bokeh.models.ColumnDataSource(data=data)

    # Calculate bounds with some padding to keep axes fixed
    x_min, x_max = X_embedded[:, 0].min(), X_embedded[:, 0].max()
    y_min, y_max = X_embedded[:, 1].min(), X_embedded[:, 1].max()
    padding_x = (x_max - x_min) * 0.05
    padding_y = (y_max - y_min) * 0.05

    p = bokeh.plotting.figure(
        title="t-SNE of HOG Features (Hover for Image)",
        width=900,
        height=900,
        x_range=(x_min - padding_x, x_max + padding_x),
        y_range=(y_min - padding_y, y_max + padding_y),
    )

    p.scatter(
        "x",
        "y",
        source=source,
        fill_color="color",
        line_color=None,
        size=12,
        fill_alpha="alpha",
    )

    # Create a Div to display the image and details
    div = bokeh.models.Div(
        width=400, height=900, text="<h3>Hover over a point to see details</h3>"
    )

    # Callback for HoverTool to update the Div
    hover_callback = bokeh.models.CustomJS(
        args=dict(source=source, div=div),
        code="""
        const indices = cb_data.index.indices;
        if (indices.length > 0) {
            const index = indices[0];
            const img_src = source.data['image'][index];
            const desc = source.data['desc'][index];
            const label = source.data['label'][index];
            div.text = `
                <div>
                    <img src="${img_src}" style="max-width: 350px; border: 1px solid #ccc;">
                    <p><b>Path:</b> ${desc}</p>
                    <p><b>Label:</b> ${label}</p>
                </div>
            `;
        }
    """,
    )

    hover = bokeh.models.HoverTool(tooltips=None, callback=hover_callback)
    p.add_tools(hover)

    # Add Select widget for filtering
    unique_labels = sorted(list(set(y_train.tolist())))
    select = bokeh.models.Select(
        title="Filter by Class:", value="All", options=["All"] + unique_labels
    )

    callback = bokeh.models.CustomJS(
        args=dict(source=source, original_source=original_source),
        code="""
        const data = source.data;
        const original_data = original_source.data;
        const selected_label = cb_obj.value;
        const n = original_data['label'].length;

        // Reset data to match original_data structure
        data['x'] = original_data['x'].slice();
        data['y'] = original_data['y'].slice();
        data['desc'] = original_data['desc'].slice();
        data['image'] = original_data['image'].slice();
        data['label'] = original_data['label'].slice();
        
        const colors = [];
        const alphas = [];
        
        for (let i = 0; i < n; i++) {
            if (selected_label === 'All') {
                colors.push(original_data['color'][i]);
                alphas.push(0.8);
            } else {
                if (original_data['label'][i] === selected_label) {
                    colors.push(original_data['color'][i]);
                    alphas.push(1.0);
                } else {
                    colors.push('#d3d3d3'); // Light gray
                    alphas.push(0.1);
                }
            }
        }
        data['color'] = colors;
        data['alpha'] = alphas;
        
        source.change.emit();
    """,
    )
    select.js_on_change("value", callback)

    layout = bokeh.layouts.column(select, bokeh.layouts.row(p, div))

    interactive_outfile = "hog_clusters_interactive.html"
    bokeh.plotting.output_file(interactive_outfile)
    bokeh.plotting.save(layout)
    print(f"Saved interactive plot to {interactive_outfile}")

    return 0


if __name__ == "__main__":
    evaluate_pipeline("", num_users_limit=352, bovw_clusters=500)
