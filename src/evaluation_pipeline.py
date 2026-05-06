import os
import random
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import defaultdict

from hog import get_hog_features
from angle import get_baseline_angle, get_slant_angle
from sharpness import calculate_sharpness_with_smoothing
from word_detector import get_word_spacing_features
from bovw import BoVWExtractor
   
def load_data(images_dir):
    user_images = defaultdict(list)

    files = os.listdir(images_dir)
    for f in files:
        # user ID (first 3 chars)
        user_id = f[:3]
        int(user_id) 
        full_path = os.path.join(images_dir, f)
        user_images[user_id].append(full_path)
            
    return user_images

def extract_features_dict(image_path, bovw_extractor=None):
    features = {}
    # HOG 
    features['hog'] = get_hog_features(image_path)
    
    # angles 
    baseline_angle = get_baseline_angle(image_path)
    slant_angle = get_slant_angle(image_path)
    features['angle'] = np.array([baseline_angle, slant_angle])
    
    # sharpness
    sharpness_score = calculate_sharpness_with_smoothing(image_path)
    if isinstance(sharpness_score, tuple):
            sharpness_score = sharpness_score[0]
    features['sharpness'] = np.array([sharpness_score])
    
    # word spacing 
    spacing_features = get_word_spacing_features(image_path)
    features['spacing'] = np.array(spacing_features)

    #BoVW 
    features['bovw'] = bovw_extractor.extract(image_path)
    
    return features

def prepare_datasets(user_images, bovw_clusters=1000, limit_users=None, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    all_users = list(user_images.keys())
    if len(all_users) > limit_users:
        selected_users = all_users[:limit_users]
        user_images = {u: user_images[u] for u in selected_users}
    
    #print(f"{len(user_images)} users")

    user_splits = {} 
    all_training_images_for_bovw = []
    
    for user_id, paths in user_images.items():
        if len(paths) < 2:
            continue
        
        if len(paths) >= 3:
            test_image, final_test_image = random.sample(paths, 2)
            train_images = [p for p in paths if p != test_image and p != final_test_image]
        elif len(paths) == 2:
            test_image, final_test_image = paths
            train_images = [test_image] 
            if not train_images:
                 print(f"User {user_id} has only 2 images, skip")
                 continue

        user_splits[user_id] = (final_test_image, test_image, train_images)
        all_training_images_for_bovw.extend(train_images)

    # training BoVW
    bovw = BoVWExtractor(n_clusters=bovw_clusters)
    bovw.fit(all_training_images_for_bovw)
    data = {
        'train': {'features': [], 'labels': []},
        'test': {'features': [], 'labels': []},
        'final_test': {'features': [], 'labels': []}
    }
    
    train_feats_list = []
    test_feats_list = []
    final_test_feats_list = []
    
    for user_id, (final_test_image, test_image, train_images) in user_splits.items():
        f_test = extract_features_dict(test_image, bovw)
        if f_test:
            test_feats_list.append(f_test)
            data['test']['labels'].append(user_id)
            
        f_final = extract_features_dict(final_test_image, bovw)
        if f_final:
            final_test_feats_list.append(f_final)
            data['final_test']['labels'].append(user_id)
            
        for img in train_images:
            f_train = extract_features_dict(img, bovw)
            if f_train:
                train_feats_list.append(f_train)
                data['train']['labels'].append(user_id)
                
    data['train']['features'] = train_feats_list
    data['test']['features'] = test_feats_list
    data['final_test']['features'] = final_test_feats_list
    
    return data, bovw

def flatten_features(features_list, use_hog=True, use_bovw=True, use_others=True):
    X = []
    for f in features_list:
        parts = []
        if use_hog:
            parts.append(f['hog'])
        if use_others:
            parts.append(f['angle'])
            parts.append(f['sharpness'])
            parts.append(f['spacing'])
        if use_bovw:
            parts.append(f['bovw'])
        
        if parts:
            X.append(np.concatenate(parts))

    return np.array(X)

def evaluate_model(X_train, y_train, X_eval, y_eval, use_pca=True, pca_components=0.70):
    y_train = np.array(y_train)
    y_eval = np.array(y_eval)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    
    if use_pca:
        n_comp = min(50, min(X_train_scaled.shape))
        if isinstance(pca_components, int) and pca_components > min(X_train_scaled.shape):
             n_comp = min(X_train_scaled.shape)
        else:
             n_comp = pca_components

        pca = PCA(n_components=n_comp, random_state=42)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_eval_final = pca.transform(X_eval_scaled)
    else:
        X_train_final = X_train_scaled
        X_eval_final = X_eval_scaled
        
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(X_train_final, y_train)
    
    # Top-K calculation
    k_values = [1, 3, 5, 7, 10]
    max_k = min(max(k_values), len(X_train))
    
    if max_k < 1:
         return {k: 0.0 for k in k_values}

    distances, indices = knn.kneighbors(X_eval_final, n_neighbors=max_k)
    
    top_k_accuracies = {}
    for k in k_values:
        if k > max_k:
            k_eff = max_k
        else:
            k_eff = k
            
        correct = 0
        for i in range(len(y_eval)):
            neighbor_indices = indices[i, :k_eff]
            neighbor_labels = y_train[neighbor_indices]
            if y_eval[i] in neighbor_labels:
                correct += 1
        top_k_accuracies[k] = correct / len(y_eval)
        
    return top_k_accuracies

def plot_line_chart(x_values, y_values, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_bar_chart(categories, values, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.savefig(filename)
    plt.close()

def run_pipeline():
    images_dir = "images"
    user_images = load_data(images_dir)
    
    TUNING_USERS = 100
    results_dir = "evaluation_pipeline_results"
    os.makedirs(results_dir, exist_ok=True)
    # BoVW tuning
    bovw_clusters_list = [50, 500, 1000, 1500, 2000, 2500, 3000]
    best_bovw_acc = -1
    best_bovw_k = 1000
    
    step1_results = {}
    step1_top1_accs = []
    
    for k in bovw_clusters_list:
        print(f"BoVW clusters: {k}")
        data, _ = prepare_datasets(user_images, bovw_clusters=k, limit_users=TUNING_USERS)
        
        X_train = flatten_features(data['train']['features'], use_hog=False, use_bovw=True, use_others=False)
        y_train = data['train']['labels']
        X_test = flatten_features(data['test']['features'], use_hog=False, use_bovw=True, use_others=False)
        y_test = data['test']['labels']

        results = evaluate_model(X_train, y_train, X_test, y_test, use_pca=True, pca_components=0.70)
        acc = results[1] 
        print(f"Accuracy for k={k}: {acc:.4f}")
        step1_results[k] = results
        step1_top1_accs.append(acc)
        
        if acc > best_bovw_acc:
            best_bovw_acc = acc
            best_bovw_k = k
            
    with open(os.path.join(results_dir, "step1_bovw.json"), "w") as f:
        json.dump(step1_results, f, indent=4)
        
    plot_line_chart(bovw_clusters_list, step1_top1_accs, 
                   "BoVW Tuning (Top-1 Accuracy)", "Number of Clusters", "Accuracy", 
                   os.path.join(results_dir, "step1_bovw_tuning.png"))


    print(f"Using BoVW clusters: {best_bovw_k}")
    data, _ = prepare_datasets(user_images, bovw_clusters=best_bovw_k, limit_users=TUNING_USERS)
    
    step2_results = {}
    
    # HOG only
    X_train = flatten_features(data['train']['features'], use_hog=True, use_bovw=False, use_others=False)
    y_train = data['train']['labels']
    X_test = flatten_features(data['test']['features'], use_hog=True, use_bovw=False, use_others=False)
    y_test = data['test']['labels']
    res_hog = evaluate_model(X_train, y_train, X_test, y_test, use_pca=True, pca_components=0.70)
    step2_results['hog_only'] = res_hog
    print(f"HOG only acc: {res_hog[1]:.4f}")
    
    # BoVW only
    X_train = flatten_features(data['train']['features'], use_hog=False, use_bovw=True, use_others=False)
    y_train = data['train']['labels']
    X_test = flatten_features(data['test']['features'], use_hog=False, use_bovw=True, use_others=False)
    y_test = data['test']['labels']
    res_bovw = evaluate_model(X_train, y_train, X_test, y_test, use_pca=True, pca_components=0.70)
    step2_results['bovw_only'] = res_bovw
    print(f"BoVW only acc: {res_bovw[1]:.4f}")
    
    # HOG + BoVW
    X_train = flatten_features(data['train']['features'], use_hog=True, use_bovw=True, use_others=False)
    y_train = data['train']['labels']
    X_test = flatten_features(data['test']['features'], use_hog=True, use_bovw=True, use_others=False)
    y_test = data['test']['labels']
    res_hog_bovw = evaluate_model(X_train, y_train, X_test, y_test, use_pca=True, pca_components=0.70)
    step2_results['hog_bovw'] = res_hog_bovw
    print(f"HOG + BoVW acc: {res_hog_bovw[1]:.4f}")
    
    # all features
    X_train = flatten_features(data['train']['features'], use_hog=True, use_bovw=True, use_others=True)
    y_train = data['train']['labels']
    X_test = flatten_features(data['test']['features'], use_hog=True, use_bovw=True, use_others=True)
    y_test = data['test']['labels']
    res_all = evaluate_model(X_train, y_train, X_test, y_test, use_pca=True, pca_components=0.70)
    step2_results['all_features'] = res_all
    print(f"All features acc: {res_all[1]:.4f}")
    
    with open(os.path.join(results_dir, "step2_features.json"), "w") as f:
        json.dump(step2_results, f, indent=4)
        
    categories = ['HOG Only', 'BoVW Only', 'HOG + BoVW', 'All Features']
    values = [res_hog[1], res_bovw[1], res_hog_bovw[1], res_all[1]]
    plot_bar_chart(categories, values, "Feature Importance (Top-1 Accuracy)", "Accuracy", 
                   os.path.join(results_dir, "step2_feature_importance.png"))

    # Others only
    others_users_list = [20, 40, 60, 80, 100]
    others_results = {}
    others_top1_accs = []
    
    for u in others_users_list:
        print(f"Others only with {u} users")
        data_others, _ = prepare_datasets(user_images, bovw_clusters=50, limit_users=u)
        
        X_train = flatten_features(data_others['train']['features'], use_hog=False, use_bovw=False, use_others=True)
        y_train = data_others['train']['labels']
        X_test = flatten_features(data_others['test']['features'], use_hog=False, use_bovw=False, use_others=True)
        y_test = data_others['test']['labels']

        res = evaluate_model(X_train, y_train, X_test, y_test, use_pca=False)
        others_results[u] = res
        others_top1_accs.append(res[1])
        print(f"Others only ({u} users) acc: {res[1]:.4f}")
        
    with open(os.path.join(results_dir, "step2_others_only.json"), "w") as f:
        json.dump(others_results, f, indent=4)
        
    plot_line_chart(others_users_list, others_top1_accs, 
                   "Others Only Features Performance", "Number of Users", "Accuracy", 
                   os.path.join(results_dir, "step2_others_only.png"))

    # PCA tuning
    data, _ = prepare_datasets(user_images, bovw_clusters=best_bovw_k, limit_users=TUNING_USERS)
    
    X_train = flatten_features(data['train']['features'], use_hog=True, use_bovw=True, use_others=True)
    y_train = data['train']['labels']
    X_test = flatten_features(data['test']['features'], use_hog=True, use_bovw=True, use_others=True)
    y_test = data['test']['labels']
    
    pca_components_list = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    best_pca_acc = -1
    best_pca_comp = 0.70
    
    step3_results = {}
    step3_top1_accs = []
    
    for comp in pca_components_list:
        res = evaluate_model(X_train, y_train, X_test, y_test, use_pca=True, pca_components=comp)
        acc = res[1]
        step3_results[str(comp)] = res
        step3_top1_accs.append(acc)
        
        if acc > best_pca_acc:
            best_pca_acc = acc
            best_pca_comp = comp
            
    print(f"Best PCA comps: {best_pca_comp} (Acc: {best_pca_acc:.4f})")
    with open(os.path.join(results_dir, "step3_pca.json"), "w") as f:
        json.dump(step3_results, f, indent=4)
        
    plot_line_chart(pca_components_list, step3_top1_accs, 
                   "PCA Tuning (Top-1 Accuracy)", "Explained Variance Ratio", "Accuracy", 
                   os.path.join(results_dir, "step3_pca_tuning.png"))

 
    user_counts = [50, 100, 150, 200, 250, 300, 352]
    
    step4_results = {}
    step4_top1_accs = []
    
    for count in user_counts:
        print(f"{count} users...")
        data, _ = prepare_datasets(user_images, bovw_clusters=best_bovw_k, limit_users=count)
        
        X_train = flatten_features(data['train']['features'], use_hog=True, use_bovw=True, use_others=True)
        y_train = data['train']['labels']
        X_final_test = flatten_features(data['final_test']['features'], use_hog=True, use_bovw=True, use_others=True)
        y_final_test = data['final_test']['labels']
      
        res = evaluate_model(X_train, y_train, X_final_test, y_final_test, use_pca=True, pca_components=best_pca_comp)
        print(f"Users {count}, final test accuracy {res[1]:.4f}")
        step4_results[count] = res
        step4_top1_accs.append(res[1])
        
    with open(os.path.join(results_dir, "step4_final.json"), "w") as f:
        json.dump(step4_results, f, indent=4)
        
    plot_line_chart(user_counts, step4_top1_accs, 
                   "Final Evaluation (Top-1 Accuracy)", "Number of Users", "Accuracy", 
                   os.path.join(results_dir, "step4_final_evaluation.png"))
        
if __name__ == "__main__":
    run_pipeline()
