import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import set_random_seed, ensure_dir, find_largest_k_items, split_prompt_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--calibration_ratio", type=float, default=0.3)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--correct_suffixes", nargs="+", default=["attn"])
    parser.add_argument("--task", type=str, default="token")
    parser.add_argument("--goals", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    

    parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters; determined automatically when omitted")
    parser.add_argument("--min_samples_per_cluster", type=int, default=5, help="Minimum number of samples per cluster")
    parser.add_argument("--max_clusters", type=int, default=20, help="Maximum number of clusters")
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans"], help="Clustering method")
    
    args = parser.parse_args()
    
    print("Parsed args:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    return args


def load_dataset_with_clustering(args, goals):
    print(f"Loading dataset for goals: {goals}")
    token_val_acc = np.load(os.path.join(args.probe_path, goals[0] if len(goals) == 1 else "combined", "val_acc.npy"))
    token_top_heads = find_largest_k_items(token_val_acc, args.top_k)
    loc, _ = zip(*token_top_heads)
    layers, heads = zip(*loc)
    layers = np.array(layers, dtype=int)
    heads = np.array(heads, dtype=int)
    model_name = args.model.split("/")[-1]

    with open(args.prompt_path, "r") as f:
        prompt_map = json.load(f)
    prompt_map, _ = split_prompt_map(prompt_map, args.calibration_ratio, args.split_seed)

    X = []
    all_negative_samples = []

    for goal in goals:
        for cuid in tqdm(prompt_map[goal], desc=f"Load {goal}"):
            x_pos, x_neg = [], []
            

            for suf in args.correct_suffixes:
                fn = f"{cuid}_{suf}.npy"
                root = "attn" if suf == "attn" else "reattn"
                path = os.path.join(args.output_dir, root, args.task, model_name, goal, "correct", fn)
                if os.path.exists(path):
                    arr = np.load(path)
                    arr = arr[layers, heads, :]
                    x_pos.append(arr)


            dir_inc = os.path.join(args.output_dir, "attn", args.task, model_name, goal, "incorrect")
            if args.task == "vision":
                arr = np.load(os.path.join(dir_inc, f"{cuid}_attn.npy"))
                x_neg.append(arr[layers, heads, :])
                all_negative_samples.append(arr[layers, heads, :])
            elif args.task == "token":
                for fn in os.listdir(dir_inc):
                    if fn.startswith(f"{cuid}_attn_") and fn.endswith(".npy"):
                        arr = np.load(os.path.join(dir_inc, fn))
                        x_neg.append(arr[layers, heads, :])
                        all_negative_samples.append(arr[layers, heads, :])
                        
            if x_pos and x_neg:
                X.append({
                    "1": np.stack(x_pos, 0),
                    "0": np.stack(x_neg, 0),
                    "layers": layers,
                    "heads": heads
                })


    if all_negative_samples:
        print(f"\nClustering {len(all_negative_samples)} negative samples...")
        negative_array = np.stack(all_negative_samples)
        

        head_clusters = {}
        head_cluster_centers = {}
        
        for head_idx in range(negative_array.shape[1]):
            head_features = negative_array[:, head_idx, :]
            
            print(f"Clustering head {head_idx}...")
            

            if args.n_clusters is not None:
                n_clusters = args.n_clusters
            else:
                n_clusters = determine_optimal_clusters(head_features, args)
                

            head_labels, head_centers = perform_clustering(
                head_features, n_clusters, args.cluster_method
            )
            
            head_clusters[head_idx] = head_labels
            head_cluster_centers[head_idx] = head_centers
            
            print(f"  Head {head_idx}: {n_clusters} clusters")
        

        X_clustered = reorganize_data_with_head_clusters(X, all_negative_samples, head_clusters, negative_array, layers, heads)
        
        return X_clustered, layers, heads, head_cluster_centers
    else:
        print("Warning: no negative samples found; falling back to the original method")
        return X, layers, heads, None


def determine_optimal_clusters(data, args):
    n_samples = len(data)
    min_k = 2
    max_k = min(args.max_clusters, n_samples // args.min_samples_per_cluster)
    
    if max_k < min_k:
        print(f"Only {n_samples} samples are available; using 2 clusters")
        return 2
    
    print(f"Selecting the number of clusters automatically from {min_k} to {max_k}")
    

    silhouette_scores = []

    wcss_scores = []

    ch_scores = []
    
    k_range = range(min_k, max_k + 1)
    
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(data)
            

            unique_labels = np.unique(labels)
            if len(unique_labels) < k:
                print(f"k={k} produced an empty cluster; skipping")
                silhouette_scores.append(-1)
                wcss_scores.append(float('inf'))
                ch_scores.append(0)
                continue
            

            sil_score = silhouette_score(data, labels)
            silhouette_scores.append(sil_score)
            

            wcss = kmeans.inertia_
            wcss_scores.append(wcss)
            

            from sklearn.metrics import calinski_harabasz_score
            ch_score = calinski_harabasz_score(data, labels)
            ch_scores.append(ch_score)
            
            print(f"k={k}: silhouette={sil_score:.3f}, wcss={wcss:.0f}, ch_index={ch_score:.1f}")
            
        except Exception as e:
            print(f"Failed to evaluate k={k}: {e}")
            silhouette_scores.append(-1)
            wcss_scores.append(float('inf'))
            ch_scores.append(0)
    

    optimal_k = select_optimal_k_comprehensive(k_range, silhouette_scores, wcss_scores, ch_scores)
    
    print(f"Selected number of clusters: {optimal_k}")
    return optimal_k


def select_optimal_k_comprehensive(k_range, sil_scores, wcss_scores, ch_scores):
    k_list = list(k_range)
    

    best_sil_idx = np.argmax(sil_scores)
    best_sil_k = k_list[best_sil_idx]
    best_sil_score = sil_scores[best_sil_idx]
    

    wcss_diffs = []
    for i in range(1, len(wcss_scores)):
        if wcss_scores[i] == float('inf') or wcss_scores[i-1] == float('inf'):
            wcss_diffs.append(0)
        else:
            diff = wcss_scores[i-1] - wcss_scores[i]
            wcss_diffs.append(diff)
    
    if wcss_diffs:
        elbow_idx = np.argmax(wcss_diffs)
        elbow_k = k_list[elbow_idx + 1]
    else:
        elbow_k = k_list[0]
    

    best_ch_idx = np.argmax(ch_scores)
    best_ch_k = k_list[best_ch_idx]
    
    print(f"Best silhouette score: k={best_sil_k} (score={best_sil_score:.3f})")
    print(f"Elbow method recommendation: k={elbow_k}")
    print(f"Best Calinski-Harabasz score: k={best_ch_k}")
    

    if best_sil_score > 0.3:
        return best_sil_k
    

    candidates = {}
    for k in k_list:
        score = 0
        if k == best_sil_k:
            score += 2
        if k == elbow_k:
            score += 1
        if k == best_ch_k:
            score += 1
        if score > 0:
            candidates[k] = score
    
    if candidates:

        max_score = max(candidates.values())
        best_candidates = [k for k, score in candidates.items() if score == max_score]
        return min(best_candidates)
    

    return best_sil_k


def perform_clustering(data, n_clusters, method="kmeans"):
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(data)
        centers = clusterer.cluster_centers_
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    return labels, centers


def analyze_clusters(labels, n_clusters):
    print("\nCluster analysis:")
    total_samples = len(labels)
    for i in range(n_clusters):
        cluster_size = np.sum(labels == i)
        percentage = cluster_size / total_samples * 100
        print(f"  Cluster {i}: {cluster_size} samples ({percentage:.1f}%)")
    

    min_cluster_size = np.min([np.sum(labels == i) for i in range(n_clusters)])
    if min_cluster_size < 3:
        print(f"Warning: the smallest cluster has only {min_cluster_size} samples and may be over-segmented")
    

    cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
    max_size = max(cluster_sizes)
    min_size = min(cluster_sizes)
    imbalance_ratio = max_size / min_size if min_size > 0 else float('inf')
    if imbalance_ratio > 10:
        print(f"Warning: cluster imbalance ratio is {imbalance_ratio:.1f}")


def save_clustering_analysis(output_dir, goals, cluster_labels, n_clusters, cluster_centers):
    goal_name = goals[0] if len(goals) == 1 else "combined"
    analysis_path = os.path.join(output_dir, "dnet_clustered", goal_name, "clustering_analysis.json")
    
    ensure_dir(os.path.dirname(analysis_path))
    

    cluster_stats = {}
    for i in range(n_clusters):
        cluster_size = int(np.sum(cluster_labels == i))
        cluster_percentage = float(cluster_size / len(cluster_labels) * 100)
        cluster_stats[f"cluster_{i}"] = {
            "size": cluster_size,
            "percentage": cluster_percentage
        }
    
    analysis = {
        "n_clusters": n_clusters,
        "total_samples": len(cluster_labels),
        "cluster_stats": cluster_stats,
        "cluster_centers_shape": cluster_centers.shape
    }
    
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Cluster analysis saved to: {analysis_path}")


def reorganize_data_with_head_clusters(X, all_negative_samples, head_clusters, negative_array, layers, heads):
    X_clustered = []
    neg_idx = 0
    
    for sample in X:
        pos_data = sample["1"]
        neg_data = sample["0"]
        sample_layers = sample["layers"]
        sample_heads = sample["heads"]
        

        neg_clusters_per_head = {}
        for head_idx in range(len(sample_heads)):
            neg_clusters_per_head[head_idx] = []
        
        for neg_sample_idx in range(neg_data.shape[0]):
            for head_idx in range(len(sample_heads)):
                cluster_label = head_clusters[head_idx][neg_idx]
                neg_clusters_per_head[head_idx].append(cluster_label)
            neg_idx += 1
            
        X_clustered.append({
            "1": pos_data,
            "0": neg_data,
            "neg_clusters_per_head": neg_clusters_per_head,
            "layers": sample_layers,
            "heads": sample_heads
        })
    
    return X_clustered


class ClusteredDirectionNet(nn.Module):
    def __init__(self, dim: int, n_clusters: int, dropout_rate=0.1):
        super().__init__()
        self.n_clusters = n_clusters
        self.dim = dim
        

        self.cluster_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
            ) for _ in range(n_clusters)
        ])
        

        for net in self.cluster_nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
    
    def forward(self, x: torch.Tensor, cluster_id: int) -> torch.Tensor:
        if cluster_id >= self.n_clusters:
            raise ValueError(f"cluster_id {cluster_id} >= n_clusters {self.n_clusters}")
        return self.cluster_nets[cluster_id](x)


class ClusteredTopKDirectionNet(nn.Module):
    def __init__(self, dim: int, layers: List[int], heads: List[int], clusters_per_head: Dict[int, int]):
        super().__init__()
        
        if len(layers) != len(heads):
            raise ValueError("layers and heads must have the same length")
        
        self.dim = dim
        self.clusters_per_head = clusters_per_head
        self.topk_pairs = list(zip(layers, heads))
        self.k = len(self.topk_pairs)
        

        self.pair_to_idx = {pair: idx for idx, pair in enumerate(self.topk_pairs)}
        

        self.networks = nn.ModuleList([
            ClusteredDirectionNet(dim, clusters_per_head[idx]) for idx in range(self.k)
        ])
    
    def get_network_for_pair(self, layer: int, head: int) -> nn.Module:
        pair = (layer, head)
        if pair not in self.pair_to_idx:
            raise ValueError(f"Unknown (layer, head) pair: {pair}")
        return self.networks[self.pair_to_idx[pair]]
    
    def get_clusters_for_head_idx(self, head_idx: int) -> int:
        return self.clusters_per_head.get(head_idx, 1)
    
    def get_topk_info(self) -> Dict:
        return {
            'topk_pairs': self.topk_pairs,
            'k': self.k,
            'pair_to_idx': self.pair_to_idx,
            'clusters_per_head': self.clusters_per_head
        }


def clustered_mse_loss(pos_np, neg_np, neg_clusters_per_head, layers, heads, model):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    pos = torch.as_tensor(pos_np, dtype=dtype, device=device)
    neg = torch.as_tensor(neg_np, dtype=dtype, device=device)

    total_loss = 0.0
    
    for head_idx in range(pos.size(1)):
        pos_head = pos[:, head_idx, :]
        neg_head = neg[:, head_idx, :]
        

        head_clusters = neg_clusters_per_head[head_idx]
        clusters = torch.as_tensor(head_clusters, dtype=torch.long, device=device)
        
        layer_idx = layers[head_idx].item()
        head_idx_val = heads[head_idx].item()
        network = model.get_network_for_pair(layer_idx, head_idx_val)
        
        head_loss = 0.0
        n_samples = 0
        

        n_clusters = model.get_clusters_for_head_idx(head_idx)
        

        for cluster_id in range(n_clusters):
            cluster_mask = (clusters == cluster_id)
            if not cluster_mask.any():
                continue
                
            neg_cluster = neg_head[cluster_mask]
            target_delta = pos_head - neg_cluster
            pred_delta = network(neg_cluster, cluster_id)
            
            cluster_loss = F.mse_loss(pred_delta, target_delta)
            head_loss += cluster_loss * cluster_mask.sum().float()
            n_samples += cluster_mask.sum().item()
        
        if n_samples > 0:
            total_loss += head_loss / n_samples

    return total_loss / pos.size(1)


def train_clustered(args, goals: list):
    MAX_EPOCH = 200

    X, layers, heads, head_cluster_centers = load_dataset_with_clustering(args, goals)
    if head_cluster_centers is None:
        raise ValueError
        
    dim = X[0]['0'].shape[-1]
    

    clusters_per_head = {}
    for head_idx in range(len(layers)):
        clusters_per_head[head_idx] = head_cluster_centers[head_idx].shape[0]
    
    model = ClusteredTopKDirectionNet(dim, layers.tolist(), heads.tolist(), clusters_per_head).float().cuda()
    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": 1e-3}
    ])
    
    print(f"Created ClusteredTopKDirectionNet with {model.k} (layer, head) pairs")
    print("Clusters per head:", clusters_per_head)
    print("Top-k pairs:", model.topk_pairs[:10], "..." if model.k > 10 else "")
    
    with tqdm(total=MAX_EPOCH) as pbar:
        for epoch in range(MAX_EPOCH):
            total_loss = 0.0
            for hidden in X:
                loss = clustered_mse_loss(
                    hidden["1"], hidden["0"], hidden["neg_clusters_per_head"], 
                    hidden["layers"], hidden["heads"], model)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(X)
            pbar.set_postfix(EPOCH=epoch+1, MSE_LOSS=f"{avg_loss:.6f}")
            

            save_dir = os.path.join(args.output_dir, "dnet_clustered", goals[0] if len(goals) == 1 else "combined")
            ensure_dir(save_dir)
            
            torch.save(model.state_dict(), os.path.join(save_dir, f"encoder_{epoch+1}.pth"))
            torch.save({
                'layers': layers,
                'heads': heads,
                'topk_pairs': model.topk_pairs,
                'pair_to_idx': model.pair_to_idx,
                'clusters_per_head': clusters_per_head,
                'head_cluster_centers': head_cluster_centers
            }, os.path.join(save_dir, "topk_info.pth"))
            
            pbar.update(1)


class ClusteredTopKDirectionPredictor:
    def __init__(self, model: ClusteredTopKDirectionNet, head_cluster_centers: Dict[int, np.ndarray]):
        self.model = model

        self.head_cluster_centers = {}
        for head_idx, centers in head_cluster_centers.items():
            self.head_cluster_centers[head_idx] = torch.tensor(centers, device=next(model.parameters()).device)
        self.model.eval()
    
    def _find_nearest_cluster_for_head(self, feature: torch.Tensor, head_idx: int) -> int:
        if feature.dim() == 1:
            feature = feature.unsqueeze(0)
        

        head_centers = self.head_cluster_centers[head_idx]
        

        distances = torch.norm(feature - head_centers, dim=1)
        return torch.argmin(distances).item()
    
    def inference(self, feature: torch.Tensor, layer_idx: int, head_idx: int) -> torch.Tensor:
        with torch.no_grad():
            if feature.dim() == 1:
                feature = feature.unsqueeze(0)
            feature = feature.to(next(self.model.parameters()).device)
            

            pair = (layer_idx, head_idx)
            if pair not in self.model.pair_to_idx:
                raise ValueError(f"Unknown (layer, head) pair: {pair}")
            
            model_head_idx = self.model.pair_to_idx[pair]
            

            cluster_id = self._find_nearest_cluster_for_head(feature, model_head_idx)
            

            network = self.model.get_network_for_pair(layer_idx, head_idx)
            delta = network(feature, cluster_id).squeeze(0)
            
        return delta


@torch.no_grad()
def compute_clustered_mse_metrics(pos_np, neg_np, neg_clusters, layers, heads, model):
    mse_loss = clustered_mse_loss(pos_np, neg_np, neg_clusters, layers, heads, model)
    
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pos = torch.as_tensor(pos_np, dtype=dtype, device=device)
    neg = torch.as_tensor(neg_np, dtype=dtype, device=device)
    clusters = torch.as_tensor(neg_clusters, dtype=torch.long, device=device)
    
    correction_improvements = []
    
    for head_idx in range(pos.size(1)):
        pos_head = pos[:, head_idx, :]
        neg_head = neg[:, head_idx, :]
        
        orig_dist = F.mse_loss(neg_head, pos_head.expand_as(neg_head))
        
        layer_idx = layers[head_idx].item()
        head_idx_val = heads[head_idx].item()
        network = model.get_network_for_pair(layer_idx, head_idx_val)
        

        corrected_neg = []
        for i, cluster_id in enumerate(clusters):
            delta = network(neg_head[i:i+1], cluster_id.item())
            corrected_neg.append(neg_head[i:i+1] + delta)
        
        corrected_neg = torch.cat(corrected_neg, dim=0)
        corr_dist = F.mse_loss(corrected_neg, pos_head.expand_as(corrected_neg))
        
        improvement = (orig_dist - corr_dist) / orig_dist if orig_dist > 0 else 0
        correction_improvements.append(improvement.item())
    
    avg_improvement = sum(correction_improvements) / len(correction_improvements)
    
    return {
        "mse_loss": mse_loss.item(),
        "correction_improvement": avg_improvement
    }


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)


    for g in args.goals:
        ensure_dir(os.path.join(args.output_dir, "dnet_clustered", g))
    if len(args.goals) == 3:
        ensure_dir(os.path.join(args.output_dir, "dnet_clustered", "combined"))

    train_clustered(args, goals=args.goals)
