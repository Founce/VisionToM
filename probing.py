import warnings; warnings.filterwarnings("ignore")

import os, json, random, argparse, shutil, copy
import numpy as np
from tqdm import tqdm

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from utils import ensure_dir, set_random_seed, split_prompt_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="llava-hf/llava-v1.6-mistral-7b-hf")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--prompt_path", required=True, help="JSON file containing the sample mapping for each task")
    p.add_argument("--calibration_ratio", type=float, default=0.3)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--goals", nargs="+", default=["goal","belief","actions"])

    p.add_argument("--correct_suffixes", nargs="+", default=["attn"])
    p.add_argument("--task", type=str)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_size", type=float, default=0.25)
    return p.parse_args()


def flatten_if_needed(attn: np.ndarray) -> np.ndarray:
    if attn.ndim == 4:
        attn = attn.mean(axis=(2,3))
        attn = attn[..., None]
    elif attn.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported shape {attn.shape}")
    return attn


def load_dataset(args):
    model_name = args.model.split("/")[-1]
    with open(args.prompt_path, "r") as f:
        prompt_map = json.load(f)
    prompt_map, _ = split_prompt_map(prompt_map, args.calibration_ratio, args.split_seed)

    X, y = [], []

    for goal in args.goals:
        cuids = prompt_map[goal]
        for cuid in tqdm(cuids, desc=f"Load {goal}"):

            for suf in args.correct_suffixes:
                fn = f"{cuid}_{suf}.npy"

                if suf == "attn":
                    path = os.path.join(args.output_dir, "attn", args.task, model_name, goal, "correct", fn)
                else:
                    path = os.path.join(args.output_dir, "reattn", args.task, model_name, goal, "correct", fn)
                if not os.path.exists(path):
                    continue
                X.append(flatten_if_needed(np.load(path)))
                y.append(1)


            dir_inc = os.path.join(args.output_dir, "attn", args.task, model_name, goal, "incorrect")
            if args.task == "vision":
                X.append(flatten_if_needed(np.load(os.path.join(dir_inc, f"{cuid}_attn.npy"))))
                y.append(0)
            elif args.task == "token":
                for fn in os.listdir(dir_inc):
                    if fn.startswith(f"{cuid}_attn_") and fn.endswith(".npy"):
                        X.append(flatten_if_needed(np.load(os.path.join(dir_inc, fn))))
                        y.append(0)

    X = np.stack(X)
    y = np.array(y).astype(bool)
    return X, y


def probe_single_case(X_tr, y_tr, X_val, y_val, seed=0):
    clf = LogisticRegression(max_iter=100, C=0.001, random_state=seed).fit(X_tr, y_tr)
    y_val_prob = clf.predict_proba(X_val)[:,1]
    return (accuracy_score(y_tr, clf.predict(X_tr)),
            accuracy_score(y_val, clf.predict(X_val)),
            roc_auc_score(y_val, y_val_prob),
            log_loss(y_val,  y_val_prob),
            clf)


def probe_all(args, X, y):
    ids = np.arange(len(X))
    X_tr, X_val, y_tr, y_val, ids_tr, ids_val = train_test_split(
        X, y, ids, test_size=args.test_size, random_state=args.seed, stratify=y)

    L, H, S = X_tr.shape[1:]
    train_acc = np.zeros((L,H)); val_acc = train_acc.copy()
    auc = train_acc.copy(); loss = train_acc.copy(); coef = np.zeros((L,H,S))

    for l in tqdm(range(L), desc="Probe"):
        for h in range(H):
            tr, va = X_tr[:,l,h,:], X_val[:,l,h,:]
            tr_acc, va_acc, va_auc, va_loss, clf = probe_single_case(tr, y_tr, va, y_val, args.seed)
            train_acc[l,h], val_acc[l,h], auc[l,h], loss[l,h] = tr_acc, va_acc, va_auc, va_loss
            coef[l,h] = clf.coef_[0]
    return train_acc, val_acc, auc, loss, coef


def plot_heatmap(mat, title, save_path):
    plt.rcParams.update({'font.size':22})
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(mat, ax=ax, cmap="Greens", vmin=0.75, vmax=1, square=True,
                cbar_kws={'drawedges':False})
    ax.set_title(title, pad=16);  plt.tight_layout()
    if save_path: plt.savefig(save_path); plt.close(fig)


def probe_and_save(args, X, y, tag="combined"):
    tr_acc, va_acc, auc, loss, coef = probe_all(args, X, y)

    save_dir = os.path.join(args.output_dir, "probe", args.task, tag)
    ensure_dir(save_dir)

    plot_heatmap(tr_acc, "Probe Train Acc.", os.path.join(save_dir, "train_acc.pdf"))
    plot_heatmap(va_acc, "Probe Val Acc.", os.path.join(save_dir, "val_acc.pdf"))


    np.save(os.path.join(save_dir, "val_acc.npy"), va_acc)
    np.save(os.path.join(save_dir, "coef.npy"),    coef)


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)


    for goal in args.goals:
        args_goal = copy.deepcopy(args)
        args_goal.goals = [goal]
        X_g, y_g = load_dataset(args_goal)
        probe_and_save(args_goal, X_g, y_g, tag=goal)

    if args.task == "vision":
        combined_args = copy.deepcopy(args)
        combined_args.goals = ["goal", "belief", "actions"]
        X_all, y_all = load_dataset(combined_args)
        probe_and_save(combined_args, X_all, y_all, tag="combined")
