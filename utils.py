import os
import torch
import random
import string
import numpy as np
from PIL import Image


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def split_prompt_map(prompt_map, calibration_ratio=0.3, split_seed=0):
    if not 0 < calibration_ratio < 1:
        raise ValueError("calibration_ratio must be between 0 and 1.")

    calibration = {}
    evaluation = {}
    for task in ("goal", "belief", "actions"):
        task_samples = prompt_map[task]
        sample_ids = list(task_samples)
        if len(sample_ids) < 2:
            raise ValueError(f"Task '{task}' needs at least two samples for a disjoint split.")

        shuffled_ids = sample_ids.copy()
        random.Random(f"{split_seed}:{task}").shuffle(shuffled_ids)
        calibration_size = int(len(shuffled_ids) * calibration_ratio)
        calibration_size = max(1, min(len(shuffled_ids) - 1, calibration_size))
        calibration_ids = set(shuffled_ids[:calibration_size])

        calibration[task] = {
            sample_id: task_samples[sample_id]
            for sample_id in sample_ids
            if sample_id in calibration_ids
        }
        evaluation[task] = {
            sample_id: task_samples[sample_id]
            for sample_id in sample_ids
            if sample_id not in calibration_ids
        }

    return calibration, evaluation

def read_frames_from_pngs(args, frame_paths):
    if args.image_process_mode == "Default":
        frames = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            frames.append(np.array(img))
        return np.stack(frames, axis=0)
    else:
        raise ValueError("Unsupported image process mode: %s" % (args.image_process_mode,))

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def colorize(text, color):
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "end": "\033[0m",
    }
    return colors[color] + text + colors["end"]

def print_colored(text, color):
    print(colorize(text, color))

def normalize_vectors(arr):

    norms = np.linalg.norm(arr, axis=-1, keepdims=True)


    norms[norms == 0] = 1


    normalized_arr = arr / norms
    return normalized_arr

def find_largest_k_items(arr, k):

    indices = np.unravel_index(np.argsort(arr.ravel())[-k:], arr.shape)
    

    largest_items = [(index, arr[index]) for index in zip(*indices)]
    return largest_items[::-1]

def get_interventions_dict(all_activations, top_heads, directions):
    device = "cuda"
    directions = normalize_vectors(directions)
    interventions_dict = {}
    for (layer, head), val_acc in top_heads:
        dir = directions[layer, head]
        activations = all_activations[:, layer, head, :]
        proj_vals = activations @ dir.T
        proj_val_std = np.std(proj_vals)
    

        if layer not in interventions_dict:
            interventions_dict[layer] = []
        dir = torch.tensor(dir).to(device)

        interventions_dict[layer].append((head, dir, proj_val_std, val_acc))
    return interventions_dict

def get_vector(all_activations, labels):
    correct = []
    incorrect = []

    for acts, label in zip(all_activations, labels):
        if label == 1:
            correct.append(acts)
        elif label == 0:
            incorrect.append(acts)
        else:
            raise ValueError("Label should be 0 or 1.")

    correct   = np.asarray(correct)
    incorrect = np.asarray(incorrect)

    if correct.size == 0 or incorrect.size == 0:
        raise ValueError("No correct or incorrect activations found.")

    correct_mean   = correct.mean(axis=0)
    incorrect_mean = incorrect.mean(axis=0)
    diff = correct_mean - incorrect_mean
    return diff
