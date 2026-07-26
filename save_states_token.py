import av
import os
import re
import ast
import glob
import json
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import set_random_seed, read_frames_from_pngs, ensure_dir, split_prompt_map


def parse_timeline(timeline_text):
    timeline = []
    for line in timeline_text.splitlines():
        match = re.match(r"(\d{2})m:(\d{2})s \| (.*)", line.strip())
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            action = match.group(3)
            time_in_seconds = minutes * 60 + seconds
            timeline.append((time_in_seconds, action))
    return timeline

def time_to_frame_index(time_in_seconds, frame_rate):
    return int(time_in_seconds * frame_rate)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='llava-hf/llava-v1.6-mistral-7b-hf')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--temperature', default=0, type=float)
    parser.add_argument('--image_process_mode', type=str, default='Default')
    parser.add_argument('--max_new_tokens', default=0, type=int)
    parser.add_argument('--indice_num', default=0, type=int)
    parser.add_argument('--prompt_path', type=str)
    parser.add_argument('--calibration_ratio', default=0.3, type=float)
    parser.add_argument('--split_seed', default=0, type=int)
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--preview', dest='preview', type=ast.literal_eval)
    parser.add_argument('--output_dir', type=str, default='results')

    args = parser.parse_args()
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    return args

def save_states(args):
    if "LLaVA-NeXT-Video" in args.model:
        from vlm import VLM
    else:
        raise NotImplementedError(
            "The core implementation supports Qwen2.5-VL and LLaVA-NeXT-Video."
        )
    prompt_path = args.prompt_path
    video_path = args.video_path

    prompts = json.load(open(prompt_path))
    prompts, _ = split_prompt_map(prompts, args.calibration_ratio, args.split_seed)

    vlm = VLM(args)
    with tqdm(total=len(prompts['goal'])+len(prompts['belief'])+len(prompts['actions'])) as pbar:
        for goal in ['goal', 'belief', 'actions']:
            for cuid in prompts[goal]:


                context = prompts[goal][cuid]['context']
                answer = prompts[goal][cuid]['answer']
                choices = prompts[goal][cuid]['choices']


                frames_dir = os.path.join(video_path, f"{cuid}_context")
                frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
                total_frames = len(frame_paths)

                if "Qwen2.5-VL" in args.model:

                    if args.indice_num == -1:

                        timeline = parse_timeline(context)
                        indices = [time_to_frame_index(t[0], 30) for t in timeline]
                    else:

                        indices = np.linspace(0, total_frames-1, args.indice_num).astype(int)
                    clip = [ frame_paths[i] for i in indices ]

                elif "LLaVA-NeXT-Video" in args.model:
                    if args.indice_num == -1:

                        timeline = parse_timeline(context)
                        indices = [time_to_frame_index(t[0], 30) for t in timeline]
                    else:

                        indices = np.linspace(0, total_frames-1, args.indice_num).astype(int)
                    clip = [ frame_paths[i] for i in indices ]
                    clip = read_frames_from_pngs(args, clip)
                else:
                    raise NotImplementedError

                if args.preview:

                    ensure_dir('./frames')
                    for idx, frame in enumerate(clip):
                        img = Image.fromarray(frame)
                        img.save(f"frames/frame_{idx:04d}.png")


                if goal == 'goal':
                    user_text = "C's future goal is " + choices[ord(answer) - ord('a')].lower()
                elif goal == 'belief':
                    user_text = "At the end of these actions, " + choices[ord(answer) - ord('a')].lower()
                elif goal == 'actions':
                    user_text = "Next, "+ choices[ord(answer) - ord('a')].lower()

                vlm.create_conversation("", user_text, clip)
                attn_list = vlm.forward()
                np.save(os.path.join(args.output_dir, 'attn', 'token', model_name, goal, 'correct', f'{cuid}_attn.npy'), attn_list)


                for i in range(len(choices)):
                    if i == ord(answer) - ord('a'):
                        continue
                    if goal == 'goal':
                        user_text = "C's future goal is " + choices[i].lower()
                    elif goal == 'belief':
                        user_text = "At the end of these actions, " + choices[i].lower()
                    elif goal == 'actions':
                        user_text = "Next, "+ choices[i].lower()
                    
                    vlm.create_conversation("", user_text, clip)
                    attn_list = vlm.forward()
                    np.save(os.path.join(args.output_dir, 'attn', 'token', model_name, goal, 'incorrect', f'{cuid}_attn_{i}.npy'), attn_list)
                pbar.update(1)
                
if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    model_name = args.model.split('/')[-1]
    for goal in ['goal', 'belief', 'actions']:
        if os.path.exists(os.path.join(args.output_dir, 'attn', 'token', model_name, goal)):
            if input("Do you want to delete the previous attn folder? (y/n): ").strip().lower() == 'y':
                shutil.rmtree(os.path.join(args.output_dir, 'attn', 'token', model_name, goal))
        os.makedirs(os.path.join(args.output_dir, 'attn', 'token', model_name, goal, 'correct'))
        os.makedirs(os.path.join(args.output_dir, 'attn', 'token', model_name, goal, 'incorrect'))
    save_states(args)
