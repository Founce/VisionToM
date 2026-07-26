import av
import os
import re
import ast
import csv
import glob
import json
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from rm import rm
from utils import set_random_seed, read_frames_from_pngs, ensure_dir, colorize, split_prompt_map


def write_text_to_csv(data, filename):
    with open(f'{filename}.csv', mode='a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['Cuid', 'Answer', 'Choices', 'Result'])
        writer.writerow(data)


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
    parser.add_argument('--open_ended_mode', dest='open_ended_mode', type=ast.literal_eval, default=False)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--alpha', default=0, type=float)
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--K', default=0, type=int)
    parser.add_argument('--direction', type=str)
    parser.add_argument('--intervene', dest='intervene', type=ast.literal_eval)

    parser.add_argument("--correct_suffixes", nargs="+", default=["attn"])

    args = parser.parse_args()
    if args.open_ended_mode:
        print("\nOpen-ended mode enabled. Output directory defaults to adding suffix '_open_ended'.")
        args.output_dir += '_open_ended'

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    return args

if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    model_name = args.model.split('/')[-1]
    eval_dir = 'interv_evaluate' if args.intervene else 'evaluate'
    
    if args.intervene and args.direction in ["vision_random", "vision"]:
        param_suffix = f"k{args.K}_a{args.alpha}"
    elif args.intervene and args.direction in ["token_random", "token", "token_seq", "token_dnet", "token_dnet_clustered"]:
        param_suffix = f"k{args.K}_b{args.beta}"
    elif args.intervene and args.direction in ["coef", "mixture", "random"]:
        param_suffix = f"k{args.K}_a{args.alpha}_b{args.beta}"
    else:
        param_suffix = ""

    if args.intervene:
        save_path = os.path.join(
            args.output_dir,
            eval_dir,
            model_name,
            args.direction,
            param_suffix
        )
    else:
        save_path = os.path.join(
            args.output_dir,
            eval_dir,
            model_name
        )

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    with open(os.path.join(save_path, 'result.txt'), 'w') as f:
        f.close()
    if "LLaVA-NeXT-Video" in args.model:
        from vlm import VLM
    else:
        raise NotImplementedError(
            "The core implementation supports Qwen2.5-VL and LLaVA-NeXT-Video."
        )
    prompt_path = args.prompt_path
    video_path = args.video_path

    prompts = json.load(open(prompt_path))
    _, prompts = split_prompt_map(prompts, args.calibration_ratio, args.split_seed)

    vlm = VLM(args)
    result = {'goal': [0, 0, 0], 'belief': [0, 0, 0], 'actions': [0, 0, 0]}
    with tqdm(total=len(prompts['goal'])+len(prompts['belief'])+len(prompts['actions'])) as pbar:
        for goal in ['goal', 'belief', 'actions']:
            if args.intervene:
                vlm.load_interv(goal)
            for cuid in prompts[goal]:
                system_text = prompts[goal][cuid]['system']
                user_text = prompts[goal][cuid]['user']
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


                system_text = ""
                if args.open_ended_mode:
                    if goal == 'goal':
                        q = "For the photographer named C of this video, what is C's future goal?"
                    elif goal == 'belief':
                        q = "For the photographer named C of this video, at the end of these actions, what does C most likely believe?"
                    elif goal == 'actions':
                        q= "For the photographer named C of this video, at the end of these actions, what will C most likely do next?"
                    user_text = f'You are an expert video question-answering assistant. Your job is to answer the question as accurately as possible **using evidence from the video**. If the video offers even partial clues, give your best reasoned prediction and label it as “Likely”. Question: {q} Your answer:'
                else:
                    user_text = 'You are an expert at predicting human behavior. Based on the video, select the best answer for each question. Then output the result as valid JSON and nothing else. JSON schema (single question): { "question": <integer> //Video question number, "choice": "A" | "B" | "C" | "D", //The option you finally choose, "explanation": "<string>" //Clear reasons } If there are multiple questions, return a JSON array of the above objects.' + prompts[goal][cuid]['user']
                vlm.create_conversation(system_text, user_text, clip)
                if args.intervene:
                    res = vlm.generate_interv()
                else:
                    res = vlm.generate()

                if args.open_ended_mode:
                    write_text_to_csv([cuid, answer, choices, res], os.path.join(save_path, f'{goal}'))
                    result[goal][0] += 1
                else:
                    rm_result = rm(choices, answer, res)
                    if rm_result == 'Right':
                        write_text_to_csv([cuid, answer, choices, res.lower()], os.path.join(save_path, f'{goal}_right'))
                        result[goal][0] += 1
                    elif rm_result == 'Wrong':
                        write_text_to_csv([cuid, answer, choices, res.lower()], os.path.join(save_path, f'{goal}_wrong'))
                        result[goal][1] += 1
                    elif rm_result == 'Error':
                        write_text_to_csv([cuid, answer, choices, res.lower()], os.path.join(save_path, f'{goal}_error'))
                        result[goal][2] += 1
                
                desc = colorize(f"Right: Wrong: Error = ({result[goal][0]}: {result[goal][1]}: {result[goal][2]}) / {sum(result[goal])}. Acc: {result[goal][0] / sum(result[goal]) * 100:.1f}%", "red")
                pbar.set_description(desc)
                pbar.update(1)


                if (sum(result[goal]) >= 50 and result[goal][2] == sum(result[goal])):
                    break

            with open(os.path.join(save_path, 'result.txt'), 'a') as f:
                f.write(f"Goal: {goal}\n")
                f.write(f"Right: {result[goal][0]}\n")
                f.write(f"Wrong: {result[goal][1]}\n")
                f.write(f"Error: {result[goal][2]}\n")
                f.write(f"Accuracy: {result[goal][0] / sum(result[goal]) * 100:.1f}%\n")
            print("=====================================")
            print(f"Goal: {goal}")
            print(f"Right: {result[goal][0]}")
            print(f"Wrong: {result[goal][1]}")
            print(f"Error: {result[goal][2]}")
            print(f"Accuracy: {result[goal][0] / sum(result[goal]) * 100:.1f}%")
            print("=====================================")
