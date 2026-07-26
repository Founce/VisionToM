import os
import json
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

def flatten_if_needed(attn: np.ndarray) -> np.ndarray:
    if attn.ndim == 4:
        attn = attn.mean(axis=(2,3))
        attn = attn[..., None]
    elif attn.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported shape {attn.shape}")
    return attn

def parse_chat_response(response):
    answer_idx = response.find('ASSISTANT:')
    response = response[answer_idx+10:].strip().rstrip()
    return response.replace("```json", "").replace("```", "")

def attn_hook(module, input, output, head_dim, layer_id, alpha, interventions_dict=None, token_interventions_list=[], beta=None, dnet=None):

    if interventions_dict is not None and layer_id in interventions_dict:
        device = output[0].device
        for (head, dir, std, _) in interventions_dict[layer_id]:
            if not isinstance(dir, torch.Tensor):
                dir = torch.tensor(dir, device=device)
            else:
                dir = dir.to(device)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std, device=device)
            else:
                std = std.to(device)
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.to(device)
            output[0][0, -1, head * head_dim: (head + 1) * head_dim] += alpha * std * dir


    if len(token_interventions_list) > 0:
        for token_interventions_dict in token_interventions_list:
            if layer_id in token_interventions_dict:
                device = output[0].device
                if dnet is not None:
                    for (head, _, _, _) in token_interventions_dict[layer_id]:
                        if isinstance(beta, torch.Tensor):
                            beta = beta.to(device)
                        hidden_states = torch.as_tensor(output[0][0, -1, head * head_dim: (head + 1) * head_dim], device=device, dtype=next(dnet.model.parameters()).dtype)
                        delta = torch.nn.functional.normalize(dnet.inference(hidden_states, layer_id, head), dim=-1).half()

                        output[0][0, -1, head * head_dim: (head + 1) * head_dim] += beta * delta
                else:
                    for (head, dir, std, _) in token_interventions_dict[layer_id]:
                        if not isinstance(dir, torch.Tensor):
                            dir = torch.tensor(dir, device=device)
                        else:
                            dir = dir.to(device)
                        if not isinstance(std, torch.Tensor):
                            std = torch.tensor(std, device=device)
                        else:
                            std = std.to(device)
                        if isinstance(beta, torch.Tensor):
                            beta = beta.to(device)
                        output[0][0, -1, head * head_dim: (head + 1) * head_dim] += beta * std * dir

    return output

class VLM():
    def __init__(self, args):
        model_dir = args.model
        self.args = args
        self.args.output_dir = self.args.output_dir.replace("EgoToM_results_open_ended", "EgoToM_results")
        self.processor = LlavaNextVideoProcessor.from_pretrained(model_dir, device_map="auto")
        self.processor.patch_size = 14
        self.processor.vision_feature_select_strategy = "default"
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)

        self.model.config.return_dict = True
        self.model.config.output_attentions = False
        self.model.config.output_hidden_states = False
        self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.eos_token_id

        if args.temperature == 0:
            self.model.config.do_sample = False
            self.model.config.temperature = None
            self.model.config.top_p = None
            self.model.config.top_k = None
            self.model.config.eos_token_id = None
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id


    def create_conversation(self, system_text, user_text, video):
        if system_text == "":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "video"},
                    ],
                },
            ]
        else:
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_text}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "video"},
                    ],
                },
            ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        self.inputs = self.processor(text=prompt, videos=video, return_tensors="pt", padding=False, truncation=False).to("cuda:0")

    def forward(self):
        with torch.no_grad():
            output = self.model(
                    **self.inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True
                )
            attentions = output.hidden_states[1:]
            all_attention_states = []
            for layer in attentions:
                atts = layer[0].cpu().numpy()
                all_attention_states.append(atts.reshape(atts.shape[0], 32, -1))
            all_attention_states = np.array(all_attention_states)

            all_attention_states = all_attention_states[:,-1]
            if np.isnan(all_attention_states).any():
                raise ValueError("Has Value NAN.")
            return all_attention_states
        

    def generate(self, top_k: int = 10, show_probs: bool = False):
        with torch.no_grad():
            out = self.model.generate(
                **self.inputs,
                max_new_tokens=self.args.max_new_tokens,
                output_scores=show_probs,
                return_dict_in_generate=True
            )


            if show_probs:

                logits_seq = torch.stack(out.scores, dim=1)[0]
                probs_seq  = F.softmax(logits_seq, dim=-1)


                new_token_ids = out.sequences[0][self.inputs["input_ids"].shape[-1]:]

                for step, (tok_id, step_probs) in enumerate(zip(new_token_ids, probs_seq), start=1):
                    chosen_prob = step_probs[tok_id].item()
                    topk_vals, topk_ids = torch.topk(step_probs, k=top_k)


                    def _decode(ids):
                        return self.processor.decode([ids] if isinstance(ids, int) else ids)

                    print(f"\nStep {step:02d}: '{_decode(tok_id)}'  P={chosen_prob:.4f}")
                    for rank, (pid, pval) in enumerate(zip(topk_ids.tolist(), topk_vals.tolist()), start=1):
                        mark = "-" if pid == tok_id else " "
                        print(f"   {rank:>2}{mark} {_decode(pid):<12} {pval:.4f}")

            decoded = self.processor.batch_decode(
                out.sequences,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]


            return parse_chat_response(decoded)

    def load_dataset(self, goal, task):
        from utils import split_prompt_map
        model_name = self.args.model.split("/")[-1]
        with open(self.args.prompt_path, "r") as f:
            prompt_map = json.load(f)
        prompt_map, _ = split_prompt_map(
            prompt_map,
            self.args.calibration_ratio,
            self.args.split_seed
        )

        X, y = [], []

        cuids = prompt_map[goal]
        for cuid in tqdm(cuids, desc=f"Load {goal}"):

            for suf in self.args.correct_suffixes:
                fn = f"{cuid}_{suf}.npy"

                if suf == "attn":
                    path = os.path.join(self.args.output_dir, "attn", task, model_name, goal, "correct", fn)
                else:
                    path = os.path.join(self.args.output_dir, "reattn", task, model_name, goal, "correct", fn)
                if not os.path.exists(path):
                    continue
                X.append(flatten_if_needed(np.load(path)))
                y.append(1)


            dir_inc = os.path.join(self.args.output_dir, "attn", task, model_name, goal, "incorrect")
            if task == "vision":
                X.append(flatten_if_needed(np.load(os.path.join(dir_inc, f"{cuid}_attn.npy"))))
                y.append(0)
            elif task == "token":
                for fn in os.listdir(dir_inc):
                    if fn.startswith(f"{cuid}_attn_") and fn.endswith(".npy"):
                        X.append(flatten_if_needed(np.load(os.path.join(dir_inc, fn))))
                        y.append(0)

        X = np.stack(X)
        y = np.array(y).astype(bool)
        return X, y


    def load_interv(self, goal):
        import os
        from utils import normalize_vectors, get_interventions_dict, find_largest_k_items, get_vector
        print(f"Load '{goal}' intervention.")

        if self.args.direction in ("coef", "random", "vision_random", "vision", "mixture"):
            vision_all_activations, vision_labels = self.load_dataset(goal, "vision")
            vision_val_acc = np.load(os.path.join(self.args.output_dir, "probe", "vision", "combined", "val_acc.npy"))
            vision_top_heads = find_largest_k_items(vision_val_acc, self.args.K)
        
        if self.args.direction in ("coef", "random", "token_random", "token", "token_dnet", "token_dnet_clustered", "token_seq", "mixture"):
            token_all_activations, token_labels = self.load_dataset(goal, "token")
            token_val_acc = np.load(os.path.join(self.args.output_dir, "probe", "token", goal, "val_acc.npy"))
            token_top_heads = find_largest_k_items(token_val_acc, self.args.K)
        self.interventions_dict = None
        self.token_interventions_list = []
        self.dnet = None
    
        if self.args.direction == "coef":
            raise NotImplementedError("Coef direction is not supported in this version.")
        
        elif self.args.direction == "mixture":

            directions = get_vector(all_activations=vision_all_activations, labels=vision_labels)
            self.interventions_dict = get_interventions_dict(vision_all_activations, vision_top_heads, directions=directions)

            from DNet_clustered import ClusteredTopKDirectionPredictor, ClusteredTopKDirectionNet

            topk_info = torch.load(os.path.join(self.args.output_dir, "dnet_clustered", goal, "topk_info.pth"))
            layers = topk_info["layers"]
            heads = topk_info["heads"]
            clusters_per_head = topk_info["clusters_per_head"]
            head_cluster_centers = topk_info["head_cluster_centers"]

            dnet = ClusteredTopKDirectionNet(128, layers.tolist(), heads.tolist(), clusters_per_head).cuda()
            dnet.load_state_dict(torch.load(os.path.join(self.args.output_dir, "dnet_clustered", goal, "encoder_150.pth")))
            self.dnet = ClusteredTopKDirectionPredictor(dnet, head_cluster_centers)

            self.token_interventions_dict = {}
            for (layer, head), val_acc in token_top_heads:
                if layer not in self.token_interventions_dict:
                    self.token_interventions_dict[layer] = []
                self.token_interventions_dict[layer].append((head, None, None, None))
            self.token_interventions_list.append(self.token_interventions_dict)
        else:
            raise NotImplementedError

    def generate_interv(self, top_k: int = 10, show_probs: bool = False):
        hooks = []
        for id, layer in enumerate(self.model.language_model.layers):
            hook = partial(attn_hook, head_dim=self.model.config.text_config.head_dim, layer_id=id, alpha=self.args.alpha, interventions_dict=self.interventions_dict, token_interventions_list=self.token_interventions_list, beta=self.args.beta, dnet=self.dnet)
            hooks.append(layer.self_attn.register_forward_hook(hook))

        with torch.no_grad():
            out = self.model.generate(
                **self.inputs,
                max_new_tokens=self.args.max_new_tokens,
                output_scores=show_probs,
                return_dict_in_generate=True
            )

            if show_probs:
                logits_seq = torch.stack(out.scores, dim=1)[0]
                probs_seq  = F.softmax(logits_seq, dim=-1)

                new_token_ids = out.sequences[0][self.inputs["input_ids"].shape[-1]:]

                for step, (tok_id, step_probs) in enumerate(zip(new_token_ids, probs_seq), start=1):
                    chosen_prob = step_probs[tok_id].item()
                    topk_vals, topk_ids = torch.topk(step_probs, k=top_k)

                    def _decode(ids):
                        return self.processor.decode([ids] if isinstance(ids, int) else ids)

                    print(f"\nStep {step:02d}: '{_decode(tok_id)}'  P={chosen_prob:.4f}")
                    for rank, (pid, pval) in enumerate(zip(topk_ids.tolist(), topk_vals.tolist()), start=1):
                        mark = "-" if pid == tok_id else " "
                        print(f"   {rank:>2}{mark} {_decode(pid):<12} {pval:.4f}")
            decoded = self.processor.batch_decode(out.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            for hook in hooks:
                hook.remove()
            return parse_chat_response(decoded)
