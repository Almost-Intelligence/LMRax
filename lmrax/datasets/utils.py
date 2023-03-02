import random

import numpy as np
import torch


def seed_worker(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def shp_to_pf(example):
    context = example["history"]
    chosen = (
        example["human_ref_A"]
        if example["labels"] == 1
        else example["human_ref_B"]
    )
    rejected = (
        example["human_ref_B"]
        if example["labels"] == 1
        else example["human_ref_A"]
    )
    s = example["score_ratio"]
    s = s / (s + 1)
    weight = s
    return {
        "context": context,
        "chosen": chosen,
        "rejected": rejected,
        "weight": weight,
    }


def hh_to_pf(example):
    chosen_turns = [
        x.strip()
        for x in example["chosen"].split("\n\nAssistant")
        if len(x.strip()) > 0
    ]
    rejected_turns = [
        x.strip()
        for x in example["rejected"].split("\n\nAssistant")
        if len(x.strip()) > 0
    ]

    context_idx = 0
    for i in range(min(len(chosen_turns), len(rejected_turns))):
        if chosen_turns[i] == rejected_turns[i]:
            context_idx = i + 1
        else:
            break
    if not 0 < context_idx < min(len(chosen_turns), len(rejected_turns)):
        assert example["chosen"] == example["rejected"]
        context_idx -= 1

    context = "\n\nAssistant".join(chosen_turns[:context_idx]).strip()
    chosen = "\n\nAssistant".join([""] + chosen_turns[context_idx:]).strip()
    rejected = "\n\nAssistant".join(
        [""] + rejected_turns[context_idx:]
    ).strip()

    weight = 1.0

    return {
        "context": context,
        "chosen": chosen,
        "rejected": rejected,
        "weight": weight,
    }


def get_filter_fn(cfg):
    name = cfg.dataset.name
    if name == "stanfordnlp/SHP":
        return lambda x: x["score_ratio"] >= cfg.dataset.score_ratio
    elif name == "Anthropic/hh-rlhf":
        return lambda x: x["chosen"] != x["rejected"]
    else:
        return lambda x: True


def get_map_fn(cfg):
    name = cfg.dataset.name
    if name == "stanfordnlp/SHP":
        return shp_to_pf
    elif name == "Anthropic/hh-rlhf":
        return hh_to_pf
    else:
        raise NotImplementedError(
            f"Dataset {name} is not supported yet. "
            f"Please open an issue. "
            f"You can also implement it and send a PR."
        )
