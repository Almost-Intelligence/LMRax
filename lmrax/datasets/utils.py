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
    chosen = example["chosen"].strip()
    rejected = example["rejected"].strip()

    context = " ".join(chosen.split(" Assistant: ")[:-1]).strip()
    chosen = chosen.split(" Assistant: ")[-1].strip()
    rejected = rejected.split(" Assistant: ")[-1].strip()

    weight = 1.0
    return {
        "context": context,
        "chosen": chosen,
        "rejected": rejected,
        "weight": weight,
    }
