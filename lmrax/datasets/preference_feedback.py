from dataclasses import dataclass
from typing import Optional

import numpy as np
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class FlaxDataCollatorForSeq2SeqPF(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    padding: str = "max_length"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    truncation: bool = True
    return_tensors: str = "np"

    def numpy_call(self, features):
        kwargs = {
            "padding": self.padding,
            "max_length": self.max_length,
            "truncation": self.truncation,  # TODO(yongchanghao): check this
            "pad_to_multiple_of": self.pad_to_multiple_of,
            "return_tensors": "np",
        }

        context = [example["context"] for example in features]
        chosen = [example["chosen"] for example in features]
        rejected = [example["rejected"] for example in features]
        weight = [example["weight"] for example in features]
        self.tokenizer.truncation_side = self.tokenizer.padding_side = "left"
        context = self.tokenizer(context, **kwargs)
        self.tokenizer.truncation_side = self.tokenizer.padding_side = "right"
        chosen = self.tokenizer(chosen, **kwargs)
        rejected = self.tokenizer(rejected, **kwargs)

        batch = {
            "context": context.data,
            "chosen": chosen.data,
            "rejected": rejected.data,
            "weight": np.array(weight),
        }

        return batch
