import copy
from typing import TYPE_CHECKING, Any, List
from warnings import warn
import math

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

SUPPORTED_DATASET_CONFIG: dict[str, Any] = {
    "magpie": {
        "config": {"path": "Magpie-Align/Magpie-Pro-MT-300K-v0.1"},
        "target": "conversations",
        "preprocess": lambda sample: "\n".join(turn["value"] for turn in sample),
    },
    "cnn_dailymail": {
        "config": {"path": "cnn_dailymail", "name": "3.0.0"},
        "target": "article",
    },
    "pile": {
        "config": {"path": "monology/pile-uncopyrighted"},
        "target": "text",
    },
    "pg19": {
        "config": {"path": "pg19"},
        "target": "text",
    },
    "wikipedia": {
        "config": {"path": "wikipedia", "name": "20220301.en"},
        "target": "text",
    },
    "c4": {
        "config": {"path": "c4", "name": "en"},
        "target": "text",
    },
}

__all__ = [
    "get_dataset_dataloader",
    "get_supported_datasets",
]


def get_supported_datasets() -> list[str]:
    """Retrieves a list of datasets supported.

    Returns:
        A list of strings, where each string is the name of a supported dataset.

    Example usage:

        .. code-block:: python

            from modelopt.torch.utils import get_supported_datasets

            print("Supported datasets:", get_supported_datasets())
    """
    return list(SUPPORTED_DATASET_CONFIG.keys())



def _get_dataset_samples(dataset_name: str, num_samples: int) -> list[str]:
    """Load a portion of train dataset with the dataset name and a given size.

    Args:
        dataset_name: Name of the dataset to load.
        num_samples: Number of samples to load from the dataset.

    Returns:
        Samples: The list of samples.
    """
    # Load the dataset
    if dataset_name in SUPPORTED_DATASET_CONFIG:
        from datasets import load_dataset

        dataset_config = SUPPORTED_DATASET_CONFIG[dataset_name]
        dataset = load_dataset(
            split="train",
            streaming=True,
            **dataset_config["config"],
        )
    else:
        raise NotImplementedError(
            f"dataset {dataset_name} is not supported. Please use one of the following:"
            f" {get_supported_datasets()}."
        )

    # Access only the required samples
    samples = []
    target_key = dataset_config["target"]
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        # Get raw value
        value = sample[target_key]

        # Apply preprocessing if defined
        if "preprocess" in dataset_config:
            value = dataset_config["preprocess"](value)

        samples.append(value)

    return samples

class _CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {
            key: val[idx] if torch.is_tensor(val[idx]) else torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        return item

    def __len__(self):
        return len(next(iter(self.encodings.values())))

def get_dataset_dataloader(
    dataset_name: str = "cnn_dailymail",
    tokenizer: "PreTrainedTokenizerBase | None" = None,
    batch_size: int = 1,
    num_samples: int = 512,
    max_sample_length: int = 512,
    device: str | None = None,
    include_labels: bool = False,
) -> List[torch.Tensor]:
    """Get a dataloader with the dataset name and toknizer of the target model.

    Args:
        dataset_name: Name of the dataset to load.
        tokenizer: Instancne of Hugginface tokenizer.
        batch_size: Batch size of the returned dataloader.
        num_samples: Number of samples from the dataset.
        max_sample_length: Maximum length of a sample.
        device: Target device for the returned dataloader.
        include_labels: Whether to include labels in the dataloader.

    Returns:
        A instance of dataloader.
    """
    assert tokenizer is not None, "Please provide a tokenizer."
    # batch_encode_plus will modify the tokenizer in place, so we need to clone it.
    tokenizer = copy.deepcopy(tokenizer)

    if tokenizer.padding_side != "left":
        warn(
            "Tokenizer with the right padding_side may impact calibration accuracy. Recommend set to left"
        )

    num_samples = math.ceil(num_samples / batch_size) * batch_size

    dataset = _get_dataset_samples(dataset_name, num_samples=num_samples)

    batch_encoded = tokenizer.batch_encode_plus(
        dataset,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_sample_length,
    )
    if device:
        batch_encoded = batch_encoded.to(device)

    if include_labels:
        # Labels are needed when backward is called in the model.
        # The labels should be a shifted version of the input_ids.
        # However, we should not shift the input_ids here since the labels are shifted by
        # Huggingface models during loss calculation as shown here -
        # https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/llama/modeling_llama.py#L1093-L1095
        batch_encoded["labels"] = torch.where(
            batch_encoded["attention_mask"] > 0.5, batch_encoded["input_ids"], -100
        )
        tokenized_dataset = _CustomDataset(batch_encoded)
    else:
        # For backward compatibility, if labels are not needed, we only return the input_ids.
        tokenized_dataset = _CustomDataset({"input_ids": batch_encoded["input_ids"]})

    calib_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

    list_tensors = [tensor for batch in calib_dataloader for _, tensor in batch.items()]

    return list_tensors