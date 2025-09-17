import torch
from torch.utils.data import Dataset
import random


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, anchor="forget"):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            anchor (str, optional): Specifies which dataset to anchor while randomly sampling from the other dataset. Defaults to 'forget'.
        """
        self.forget = forget
        self.retain = retain
        self.anchor = anchor

    def __len__(self):
        """Ensures the sampled dataset matches the anchor dataset's length."""
        if self.anchor == "forget":
            assert self.forget is not None, ValueError(
                "forget dataset can't be None when anchor=forget"
            )
            if isinstance(self.forget, dict):
                return sum(len(dataset) for dataset in self.forget.values())
            else:
                return len(self.forget)
        elif self.anchor == "retain":
            assert self.retain is not None, ValueError(
                "retain dataset can't be None when anchor=retain"
            )
            if isinstance(self.retain, dict):
                return sum(len(dataset) for dataset in self.retain.values())
            else:
                return len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} can be only forget or retain")

    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            # Handle forget dataset (could be dict or single dataset)
            if isinstance(self.forget, dict):
                forget_keys = list(self.forget.keys())
                random_key = random.choice(forget_keys)
                item["forget"] = self.forget[random_key][random.randint(0, len(self.forget[random_key]) - 1)]
            else:
                forget_idx = random.randint(0, len(self.forget) - 1)
                item["forget"] = self.forget[forget_idx]
            
            if self.retain:
                # Handle retain dataset (could be dict or single dataset)
                if isinstance(self.retain, dict):
                    retain_keys = list(self.retain.keys())
                    random_key = random.choice(retain_keys)
                    item["retain"] = self.retain[random_key][random.randint(0, len(self.retain[random_key]) - 1)]
                else:
                    retain_idx = random.randint(0, len(self.retain) - 1)
                    item["retain"] = self.retain[retain_idx]
                    
        elif self.anchor == "retain":
            # Handle retain dataset (could be dict or single dataset)
            if isinstance(self.retain, dict):
                retain_keys = list(self.retain.keys())
                key = retain_keys[idx % len(retain_keys)]
                dataset_idx = idx // len(retain_keys)
                item["retain"] = self.retain[key][dataset_idx]
            else:
                item["retain"] = self.retain[idx]
                
            if self.forget:
                # Handle forget dataset (could be dict or single dataset)
                if isinstance(self.forget, dict):
                    forget_keys = list(self.forget.keys())
                    random_key = random.choice(forget_keys)
                    item["forget"] = self.forget[random_key][random.randint(0, len(self.forget[random_key]) - 1)]
                else:
                    forget_idx = random.randint(0, len(self.forget) - 1)
                    item["forget"] = self.forget[forget_idx]
        return item
