from typing import Any
import torch
import albumentations as A
import einops
from albumentations.core.transforms_interface import ImageOnlyTransform
from transformers import PreTrainedTokenizerBase


class RepeatChannelsEinopsd(ImageOnlyTransform):
    def __init__(self, always_apply: bool = True, p: float = 1.0):
        # Initialize parent ImageOnlyTransform class
        super(RepeatChannelsEinopsd, self).__init__(always_apply, p)

    def apply(self, image: torch.Tensor, **kwargs):
        # This method is called to apply the transform
        image = einops.repeat(image, "h w -> h w c", c=3)
        return image

    def get_transform_init_args_names(self):
        # This method returns the names of the init args that need to be saved
        # in the serialized representation of the transform (if any).
        # Since this transform does not have any initialization arguments that
        # need to be saved, we return an empty tuple.
        return ()


def prepare_transforms_embed(
    seed: int,
    p: float = 1.0,
    alpha: int = 10,
    sigma: int = 15,
    mean: float = 0.3089279,
    std: float = 0.25053555408335154,
):
    augmentations = A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Affine(
                rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20
            ),  # should probably consider removing this and the elastic transform since I don't want to deform the concepts
            A.ElasticTransform(alpha=alpha, sigma=sigma),
        ],
        p=p,
        seed=seed,
    )

    preprocessing = A.Compose(
        [
            A.Normalize(mean=(mean), std=(std)),
            RepeatChannelsEinopsd(),
            A.pytorch.ToTensorV2(),
        ],
        p=1.0,  # alwyas apply preprocessing
        seed=seed,  # normalize with the mean and std provided by MammoCLIP (I assume these values are obtained on their private dataset), then to tensor -> torch.float32 [1,H,W]
    )

    return augmentations, preprocessing


def mica_collate_hf_tokenizer(
    batch: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase, max_length: int
):
    pad_id = tokenizer.vocab.get("[PAD]", 0)
    cls_id = tokenizer.vocab["[CLS]"]
    sep_id = tokenizer.vocab["[SEP]"]
    unk_id = tokenizer.vocab.get("[UNK]", 100)

    # images
    imgs = torch.stack([b["x"] for b in batch])  # [B, C, H, W]

    input_ids = []
    attention_mask = []
    token_type_ids = []

    for b in batch:
        text = b["caption"]
        toks = tokenizer.tokenize(text)
        # map unknown tokens safely
        toks = [t if t in tokenizer.vocab else "[UNK]" for t in toks]
        ids = tokenizer.convert_tokens_to_ids(toks)

        # truncate for special tokens
        ids = ids[: max_length - 2]
        ids = [cls_id] + ids + [sep_id]

        mask = [1] * len(ids)
        types = [0] * len(ids)

        # pad
        pad_n = max_length - len(ids)
        if pad_n > 0:
            ids += [pad_id] * pad_n
            mask += [0] * pad_n
            types += [0] * pad_n
        else:
            ids = ids[:max_length]
            mask = mask[:max_length]
            types = types[:max_length]

        input_ids.append(ids)
        attention_mask.append(mask)
        token_type_ids.append(types)

    concept_labels = torch.stack([b["concept_labels"] for b in batch]).float()

    return {
        "x": imgs,
        "caption_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "concept_labels": concept_labels,
        "filename": [b["filename"] for b in batch],
        "y": torch.stack([b["y"] for b in batch]),
    }
