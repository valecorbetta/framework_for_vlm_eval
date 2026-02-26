import torch


def encode_batch_fulltokenizer(fulltok, texts, max_length: int, device=None):
    # Required tokens
    for t in ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]:
        if t not in fulltok.vocab:
            raise ValueError(f"Tokenizer vocab missing required token: {t}")

    pad_id = fulltok.vocab["[PAD]"]
    cls_id = fulltok.vocab["[CLS]"]
    sep_id = fulltok.vocab["[SEP]"]
    unk_id = fulltok.vocab["[UNK]"]

    all_ids, all_attn, all_tt = [], [], []

    for text in texts:
        tokens = fulltok.tokenize(text)
        ids = [cls_id] + fulltok.convert_tokens_to_ids(tokens) + [sep_id]

        # truncate
        ids = ids[:max_length]

        attn = [1] * len(ids)
        tt = [0] * len(ids)

        # pad
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids = ids + [pad_id] * pad_len
            attn = attn + [0] * pad_len
            tt = tt + [0] * pad_len

        all_ids.append(ids)
        all_attn.append(attn)
        all_tt.append(tt)

    out = {
        "caption_ids": torch.tensor(all_ids, dtype=torch.long),
        "attention_mask": torch.tensor(all_attn, dtype=torch.long),
        "token_type_ids": torch.tensor(all_tt, dtype=torch.long),
    }
    if device is not None:
        out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
    return out
