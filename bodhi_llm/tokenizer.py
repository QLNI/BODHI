#!/usr/bin/env python3
"""Tokenizers: byte-level and sentencepiece subword."""

import os
from typing import List

import sentencepiece as spm


class ByteTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    OFFSET = 4
    VOCAB_SIZE = 260

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        b = text.encode("utf-8", errors="replace")
        ids = [self.BOS] if add_bos else []
        ids.extend(int(x) + self.OFFSET for x in b)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: List[int]) -> str:
        out = bytearray()
        for tid in ids:
            if tid < self.OFFSET:
                continue
            out.append((tid - self.OFFSET) & 0xFF)
        return out.decode("utf-8", errors="replace")


class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.PAD = int(self.sp.pad_id()) if self.sp.pad_id() >= 0 else 0
        self.BOS = int(self.sp.bos_id()) if self.sp.bos_id() >= 0 else 1
        self.EOS = int(self.sp.eos_id()) if self.sp.eos_id() >= 0 else 2
        self.UNK = int(self.sp.unk_id()) if self.sp.unk_id() >= 0 else 3
        self.VOCAB_SIZE = int(self.sp.vocab_size())

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = list(self.sp.encode(text, out_type=int))
        if add_bos:
            ids = [self.BOS] + ids
        if add_eos:
            ids = ids + [self.EOS]
        return ids

    def decode(self, ids: List[int]) -> str:
        filtered = [int(x) for x in ids if int(x) >= 0 and int(x) < self.VOCAB_SIZE]
        return self.sp.decode(filtered)


def train_sentencepiece(text_path: str, model_prefix: str, vocab_size: int = 8000):
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=text_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_id=0,
        input_sentence_size=2000000,
        shuffle_input_sentence=True,
    )

