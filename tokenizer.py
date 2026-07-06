"""Hugging Face byte-level BPE tokenizer wrapper.

The project uses the public signatures from ``tokenizers.Tokenizer`` for the
regular tokenizer operations, plus a few project helpers for ID-only encoding
and default byte-level BPE training.
"""

import json

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers


def _new_byte_level_bpe_tokenizer():
    tokenizer = HFTokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def _byte_to_unicode():
    byte_values = (
        list(range(33, 127))
        + list(range(161, 173))
        + list(range(174, 256))
    )
    codepoints = byte_values[:]
    next_codepoint = 0

    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            codepoints.append(256 + next_codepoint)
            next_codepoint += 1

    return dict(zip(byte_values, (chr(codepoint) for codepoint in codepoints)))


_BYTE_ENCODER = _byte_to_unicode()


def _legacy_token_to_hf(raw_token, special_tokens):
    for special_token in special_tokens:
        if raw_token == special_token.encode("utf-8"):
            return special_token

    return "".join(_BYTE_ENCODER[byte_value] for byte_value in raw_token)


def _load_legacy_tokenizer(path, original_error):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        raise original_error

    if not {"vocab", "merges", "special_tokens"}.issubset(data):
        raise original_error

    special_tokens = data["special_tokens"]
    vocab = {
        _legacy_token_to_hf(bytes.fromhex(hex_token), special_tokens): int(token_id)
        for token_id, hex_token in data["vocab"].items()
    }
    merges = [
        (
            _legacy_token_to_hf(bytes.fromhex(left), special_tokens),
            _legacy_token_to_hf(bytes.fromhex(right), special_tokens),
        )
        for left, right in data["merges"]
    ]

    tokenizer = HFTokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


class Tokenizer:
    """Small project wrapper around ``tokenizers.Tokenizer``."""

    def __init__(self, model=None):
        self._loaded_from_legacy = False

        if isinstance(model, HFTokenizer):
            self._tokenizer = model
        elif model is None:
            self._tokenizer = _new_byte_level_bpe_tokenizer()
        else:
            self._tokenizer = HFTokenizer(model)
            self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False,
                use_regex=True,
            )
            self._tokenizer.decoder = decoders.ByteLevel()

    @staticmethod
    def from_file(path):
        try:
            return Tokenizer(HFTokenizer.from_file(path))
        except Exception as exc:
            tokenizer = Tokenizer(_load_legacy_tokenizer(path, exc))
            tokenizer._loaded_from_legacy = True
            return tokenizer

    @staticmethod
    def from_str(json):
        return Tokenizer(HFTokenizer.from_str(json))

    @staticmethod
    def from_buffer(buffer):
        return Tokenizer(HFTokenizer.from_buffer(buffer))

    @staticmethod
    def from_pretrained(identifier, revision="main", token=None):
        return Tokenizer(
            HFTokenizer.from_pretrained(
                identifier,
                revision=revision,
                token=token,
            )
        )

    @staticmethod
    def build_bpe_trainer(vocab_size=30000, special_tokens=None, show_progress=True):
        return trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=0,
            show_progress=show_progress,
            special_tokens=special_tokens or [],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

    @classmethod
    def load(cls, path):
        return cls.from_file(path)

    @property
    def hf_tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    @property
    def loaded_from_legacy(self):
        return self._loaded_from_legacy

    @property
    def model(self):
        return self._tokenizer.model

    @model.setter
    def model(self, value):
        self._tokenizer.model = value

    @property
    def normalizer(self):
        return self._tokenizer.normalizer

    @normalizer.setter
    def normalizer(self, value):
        self._tokenizer.normalizer = value

    @property
    def pre_tokenizer(self):
        return self._tokenizer.pre_tokenizer

    @pre_tokenizer.setter
    def pre_tokenizer(self, value):
        self._tokenizer.pre_tokenizer = value

    @property
    def post_processor(self):
        return self._tokenizer.post_processor

    @post_processor.setter
    def post_processor(self, value):
        self._tokenizer.post_processor = value

    @property
    def decoder(self):
        return self._tokenizer.decoder

    @decoder.setter
    def decoder(self, value):
        self._tokenizer.decoder = value

    @property
    def padding(self):
        return self._tokenizer.padding

    @property
    def truncation(self):
        return self._tokenizer.truncation

    @property
    def encode_special_tokens(self):
        return self._tokenizer.encode_special_tokens

    @encode_special_tokens.setter
    def encode_special_tokens(self, value):
        self._tokenizer.encode_special_tokens = value

    def train(self, files, trainer=None):
        return self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(self, iterator, trainer=None, length=None):
        return self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        )

    def encode(self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True):
        return self._tokenizer.encode(
            sequence,
            pair=pair,
            is_pretokenized=is_pretokenized,
            add_special_tokens=add_special_tokens,
        )

    def encode_batch(self, input, is_pretokenized=False, add_special_tokens=True):
        return self._tokenizer.encode_batch(
            input,
            is_pretokenized=is_pretokenized,
            add_special_tokens=add_special_tokens,
        )

    def encode_batch_fast(self, input, is_pretokenized=False, add_special_tokens=True):
        return self._tokenizer.encode_batch_fast(
            input,
            is_pretokenized=is_pretokenized,
            add_special_tokens=add_special_tokens,
        )

    def decode(self, ids, skip_special_tokens=True):
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, sequences, skip_special_tokens=True):
        return self._tokenizer.decode_batch(
            sequences,
            skip_special_tokens=skip_special_tokens,
        )

    def post_process(self, encoding, pair=None, add_special_tokens=True):
        return self._tokenizer.post_process(
            encoding,
            pair=pair,
            add_special_tokens=add_special_tokens,
        )

    def get_vocab(self, with_added_tokens=True):
        return self._tokenizer.get_vocab(with_added_tokens=with_added_tokens)

    def get_vocab_size(self, with_added_tokens=True):
        return self._tokenizer.get_vocab_size(with_added_tokens=with_added_tokens)

    def token_to_id(self, token):
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id):
        return self._tokenizer.id_to_token(id)

    def add_special_tokens(self, tokens):
        return self._tokenizer.add_special_tokens(tokens)

    def add_tokens(self, tokens):
        return self._tokenizer.add_tokens(tokens)

    def get_added_tokens_decoder(self):
        return self._tokenizer.get_added_tokens_decoder()

    def enable_padding(
        self,
        direction="right",
        pad_id=0,
        pad_type_id=0,
        pad_token="[PAD]",
        length=None,
        pad_to_multiple_of=None,
    ):
        return self._tokenizer.enable_padding(
            direction=direction,
            pad_id=pad_id,
            pad_type_id=pad_type_id,
            pad_token=pad_token,
            length=length,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def no_padding(self):
        return self._tokenizer.no_padding()

    def enable_truncation(
        self,
        max_length,
        stride=0,
        strategy="longest_first",
        direction="right",
    ):
        return self._tokenizer.enable_truncation(
            max_length,
            stride=stride,
            strategy=strategy,
            direction=direction,
        )

    def no_truncation(self):
        return self._tokenizer.no_truncation()

    def num_special_tokens_to_add(self, is_pair):
        return self._tokenizer.num_special_tokens_to_add(is_pair)

    def save(self, path, pretty=True):
        return self._tokenizer.save(path, pretty=pretty)

    def to_str(self, pretty=False):
        return self._tokenizer.to_str(pretty=pretty)

    def encode_ids(
        self,
        sequence,
        pair=None,
        is_pretokenized=False,
        add_special_tokens=True,
    ):
        return self.encode(
            sequence,
            pair=pair,
            is_pretokenized=is_pretokenized,
            add_special_tokens=add_special_tokens,
        ).ids

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)
