"""
A BPE Tokenizer implementation.

Components:
    PAT: Regex pattern for splitting text into chunks.
    train: Class method to train a tokenizer on texts.
    merge_pair: Static method to merge byte pairs during training.
    encode: Convert text to a list of token IDs.
    decode: Convert token IDs back to text.
    save/load: Persist and restore a trained tokenizer.

Usage:
    tokenizer = Tokenizer.train(texts, vocab_size, special_tokens)
    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)
    tokenizer.save(path)
    tokenizer = Tokenizer.load(path)
"""

import regex as re
from collections import Counter, defaultdict

class Tokenizer:

    # OpenAI's GPT tokenizer regex
    # break text into chunks
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab                                      # dict[int, bytes]
        self.merges = merges                                    # list[tuple[bytes, bytes]]
        self.special_tokens = special_tokens                    # list[str]
        self.merge_priority = {pair: i for i, pair in enumerate(self.merges)}
        self.bytes_to_id = {v: k for k, v in vocab.items()}

    @classmethod
    def train(cls, texts, vocab_size, special_tokens):
        """
        Train a BPE tokenizer on the given texts.

        Args:
            list[str] texts: List of strings to train on.
            int vocab_size: Target vocabulary size.
            list[str] special_tokens: List of special token strings.

        Returns:
            A trained Tokenizer instance.
        """
        data = Counter()
        for text in texts:
            for match in re.finditer(cls.PAT, text):
                elements = match.group().encode('utf-8')
                token_tuple = tuple(bytes([b]) for b in elements)
                data[token_tuple] += 1

        vocab_elems = []
        for token_str in special_tokens:
            vocab_elems.append(token_str.encode("utf-8"))
        vocab_elems += [bytes([i]) for i in range(256)]

        merges = []
        pair_counts = defaultdict(int)
        for token_tuple, count in data.items():
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i+1])
                pair_counts[pair] += count

        while len(vocab_elems) < vocab_size:
            #1, find the one with max number and break tie with lexicographical greater
            best_pair, max_count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
            merges.append(best_pair)

            #2, append it at vocab_elements
            new_token = best_pair[0] + best_pair[1]
            vocab_elems.append(new_token)

            #3, update the keys in data
            new_data = Counter()
            total_deltas = defaultdict(int)

            for token_tuple, count in data.items():
                # Create a new token tuple by merging the best_pair
                # (b'h', b'e', b'l', b'l', b'o') -> (b'h', b'e', b'll', b'o')
                new_token_tuple, deltas = cls.merge_pair(
                    token_tuple, best_pair, new_token, count
                )
                new_data[new_token_tuple] += count
                for p, d in deltas.items():
                    total_deltas[p] += d

            #4, update pair_counts with deltas
            for p, d in total_deltas.items():
                pair_counts[p] = pair_counts.get(p, 0) + d
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)

            # Replace old data with the newly merged data for the next loop
            data = new_data

        vocab = {i: token for i, token in enumerate(vocab_elems)}

        return cls(vocab, merges, special_tokens)
    
    @staticmethod
    def merge_pair(
        token_tuple: tuple[bytes, ...],
        pair_to_merge: tuple[bytes, bytes],
        new_token: bytes,
        count: int,
    ) -> tuple[tuple[bytes, ...], dict[tuple[bytes, bytes], int]]:
        """
        Merge all occurrences of a pair in a token tuple.

        Args:
            token_tuple: Tuple of byte objects.
            pair_to_merge: The pair of bytes to find and merge.
            new_token: The merged bytes object.
            count: Frequency count for delta tracking.

        Returns:
            (new_tuple, deltas): Updated tuple and pair frequency changes.

        Example:
            merge_pair((b'h', b'e', b'l', b'l', b'o'), (b'l', b'l'), b'll', 1)
            Returns: ((b'h', b'e', b'll', b'o'), {...deltas...})
        """

        deltas = defaultdict(int)
        out = []
        i = 0
        while i < len(token_tuple):
            # Check if the pair (b'l', b'l') exists at the current position
            if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i+1]) == pair_to_merge:
                left = out[-1] if out else None
                right = token_tuple[i+2] if i + 2 < len(token_tuple) else None

                deltas[(token_tuple[i], token_tuple[i+1])] -= count
                if left is not None:
                    deltas[(left, token_tuple[i])] -= count
                    deltas[(left, new_token)] += count
                if right is not None:
                    deltas[(token_tuple[i+1], right)] -= count
                    deltas[(new_token, right)] += count

                out.append(new_token)
                i += 2 # Skip both elements of the pair
            else:
                # No match, just append the current element
                out.append(token_tuple[i])
                i += 1
        return tuple(out), deltas

    def encode(self, text):
        ids = []

        # 1. Split on special tokens, keeping them in the result
        special_pattern = '(' + '|'.join(re.escape(s) for s in self.special_tokens) + ')'
        parts = re.split(special_pattern, text) if self.special_tokens else [text]

        for part in parts:
            if not part:
                continue

            # 2. If it's a special token, look up directly
            if part in self.special_tokens:
                ids.append(self.bytes_to_id[part.encode("utf-8")])
                continue

            # 3. Regex split into pre-tokens, then apply merges
            for match in re.finditer(self.PAT, part):
                token_tuple = tuple(bytes([b]) for b in match.group().encode("utf-8"))

                # Repeatedly merge the highest-priority pair
                while len(token_tuple) > 1:
                    best_pair = None
                    best_idx = float('inf')
                    for i in range(len(token_tuple) - 1):
                        pair = (token_tuple[i], token_tuple[i + 1])
                        if pair in self.merge_priority and self.merge_priority[pair] < best_idx:
                            best_idx = self.merge_priority[pair]
                            best_pair = pair

                    if best_pair is None:
                        break

                    new_token = best_pair[0] + best_pair[1]
                    token_tuple, _ = self.merge_pair(token_tuple, best_pair, new_token, 1)

                # Map each resulting bytes token to its ID
                for token in token_tuple:
                    ids.append(self.bytes_to_id[token])

        return ids
    
    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
    
    @property
    def vocab_size(self):
        return len(self.vocab)

    def save(self, path):
        import json
        data = {
            "vocab": {i: token.hex() for i, token in self.vocab.items()},
            "merges": [(a.hex(), b.hex()) for a, b in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        import json
        with open(path, "r") as f:
            data = json.load(f)
        vocab = {int(i): bytes.fromhex(h) for i, h in data["vocab"].items()}
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data["merges"]]
        special_tokens = data["special_tokens"]
        return cls(vocab, merges, special_tokens)