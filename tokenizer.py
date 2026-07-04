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

import heapq
from multiprocessing import Process, Queue
import regex as re
from collections import Counter, defaultdict

TOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _chunk_sequence(sequence, chunk_count):
    item_count = len(sequence)
    if item_count == 0:
        return []

    chunk_count = max(1, min(chunk_count, item_count))
    chunk_size = (item_count + chunk_count - 1) // chunk_count
    return [sequence[i:i + chunk_size] for i in range(0, item_count, chunk_size)]


def _resolve_worker_count(num_workers, item_count):
    if num_workers is None or item_count <= 1:
        return 1
    return max(1, min(int(num_workers), item_count))


def _count_text_chunk(texts):
    data = Counter()
    for text in texts:
        for match in re.finditer(TOKEN_PATTERN, text):
            elements = match.group().encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in elements)
            data[token_tuple] += 1
    return data


def _count_pair_chunk(items):
    pair_counts = Counter()
    for token_tuple, count in items:
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair] += count
    return pair_counts


def _merge_pair_tokens(token_tuple, pair_to_merge, new_token, count):
    deltas = defaultdict(int)
    out = []
    i = 0
    while i < len(token_tuple):
        if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == pair_to_merge:
            left = out[-1] if out else None
            right = token_tuple[i + 2] if i + 2 < len(token_tuple) else None

            deltas[(token_tuple[i], token_tuple[i + 1])] -= count
            if left is not None:
                deltas[(left, token_tuple[i])] -= count
                deltas[(left, new_token)] += count
            if right is not None:
                deltas[(token_tuple[i + 1], right)] -= count
                deltas[(new_token, right)] += count

            out.append(new_token)
            i += 2
        else:
            out.append(token_tuple[i])
            i += 1
    return tuple(out), deltas


def _tokenizer_train_worker(worker_id, texts, task_queue, result_queue):
    data = _count_text_chunk(texts)
    pair_counts = _count_pair_chunk(data.items())
    result_queue.put(("ready", worker_id, pair_counts, len(data)))

    while True:
        task = task_queue.get()
        if task is None:
            break

        best_pair, new_token = task
        new_data = Counter()
        total_deltas = Counter()

        for token_tuple, count in data.items():
            new_token_tuple, deltas = _merge_pair_tokens(
                token_tuple, best_pair, new_token, count
            )
            new_data[new_token_tuple] += count
            total_deltas.update(deltas)

        data = new_data
        result_queue.put(("merged", worker_id, total_deltas, len(data)))


class Tokenizer:

    # OpenAI's GPT tokenizer regex
    # break text into chunks
    PAT = TOKEN_PATTERN

    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab                                      # dict[int, bytes]
        self.merges = merges                                    # list[tuple[bytes, bytes]]
        self.special_tokens = special_tokens                    # list[str]
        self.merge_priority = {pair: i for i, pair in enumerate(self.merges)}
        self.bytes_to_id = {v: k for k, v in vocab.items()}

    @classmethod
    def train(cls, texts, vocab_size, special_tokens, num_workers=1, progress_interval=1000):
        """
        Train a BPE tokenizer on the given texts.

        Args:
            list[str] texts: List of strings to train on.
            int vocab_size: Target vocabulary size.
            list[str] special_tokens: List of special token strings.
            int num_workers: Number of worker processes for counting/update work.

        Returns:
            A trained Tokenizer instance.
        """
        text_count = len(texts)
        worker_count = _resolve_worker_count(num_workers, text_count)

        if worker_count > 1:
            print(f"Training tokenizer with {worker_count} workers ...", flush=True)
            return cls._train_parallel(
                texts,
                worker_count,
                vocab_size,
                special_tokens,
                progress_interval,
            )

        data = _count_text_chunk(texts)
        pair_counts = _count_pair_chunk(list(data.items()))

        return cls._train_from_counts(
            data,
            pair_counts,
            vocab_size,
            special_tokens,
            worker_count=1,
            progress_interval=progress_interval,
        )

    @classmethod
    def _train_parallel(
        cls,
        texts,
        worker_count,
        vocab_size,
        special_tokens,
        progress_interval,
    ):
        text_chunks = _chunk_sequence(texts, worker_count)
        worker_count = len(text_chunks)
        task_queues = [Queue() for _ in range(worker_count)]
        result_queue = Queue()
        processes = []

        for worker_id, chunk in enumerate(text_chunks):
            process = Process(
                target=_tokenizer_train_worker,
                args=(worker_id, chunk, task_queues[worker_id], result_queue),
            )
            process.start()
            processes.append(process)

        del text_chunks

        try:
            pair_counts = Counter()
            active_token_count = 0
            for _ in processes:
                message, _worker_id, chunk_counts, data_len = result_queue.get()
                if message != "ready":
                    raise RuntimeError(f"Unexpected tokenizer worker message: {message}")
                pair_counts.update(chunk_counts)
                active_token_count += data_len

            return cls._train_from_counts(
                data=None,
                pair_counts=pair_counts,
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                task_queues=task_queues,
                result_queue=result_queue,
                worker_count=worker_count,
                active_token_count=active_token_count,
                progress_interval=progress_interval,
            )
        finally:
            for task_queue in task_queues:
                task_queue.put(None)

            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join()

    @classmethod
    def _train_from_counts(
        cls,
        data,
        pair_counts,
        vocab_size,
        special_tokens,
        task_queues=None,
        result_queue=None,
        worker_count=1,
        active_token_count=None,
        progress_interval=1000,
    ):
        """Finish BPE training after text and initial pair counts are built."""

        vocab_elems = []
        for token_str in special_tokens:
            vocab_elems.append(token_str.encode("utf-8"))
        vocab_elems += [bytes([i]) for i in range(256)]

        merges = []
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        while len(vocab_elems) < vocab_size:
            #1, find the one with max number and break tie with lexicographical greater
            while heap:
                neg_count, best_pair = heapq.heappop(heap)
                if best_pair in pair_counts and pair_counts[best_pair] == -neg_count:
                    break
            else:
                break
            merges.append(best_pair)

            #2, append it at vocab_elements
            new_token = best_pair[0] + best_pair[1]
            vocab_elems.append(new_token)

            #3, update the keys in data
            if task_queues is not None:
                total_deltas = Counter()
                active_token_count = 0

                for task_queue in task_queues:
                    task_queue.put((best_pair, new_token))

                for _ in range(worker_count):
                    message, _worker_id, chunk_deltas, data_len = result_queue.get()
                    if message != "merged":
                        raise RuntimeError(f"Unexpected tokenizer worker message: {message}")
                    total_deltas.update(chunk_deltas)
                    active_token_count += data_len
            else:
                new_data = Counter()
                total_deltas = Counter()

                for token_tuple, count in data.items():
                    # Create a new token tuple by merging the best_pair
                    # (b'h', b'e', b'l', b'l', b'o') -> (b'h', b'e', b'll', b'o')
                    new_token_tuple, deltas = cls.merge_pair(
                        token_tuple, best_pair, new_token, count
                    )
                    new_data[new_token_tuple] += count
                    total_deltas.update(deltas)

            #4, update pair_counts with deltas
            for p, d in total_deltas.items():
                pair_counts[p] = pair_counts.get(p, 0) + d
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                else:
                    heapq.heappush(heap, (-pair_counts[p], p))

            # Replace old data with the newly merged data for the next loop
            if task_queues is None:
                data = new_data

            if progress_interval and len(vocab_elems) % progress_interval == 0:
                active_tokens = active_token_count if task_queues is not None else len(data)
                print(
                    f"  vocab {len(vocab_elems):,}/{vocab_size:,} | "
                    f"active tokens {active_tokens:,} | pairs {len(pair_counts):,}",
                    flush=True,
                )

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

        return _merge_pair_tokens(token_tuple, pair_to_merge, new_token, count)

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
