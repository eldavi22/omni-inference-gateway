from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Encoding, Tokenizer, models, pre_tokenizers, trainers


@dataclass
class FastTokenizer:
    """HuggingFace tokenizers adapter used before every routing decision.

    Set TOKENIZER_JSON to a production tokenizer.json. Without it, a small
    per-input WordLevel fallback keeps this example runnable, but production
    deployments should always supply a trained tokenizer artifact.
    """

    tokenizer_path: str | None

    def __post_init__(self) -> None:
        self._tokenizer: Tokenizer | None = None
        if self.tokenizer_path:
            self._tokenizer = Tokenizer.from_file(self.tokenizer_path)

    def _fallback_tokenizer(self, text: str) -> Tokenizer:
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(
            special_tokens=["[UNK]"],
            min_frequency=1,
        )
        tokenizer.train_from_iterator([text], trainer=trainer)
        return tokenizer

    def encode(self, text: str) -> Encoding:
        tokenizer = self._tokenizer or self._fallback_tokenizer(text)
        return tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError(
                "Decoding requires TOKENIZER_JSON so token IDs have stable vocabulary."
            )
        return self._tokenizer.decode(token_ids)
