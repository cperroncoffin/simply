# Copyright 2024 The Simply Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenizers."""

from collections.abc import Mapping
import functools
import json
from typing import Any, ClassVar, Generic, Protocol, cast

from etils import epath
import sentencepiece as spm
from simply.utils import common
from simply.utils import registry
import tokenizers


class TokenizerRegistry(registry.RootRegistry):
  """Tokenizer registry."""

  namespace: ClassVar[str] = 'tokenizer'


class SimplyVocab(Protocol, Generic[common.RawT]):
  pad_id: int | None
  bos_id: int | None
  eos_id: int | None

  def encode(self, text: common.RawT) -> list[int]:
    ...

  def decode(self, token_ids: list[int]) -> common.RawT:
    ...


class TestVocab(SimplyVocab[str]):
  """Test vocab."""

  def __init__(self, vocab_list, bos_id=2, eos_id=-1, pad_id=0, unk_id=3):
    self.bos_id = bos_id
    self.eos_id = eos_id
    self.pad_id = pad_id
    self.unk_id = unk_id
    start_id = max(unk_id, pad_id, eos_id, bos_id) + 1
    self._vocab_dict = dict(
        [(w, (i + start_id)) for i, w in enumerate(vocab_list)]
    )
    self._rev_vocab_dict = {v: k for k, v in self._vocab_dict.items()}

  def encode(self, text: str) -> list[int]:
    return [self._vocab_dict.get(w, self.unk_id) for w in text.split()]

  def decode(self, token_ids: list[int]) -> str:
    return ' '.join([self._rev_vocab_dict.get(i, '<unk>') for i in token_ids])


class SentencePieceVocabulary(SimplyVocab[str]):
  """SentencePiece vocabulary wrapper.

  This is a drop-in replacement for seqio.SentencePieceVocabulary.
  """

  def __init__(self, vocab_path: str):
    self._vocab_path = vocab_path
    self._sp_model: spm.SentencePieceProcessor | None = None

  @property
  def sp_model(self) -> spm.SentencePieceProcessor:
    if self._sp_model is None:
      self._sp_model = spm.SentencePieceProcessor()
      self._sp_model.Load(self._vocab_path)
    return self._sp_model

  @functools.cached_property
  def vocab_size(self) -> int:
    return self.sp_model.GetPieceSize()

  @functools.cached_property
  def bos_id(self) -> int | None:
    bos_id = self.sp_model.bos_id()
    return bos_id if bos_id >= 0 else None

  @functools.cached_property
  def eos_id(self) -> int | None:
    eos_id = self.sp_model.eos_id()
    return eos_id if eos_id >= 0 else None

  @functools.cached_property
  def pad_id(self) -> int | None:
    pad_id = self.sp_model.pad_id()
    return pad_id if pad_id >= 0 else None

  @functools.cached_property
  def unk_id(self) -> int | None:
    unk_id = self.sp_model.unk_id()
    return unk_id if unk_id >= 0 else None

  def encode(self, text: str) -> list[int]:
    return self.sp_model.EncodeAsIds(text)

  def decode(self, token_ids: list[int]) -> str:
    return self.sp_model.DecodeIds(token_ids)

  def piece_to_id(self, piece: str) -> int:
    """Returns the id of the given piece."""
    return self.sp_model.PieceToId(piece)

  def id_to_piece(self, token_id: int) -> str:
    """Returns the piece of the given id."""
    return self.sp_model.IdToPiece(token_id)


class ByteVocabulary(SimplyVocab[str]):
  """A simple byte-level vocabulary.

  This is a drop-in replacement for seqio.ByteVocabulary, useful for testing.
  Each byte (0-255) maps to a token id, with special tokens added after.
  """

  def __init__(self):
    # Special tokens are placed after the byte range
    self._num_bytes = 256
    self._pad_id = self._num_bytes
    self._eos_id = self._num_bytes + 1
    self._bos_id = self._num_bytes + 2
    self._unk_id = self._num_bytes + 3

  @property
  def vocab_size(self) -> int:
    return self._num_bytes + 4  # 256 bytes + pad, eos, bos, unk

  @property
  def pad_id(self) -> int:
    return self._pad_id

  @property
  def eos_id(self) -> int:
    return self._eos_id

  @property
  def bos_id(self) -> int:
    return self._bos_id

  @property
  def unk_id(self) -> int:
    return self._unk_id

  def encode(self, text: str) -> list[int]:
    """Encode text to byte token ids."""
    return list(text.encode('utf-8'))

  def decode(self, token_ids: list[int]) -> str:
    """Decode byte token ids to text."""
    # Filter out special tokens and invalid byte values
    bytes_list = [
        tid for tid in token_ids
        if 0 <= tid < self._num_bytes
    ]
    return bytes(bytes_list).decode('utf-8', errors='replace')


class HuggingFaceVocab(SimplyVocab[str]):
  """Generic class for HuggingFace vocab."""

  def __init__(self, vocab_path: str):
    self.vocab_path = vocab_path

  @functools.cached_property
  def tokenizer(self) -> tokenizers.Tokenizer:
    vocab_path = epath.Path(self.vocab_path)
    return tokenizers.Tokenizer.from_file(
        (vocab_path / 'tokenizer.json').as_posix()
    )

  @functools.cached_property
  def tokenizer_config(self) -> Mapping[str, Any]:
    vocab_path = epath.Path(self.vocab_path)
    with (vocab_path / 'tokenizer_config.json').open() as f:
      return json.load(f)

  def get_token_id(self, name: str) -> int | None:
    token = self.tokenizer_config[name]
    if token is None:
      return None
    if not isinstance(token, str):
      token = token['content']
    if not isinstance(token, str):
      raise ValueError(f'{token=} is not a string ({name=}).')
    return self.tokenizer.token_to_id(token)

  @functools.cached_property
  def bos_id(self) -> int | None:
    return self.get_token_id('bos_token')

  @functools.cached_property
  def eos_id(self) -> int | None:
    return self.get_token_id('eos_token')

  @functools.cached_property
  def pad_id(self) -> int | None:
    return self.get_token_id('pad_token')

  def encode(self, text: str) -> list[int]:
    encoded = self.tokenizer.encode(text)
    return cast(tokenizers.Encoding, encoded).ids

  def decode(self, token_ids: list[int]) -> str:
    return self.tokenizer.decode(token_ids)
