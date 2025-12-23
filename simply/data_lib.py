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
r"""Utilities for dataset creation.
"""

import dataclasses
import functools
import glob as glob_module
import json
import os
from typing import Any, Callable, ClassVar, Iterator, Mapping, MutableMapping, Protocol, Union

import datasets as hf_datasets
import einops
from etils import epath
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
from simply.utils import common
from simply.utils import registry
from simply.utils import tokenization

################################################################################
# Type aliases.
Batch = MutableMapping[str, Union[np.ndarray, jnp.ndarray]]
Processor = Callable[[Batch], Batch]

DATASETS_DIR = os.getenv('SIMPLY_DATASETS', os.path.expanduser('~/.cache/simply/datasets/'))
VOCABS_DIR = os.getenv('SIMPLY_VOCABS', os.path.expanduser('~/.cache/simply/vocabs/'))

################################################################################
# Tokenizers / vocabularies.

OPENMIX_V1_32768_VOCAB = os.path.join(VOCABS_DIR, 'spm-32768-open_mix_v2_edu-r100-v1p1-07122024.model')
OPENMIX_V1_100864_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-open_mix_v1-reserved_100-02272024.model')
FWEDU_100864_V1_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-fwedu-r100-v1-07102024.model')
OPENMIX_V2_EDU_100864_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-open_mix_v2_edu-r100-v1-07122024.model')
OPENMIX_V2_EDU_100864_V1P1_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-open_mix_v2_edu-r100-v1p1-07122024.model')
OPENMIX_V3_100864_V1_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-openmix_v3-r100-v1-08312024.model')
OPENMIX_V3_100864_V2_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-openmix_v3-r100-v2-08312024.model')
GEMMA2_VOCAB = os.path.join(VOCABS_DIR, 'gemma2_tokenizer.model')
GEMMA3_VOCAB = os.path.join(VOCABS_DIR, 'gemma3_cleaned_262144_v2.spiece.model')
QWEN3_VOCAB = os.path.join(VOCABS_DIR, 'Qwen3')

OPENMIX_V1_VOCABS = [
    ('vb100864_openmix_v1', OPENMIX_V1_100864_VOCAB),
    ('vb32768_openmix_v1', OPENMIX_V1_32768_VOCAB)]
OPENMIX_V2_VOCABS = [
    ('vb100864_v1p1_openmix_v2_edu', OPENMIX_V2_EDU_100864_V1P1_VOCAB)]
OPENMIX_V3_VOCABS = [
    ('vb100864_v2_openmix_v3', OPENMIX_V3_100864_V2_VOCAB)]
GEMMA2_VOCABS = [('vb256128_gemma2', GEMMA2_VOCAB)]
T5_CC_VOCABS = [
    ('vb32000_t5_cc',
     'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model')]


def register_vocabs():
  vocabs = (
      OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS +
      OPENMIX_V3_VOCABS + GEMMA2_VOCABS)
  for name, vocab_path in vocabs:
    tokenization.TokenizerRegistry.register_value(
        tokenization.SentencePieceVocabulary(vocab_path), name=name)

register_vocabs()

tokenization.TokenizerRegistry.register_value(
    tokenization.SentencePieceVocabulary(GEMMA3_VOCAB), name='vb262144_gemma3'
)

tokenization.TokenizerRegistry.register_value(
    tokenization.HuggingFaceVocab(QWEN3_VOCAB), name='Qwen3'
)

PILE_50432_V1_VOCAB = os.path.join(VOCABS_DIR, 'spm-50432-pile-train00-02122024.model')
PILE_50432_V2_VOCAB = os.path.join(VOCABS_DIR, 'spm-50432-pile-train00+01-02122024.model')
PILE_50432_V3_VOCAB = os.path.join(VOCABS_DIR, 'spm-50432-pile-train00-spc2_24-02252024.model')
PILE_100864_V1_VOCAB = os.path.join(VOCABS_DIR, 'spm-100864-pile-train00+01-02142024.model')
PILE_256000_V1_VOCAB = os.path.join(VOCABS_DIR, 'spm-256000-pile-train00+01-02162024.model')

PILE_VOCABS = [
    ('vb50432_v3_pile', PILE_50432_V3_VOCAB),
    ('vb100864_v1_pile', PILE_100864_V1_VOCAB),
    ('vb256000_v1_pile', PILE_256000_V1_VOCAB),
]


USER_TOKEN = '<reserved_1>'
ASSISTANT_TOKEN = '<reserved_2>'
SYSTEM_TOKEN = '<reserved_3>'
END_OF_MESSAGE_TOKEN = '<reserved_4>'


################################################################################
# Grain-based Dataset Configurations and Transforms.


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
  """Configuration for a dataset."""
  vocab_name: str
  vocab_path: str
  seq_len: int = 1024
  add_eos: bool = False
  add_bos: bool = False
  use_packing: bool = False


class TokenizeTransform(grain.MapTransform):
  """Tokenizes text using a SentencePiece tokenizer."""

  def __init__(
      self, vocab: tokenization.SentencePieceVocabulary,
      text_key: str = 'text',
      add_eos: bool = False,
      add_bos: bool = False):
    self.vocab = vocab
    self.text_key = text_key
    self.add_eos = add_eos
    self.add_bos = add_bos

  def map(self, features: dict[str, Any]) -> dict[str, Any]:
    text = features[self.text_key]
    if isinstance(text, bytes):
      text = text.decode('utf-8')
    tokens = self.vocab.encode(text)
    if self.add_bos and self.vocab.bos_id is not None:
      tokens = [self.vocab.bos_id] + tokens
    if self.add_eos and self.vocab.eos_id is not None:
      tokens = tokens + [self.vocab.eos_id]
    return {'targets': np.array(tokens, dtype=np.int32)}


class LMFeatureConverter(grain.MapTransform):
  """Converts tokenized examples to LM format with inputs/targets shift."""

  def __init__(self, seq_len: int, bos_id: int = 0):
    self.seq_len = seq_len
    self.bos_id = bos_id

  def map(self, features: dict[str, Any]) -> dict[str, Any]:
    targets = features['targets']
    # Pad or truncate to seq_len + 1 (to create input/target pairs)
    if len(targets) > self.seq_len:
      targets = targets[:self.seq_len]
    elif len(targets) < self.seq_len:
      padding = np.zeros(self.seq_len - len(targets), dtype=np.int32)
      targets = np.concatenate([targets, padding])

    # Create decoder_input_tokens with BOS prepended and decoder_target_tokens
    decoder_input_tokens = np.concatenate([[self.bos_id], targets[:-1]])
    decoder_target_tokens = targets
    decoder_loss_weights = (decoder_target_tokens != 0).astype(np.float32)

    return {
        'decoder_input_tokens': decoder_input_tokens.astype(np.int32),
        'decoder_target_tokens': decoder_target_tokens.astype(np.int32),
        'decoder_loss_weights': decoder_loss_weights,
    }


################################################################################
# Grain-based Data Sources.


class HuggingFaceDataSource:
  """Grain-compatible data source for HuggingFace Datasets."""

  def __init__(
      self,
      dataset_name: str,
      split: str = 'train',
      text_key: str = 'text',
      config_name: str | None = None,
      cache_dir: str | None = None,
      trust_remote_code: bool = False,
  ):
    self.dataset_name = dataset_name
    self.split = split
    self.text_key = text_key
    self.config_name = config_name
    self.cache_dir = cache_dir
    self.trust_remote_code = trust_remote_code
    self._dataset = None

  def _load_dataset(self):
    if self._dataset is None:
      self._dataset = hf_datasets.load_dataset(
          self.dataset_name,
          name=self.config_name,
          split=self.split,
          cache_dir=self.cache_dir,
          trust_remote_code=self.trust_remote_code,
      )

  def __len__(self) -> int:
    self._load_dataset()
    return len(self._dataset)

  def __getitem__(self, index: int) -> dict[str, Any]:
    self._load_dataset()
    example = self._dataset[index]
    text = example[self.text_key]
    if isinstance(text, bytes):
      text = text.decode('utf-8')
    return {self.text_key: text}


class ParquetDataSource:
  """Grain-compatible data source for Parquet files."""

  def __init__(
      self,
      file_pattern: str,
      text_key: str = 'text',
      text_preprocessor: Callable[[str], str] | None = None,
  ):
    self.file_pattern = file_pattern
    self.text_key = text_key
    self.text_preprocessor = text_preprocessor
    self._dataset = None
    self._files = None

  def _load_dataset(self):
    if self._dataset is None:
      self._files = sorted(glob_module.glob(self.file_pattern))
      if not self._files:
        raise ValueError(f'No files found matching pattern: {self.file_pattern}')
      # Load all parquet files into a single HF dataset
      self._dataset = hf_datasets.load_dataset(
          'parquet',
          data_files=self._files,
          split='train',
      )

  def __len__(self) -> int:
    self._load_dataset()
    return len(self._dataset)

  def __getitem__(self, index: int) -> dict[str, Any]:
    self._load_dataset()
    example = self._dataset[index]
    text = example[self.text_key]
    if isinstance(text, bytes):
      text = text.decode('utf-8')
    if self.text_preprocessor is not None:
      text = self.text_preprocessor(text)
    return {'text': text}


class JSONLDataSource:
  """Grain-compatible data source for JSONL files."""

  def __init__(
      self,
      file_pattern: str,
      text_key: str = 'text',
      text_preprocessor: Callable[[str], str] | None = None,
  ):
    self.file_pattern = file_pattern
    self.text_key = text_key
    self.text_preprocessor = text_preprocessor
    self._dataset = None
    self._files = None

  def _load_dataset(self):
    if self._dataset is None:
      self._files = sorted(glob_module.glob(self.file_pattern))
      if not self._files:
        raise ValueError(f'No files found matching pattern: {self.file_pattern}')
      # Load all JSONL files into a single HF dataset
      self._dataset = hf_datasets.load_dataset(
          'json',
          data_files=self._files,
          split='train',
      )

  def __len__(self) -> int:
    self._load_dataset()
    return len(self._dataset)

  def __getitem__(self, index: int) -> dict[str, Any]:
    self._load_dataset()
    example = self._dataset[index]
    text = example[self.text_key]
    if isinstance(text, bytes):
      text = text.decode('utf-8')
    if self.text_preprocessor is not None:
      text = self.text_preprocessor(text)
    return {'text': text}


################################################################################
# Dataset Registry for PT and SFT Datasets.


# Register HuggingFace datasets
# Format: (name, hf_dataset_name, hf_config, splits, vocabs)
_HF_DATASETS = [
    ('lm1b', 'lm1b', None, {
        'train': 'train[:90%]',
        'validation': 'train[90%:]',
        'test': 'test'
    }, OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS + [('vb32768_openmix_v1', OPENMIX_V1_32768_VOCAB)]),
    ('minilm1b', 'lm1b', None, {
        'train': 'train[:500]',
        'validation': 'train[500:1000]',
        'test': 'test'
    }, OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS + [('vb32768_openmix_v1', OPENMIX_V1_32768_VOCAB)]),
    ('c4', 'allenai/c4', 'en', {
        'train': 'train',
        'validation': 'validation',
    }, OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS),
    ('imdb_reviews', 'imdb', 'plain_text', {
        'train': 'train[:90%]',
        'validation': 'train[90%:]',
        'test': 'test'
    }, T5_CC_VOCABS),
]


def process_conversation(serialized_conversation: str) -> str:
  """Process a serialized conversation into a single text string."""
  conversation = json.loads(serialized_conversation)
  text = []
  role_token_dict = {
      'user': USER_TOKEN,
      'assistant': ASSISTANT_TOKEN,
      'system': SYSTEM_TOKEN}
  for message in conversation:
    content = message['content']
    role = message['role']
    text.append(f'{role_token_dict[role]}{content}{END_OF_MESSAGE_TOKEN}')
  return ''.join(text)


################################################################################
# Mixtures.


# ###############################################################################
# # Dataset utilities.


class DataSourceRegistry(registry.RootRegistry):
  """Data source registry."""
  namespace: ClassVar[str] = 'datasource'


class SimpleDataSource(Protocol):

  def __len__(self):
    ...

  def __getitem__(self, index: int):
    ...


@functools.partial(DataSourceRegistry.register, name='simply_json:gsm8k_train')
@dataclasses.dataclass(frozen=True)
class GSM8KJSONTrain(SimpleDataSource):
  """GSM8K dataset in json format."""
  path: str = os.path.join(DATASETS_DIR, 'gsm8k/gsm8k.json')
  example_start_index: int | None = None
  example_end_index: int | None = None
  split: str = 'train'

  def load(self):
    with epath.Path(self.path).open('r') as f:
      data = json.load(f)
    examples = data[self.split]
    for i, example in enumerate(examples):
      example['uid'] = f'gsm8k_{self.split}-{i}'
      example['id'] = i
    return examples[self.example_start_index:self.example_end_index]


@functools.partial(DataSourceRegistry.register, name='simply_json:gsm8k_test')
@dataclasses.dataclass(frozen=True)
class GSM8KJSONTest(GSM8KJSONTrain):
  split: str = 'test'


def register_gsm8k_json_variants():
  config = GSM8KJSONTrain()
  for num_examples in [4, 32, 128]:
    new_config = dataclasses.replace(
        config, example_start_index=0, example_end_index=num_examples)
    DataSourceRegistry.register_value(
        new_config, name=f'simply_json:gsm8k_train{num_examples}')

register_gsm8k_json_variants()


@functools.partial(
    DataSourceRegistry.register, name='simply_json:simple_qa_test'
)
@dataclasses.dataclass(frozen=True)
class SimpleQATest(SimpleDataSource):
  """Simple QA dataset in json format.

  Source: https://openai.com/index/introducing-simpleqa/
  """

  path: str = os.path.join(DATASETS_DIR, 'simple_qa/simple_qa_test_set.json')
  split: str = 'test'

  def load(self):
    with epath.Path(self.path).open('r') as f:
      data = json.load(f)
    examples = data[self.split]
    for i, example in enumerate(examples):
      example['uid'] = f'simple_qa_{self.split}-{i}'
      example['id'] = i
    return examples


@functools.partial(
    DataSourceRegistry.register, name='simply_json:simple_qa_num'
)
@dataclasses.dataclass(frozen=True)
class SimpleQATestNumberOnly(SimpleQATest):
  """Simple QA dataset with only number-only answers."""

  path: str = os.path.join(
      DATASETS_DIR, 'simple_qa/simple_qa_test_set_number_only.json')


@functools.partial(DataSourceRegistry.register, name='simply_json:mmlu_test')
@dataclasses.dataclass(frozen=True)
class MMLUJSONTest(SimpleDataSource):
  """MMLU dataset in json format."""
  path: str = os.path.join(DATASETS_DIR, 'mmlu/mmlu.json')
  example_start_index: int | None = None
  example_end_index: int | None = None
  split: str = 'test'

  def load(self):
    with epath.Path(self.path).open('r') as f:
      data = json.load(f)
    examples = data['data'][self.split]
    for i, example in enumerate(examples):
      example['uid'] = f'mmlu_{self.split}-{i}'
      example['id'] = i
    return examples[self.example_start_index:self.example_end_index]


@functools.partial(
    DataSourceRegistry.register, name='simply_json:dsr40k_train')
@dataclasses.dataclass(frozen=True)
class DeepScaleRJSONTrain(SimpleDataSource):
  """DeepScaleR dataset in json format."""
  path: str = os.path.join(DATASETS_DIR, 'deepscaler/deepscaler.json')
  example_start_index: int | None = None
  example_end_index: int | None = None

  def load(self):
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
      new_examples.append({
          'question': example['problem'],
          'short_answer': example['answer'],
          'answer': example['solution'],
          'uid': f'dsr40k_train-{i}',
          'id': i,
      })
    return new_examples[self.example_start_index:self.example_end_index]


# TODO: add a unified interface for filtering AIME examples
@functools.partial(
    DataSourceRegistry.register, name='simply_json:aime24')
@dataclasses.dataclass(frozen=True)
class AIME24JSON(SimpleDataSource):
  """AIME24 dataset in json format."""
  path: str = os.path.join(DATASETS_DIR, 'aime/aime_v2.json')
  example_start_index: int | None = None
  example_end_index: int | None = None

  def load(self):
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
      if int(example['year']) == 2024:
        # using the same keys as DeepScaleR
        new_examples.append({
            'question': example['problem'],
            'short_answer': example['answer'],
            'answer': example['solution'],
            'uid': f'aime24-{i}',
            'id': i,
        })
    return new_examples[self.example_start_index:self.example_end_index]


@functools.partial(
    DataSourceRegistry.register, name='simply_json:aime25')
@dataclasses.dataclass(frozen=True)
class AIME25JSON(SimpleDataSource):
  """AIME25 dataset in json format."""
  path: str = os.path.join(DATASETS_DIR, 'aime/aime_v2.json')
  example_start_index: int | None = None
  example_end_index: int | None = None

  def load(self):
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
      if int(example['year']) == 2025:
        # using the same keys as DeepScaleR
        new_examples.append({
            'question': example['problem'],
            'short_answer': example['answer'],
            'answer': example['solution'],
            'uid': f'aime25-{i}',
            'id': i,
        })
    return new_examples[self.example_start_index:self.example_end_index]


# TODO: check the 14B eval accuracy
@functools.partial(
    DataSourceRegistry.register, name='simply_json:math500_test')
@dataclasses.dataclass(frozen=True)
class MATH500JSONTest(SimpleDataSource):
  """MATH500 test set in json format."""
  path: str = os.path.join(DATASETS_DIR, 'math500/test.json')
  example_start_index: int | None = None
  example_end_index: int | None = None

  def load(self):
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
      # using the same keys as DeepScaleR
      new_examples.append({
          'question': example['problem'],
          'short_answer': example['answer'],
          'answer': example['solution'],
          'subject': example['subject'],
          'level': example['level'],
          'original_unique_id': example['unique_id'],
          'uid': f'math500_test-{i}',
          'id': i,
      })
    return new_examples[self.example_start_index:self.example_end_index]


# TODO: check the 14B eval accuracy
@functools.partial(
    DataSourceRegistry.register, name='simply_json:gpqa_diamond')
@dataclasses.dataclass(frozen=True)
class GPQADiamondJSON(SimpleDataSource):
  """GPQA-Diamond dataset in json format."""
  path: str = os.path.join(DATASETS_DIR, 'gpqa/gpqa_diamond.json')
  example_start_index: int | None = None
  example_end_index: int | None = None

  def load(self):
    with epath.Path(self.path).open('r') as f:
      examples = json.load(f)
    new_examples = []
    for i, example in enumerate(examples):
      # using the same keys as DeepScaleR
      new_examples.append({
          'question': example['Question'],
          'correct_answer': example['Correct Answer'],
          'incorrect_answer_1': example['Incorrect Answer 1'],
          'incorrect_answer_2': example['Incorrect Answer 2'],
          'incorrect_answer_3': example['Incorrect Answer 3'],
          'example_id': example['Record ID'],
          'uid': f'gpqa_diamond-{i}',
          'id': i,
      })
    return new_examples[self.example_start_index:self.example_end_index]


def create_simple_dataset(
    name: str, batch_size: int, seed: int, shuffle: bool, num_epochs: int | None
) -> grain.IterDataset[common.PyTree]:
  datasource = DataSourceRegistry.get_instance(name)
  data = datasource.load()
  dataset = grain.MapDataset.source(data)
  if shuffle:
    dataset = dataset.shuffle(seed=seed)
  return (
      dataset.repeat(num_epochs)
      .batch(batch_size, batch_fn=lambda x: x)
      .to_iter_dataset()
  )


def _get_dataset_info(dataset_name: str) -> tuple[str, str, str, Any]:
  """Parse dataset name and return (base_name, vocab_name, ds_type, info).

  Dataset name format: <base_name>.<vocab_name>
  Returns: base_name, vocab_name, dataset_type ('huggingface' or 'file'), info
  """
  parts = dataset_name.rsplit('.', 1)
  if len(parts) != 2:
    raise ValueError(f'Invalid dataset name format: {dataset_name}. '
                     'Expected format: <base_name>.<vocab_name>')
  base_name, vocab_name = parts

  # Check HuggingFace datasets
  for name, hf_name, hf_config, splits, vocabs in _HF_DATASETS:
    for vname, _ in vocabs:
      if name == base_name and vname == vocab_name:
        return base_name, vocab_name, 'huggingface', (hf_name, hf_config, splits)

  # Check file-based datasets (parquet/jsonl)
  file_configs = _get_file_configs()
  for name, config in file_configs.items():
    if name == base_name:
      for vname, _ in config['vocabs']:
        if vname == vocab_name:
          return base_name, vocab_name, 'file', config

  raise ValueError(f'Unknown dataset: {dataset_name}')


def _get_file_configs() -> dict[str, dict]:
  """Return file-based dataset configurations (parquet/jsonl).

  NOTE: These dataset paths have been updated from TFRecord format to parquet/jsonl.
  You may need to convert your existing TFRecord datasets to parquet format.
  """
  return {
      'the_pile_lm': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'pile/parquet/train*.parquet'),
              'validation': os.path.join(DATASETS_DIR, 'pile/parquet/val*.parquet'),
              'test': os.path.join(DATASETS_DIR, 'pile/parquet/test*.parquet'),
          },
          'file_type': 'parquet',
          'vocabs': PILE_VOCABS,
      },
      'redpajama_1t_arxiv': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'redpajama_1t/parquet/arxiv*.parquet'),
          },
          'file_type': 'parquet',
          'vocabs': OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS + OPENMIX_V3_VOCABS,
      },
      'redpajama_1t_wikipedia': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'redpajama_1t/parquet/wikipedia*.parquet'),
          },
          'file_type': 'parquet',
          'vocabs': OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS + OPENMIX_V3_VOCABS,
      },
      'redpajama_1t_book': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'redpajama_1t/parquet/book*.parquet'),
          },
          'file_type': 'parquet',
          'vocabs': OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS + OPENMIX_V3_VOCABS,
      },
      'redpajama_1t_stackexchange': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'redpajama_1t/parquet/stackexchange*.parquet'),
          },
          'file_type': 'parquet',
          'vocabs': OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS + OPENMIX_V3_VOCABS,
      },
      'starcoder': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'starcoder/parquet/train*.parquet'),
          },
          'file_type': 'parquet',
          'vocabs': OPENMIX_V1_VOCABS,
      },
      'refinedweb': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'refinedweb/parquet/train*.parquet'),
          },
          'file_type': 'parquet',
          'vocabs': OPENMIX_V1_VOCABS,
      },
      'fineweb_edu': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'fineweb-edu/parquet/train*.parquet'),
          },
          'file_type': 'parquet',
          'vocabs': [['fwedu_100864_v1', FWEDU_100864_V1_VOCAB]] + OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS,
      },
      'dclm_baseline_1p0': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'dclm-baseline-1p0/parquet/**/*.parquet'),
          },
          'file_type': 'parquet',
          'vocabs': OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS + OPENMIX_V3_VOCABS,
      },
      'stack_v2_smol_repo': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'stack_v2/parquet/train-smol-1/*.parquet'),
          },
          'file_type': 'parquet',
          'vocabs': OPENMIX_V2_VOCABS + OPENMIX_V3_VOCABS,
      },
      'stack_v2_smol_file': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'stack_v2/parquet/train-smol-1-file/*.parquet'),
          },
          'file_type': 'parquet',
          'vocabs': OPENMIX_V2_VOCABS + OPENMIX_V3_VOCABS,
      },
      'openhermes_2p5': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'openhermes-2p5/train.jsonl'),
          },
          'file_type': 'jsonl',
          'vocabs': OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS + OPENMIX_V3_VOCABS,
          'text_key': 'conversation',
          'text_preprocessor': process_conversation,
      },
      'tulu_v2_sft': {
          'file_patterns': {
              'train': os.path.join(DATASETS_DIR, 'tulu-v2-sft-mixture/train.jsonl'),
          },
          'file_type': 'jsonl',
          'vocabs': OPENMIX_V1_VOCABS + OPENMIX_V2_VOCABS,
          'text_key': 'conversation',
          'text_preprocessor': process_conversation,
      },
  }


def _get_vocab_path(vocab_name: str, vocabs: list[tuple[str, str]]) -> str:
  """Get vocab path from vocab name."""
  for vname, vpath in vocabs:
    if vname == vocab_name:
      return vpath
  raise ValueError(f'Vocab not found: {vocab_name}')


def create_iter_dataset(
    config, training: bool = True
) -> grain.IterDataset[common.PyTree]:
  """Create a Grain IterDataset for training or evaluation.

  This function supports:
  - simply_json:* datasets (JSON-based, uses create_simple_dataset)
  - HuggingFace datasets (lm1b, c4, imdb_reviews, etc.)
  - File-based datasets in parquet/jsonl format
  """
  dataset_name = config.dataset_name
  batch_size = config.batch_size

  if training:
    split = 'train'
    shuffle = True
    num_epochs = None
  else:
    split = 'validation'
    if config.validation_dataset_name:
      dataset_name = config.validation_dataset_name
    if config.validation_eval_batch_size > 0:
      batch_size = config.validation_eval_batch_size
    shuffle = False
    num_epochs = config.validation_eval_epochs

  # Handle simply_json datasets (already Grain-native)
  if dataset_name.startswith('simply_json:'):
    return create_simple_dataset(
        dataset_name, batch_size, config.dataset_seed, shuffle, num_epochs
    )

  # Parse dataset name
  base_name, vocab_name, ds_type, ds_info = _get_dataset_info(dataset_name)

  # Get vocab
  bos_id = getattr(config, 'bos_id', 0)

  if ds_type == 'huggingface':
    hf_name, hf_config, splits = ds_info
    # Find vocab path
    for name, hf_n, hf_c, sp, vocabs in _HF_DATASETS:
      if name == base_name:
        vocab_path = _get_vocab_path(vocab_name, vocabs)
        break
    vocab = tokenization.SentencePieceVocabulary(vocab_path)

    # Get the split string
    split_str = splits.get(split, split)

    # Create HuggingFace data source
    data_source = HuggingFaceDataSource(
        dataset_name=hf_name,
        split=split_str,
        text_key='text',
        config_name=hf_config,
    )

  elif ds_type == 'file':
    file_config = ds_info
    vocab_path = _get_vocab_path(vocab_name, file_config['vocabs'])
    vocab = tokenization.SentencePieceVocabulary(vocab_path)

    file_pattern = file_config['file_patterns'].get(split)
    if file_pattern is None:
      raise ValueError(f'Split {split} not available for dataset {base_name}')

    file_type = file_config.get('file_type', 'parquet')
    text_key = file_config.get('text_key', 'text')
    text_preprocessor = file_config.get('text_preprocessor', None)

    # Create file-based data source
    if file_type == 'parquet':
      data_source = ParquetDataSource(
          file_pattern=file_pattern,
          text_key=text_key,
          text_preprocessor=text_preprocessor,
      )
    elif file_type == 'jsonl':
      data_source = JSONLDataSource(
          file_pattern=file_pattern,
          text_key=text_key,
          text_preprocessor=text_preprocessor,
      )
    else:
      raise ValueError(f'Unknown file type: {file_type}')

  else:
    raise ValueError(f'Unknown dataset type: {ds_type}')

  # Build Grain pipeline
  dataset = grain.MapDataset.source(data_source)

  # Apply tokenization transform
  dataset = dataset.map(TokenizeTransform(
      vocab=vocab,
      text_key='text',
      add_eos=True,
      add_bos=False,
  ))

  # Apply LM feature converter
  dataset = dataset.map(LMFeatureConverter(
      seq_len=config.seq_len,
      bos_id=bos_id,
  ))

  # Shuffle if needed
  if shuffle:
    dataset = dataset.shuffle(seed=config.dataset_seed)

  # Repeat and batch
  dataset = (
      dataset
      .repeat(num_epochs)
      .batch(batch_size, drop_remainder=True)
  )

  # Convert to IterDataset and add multiprocessing prefetch
  return dataset.to_iter_dataset().mp_prefetch(
      grain.MultiprocessingOptions(
          num_workers=config.prefetch_num_workers,
          per_worker_buffer_size=config.prefetch_per_worker_buffer_size,
      )
  )


def create_chat_loss_mask(token_ids, mask_start_id, mask_end_id):
  def f(carry, a):
    new_carry = jnp.where(
        a == mask_end_id, -2, jnp.where(a == mask_start_id, -1, carry)
    )
    return new_carry, carry

  token_ids = einops.rearrange(token_ids, 'b t -> t b')
  result = jax.lax.scan(f, jnp.full(token_ids.shape[1], -2), token_ids)[1] + 2
  return einops.rearrange(result, 't b -> b t')


def add_chat_loss_mask(batch, mask_start_id, mask_end_id):
  batch['decoder_loss_weights'] = create_chat_loss_mask(
      batch['decoder_target_tokens'], mask_start_id=mask_start_id,
      mask_end_id=mask_end_id) * batch['decoder_loss_weights']
  return batch
