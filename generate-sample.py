import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

output_dir = './model_bn_custom/'

tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model = TFGPT2LMHeadModel.from_pretrained(output_dir)

text = "To start with "
# encoding the input text
input_ids = tokenizer.encode(text, return_tensors='tf')
# getting out output
beam_output = model.generate(
  input_ids,
  max_length = 200,
  num_beams = 5,
  temperature = 0.7,
  no_repeat_ngram_size=2,
  num_return_sequences=5
)

print(tokenizer.decode(beam_output[0]))