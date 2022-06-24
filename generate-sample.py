#Tuned using principles from https://huggingface.co/blog/how-to-generate

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

text = "Harry looked at "
# encoding the input text
input_ids = tokenizer.encode(text, return_tensors='tf')
# getting out output
beam_output = model.generate(
  input_ids,
  # num_beams = 5,
  # no_repeat_ngram_size = 2,
  # temperature = 0.7
  # early_stopping=True
  max_length = 200,
  do_sample=True,
  top_k = 50,
  top_p = 0.95,
  num_return_sequences = 3
)

print(tokenizer.decode(beam_output[0]))

with open("sample.txt", "a") as file_object:
  file_object.write(tokenizer.decode(beam_output[0]))
