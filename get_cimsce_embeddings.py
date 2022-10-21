import argparse
import pdb
import json
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import nltk 

import torch 
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

import utils
from logger import logger
from politifactDataloader import Dataset

def run(args):
  if not os.path.exists(args.data_root):
    logger.error("Data directory (%s) doesnt exist"%args.data_root)
    exit()
  if not os.path.exists(args.out_dir):
    logger.warning("Output directory (%s) doesnt exist"%args.out_dir)
    os.makedirs(args.out_dir)

  dataset = Dataset(args.data_root, concat_before=args.concat_before, 
    concat_after=args.concat_after)
  verified_claims = dataset.verified_claims
  verified_claims = verified_claims.replace(np.nan, '', regex=True)

  transcripts = dataset.transcripts
  transcript_names = list(transcripts.keys())
  transcript_names.sort(key=lambda name: len(transcripts[name]))

  tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
  model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

  device = torch.device('cuda')
  tokenizer.to(device)
  model.to(device)

  if 'vclaim' in args.input:
    batch_size = 500
    logger.info('Getting cimsce embeddings for vclaims')
    out_path = os.path.join(args.out_dir, 'vclaim.npy')
    vclaims = verified_claims.vclaim.tolist()
    vclaims = [vclaims[i: i+batch_size] for i in range(0, len(vclaims), batch_size)]
    vclaims_embeddings_list = []
    for vclaim_batch in tqdm(vclaims):
      inputs = tokenizer(vclaim_batch, padding=True, truncation=True, return_tensors="pt")
      with torch.no_grad():
        vclaim_embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
      vclaims_embeddings_list.extend(vclaim_embeddings.tolist())
    np.save(out_path, vclaims_embeddings_list, allow_pickle=True)

  if 'title' in args.input:
    batch_size = 500
    logger.info('Getting cimsce embeddings for vclaim titles')
    out_path = os.path.join(args.out_dir, 'vclaim.title.npy')
    vclaim_title = verified_claims.title.tolist()
    vclaim_title = [vclaim_title[i: i+batch_size] for i in range(0, len(vclaim_title), batch_size)]
    vclaim_title_embeddings_list = []
    for vclaim_title_batch in tqdm(vclaim_title):
      inputs = tokenizer(vclaim_title_batch, padding=True, truncation=True, return_tensors="pt")
      with torch.no_grad():
        vclaim_title_embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
      vclaim_title_embeddings_list.extend(vclaim_title_embeddings.tolist())
    np.save(out_path, vclaim_title_embeddings_list, allow_pickle=True)

  if 'text' in args.input:
    logger.info('Getting cimsce embeddings for vclaim text')
    out_path = os.path.join(args.out_dir, 'vclaim.text.npy')
    vclaim_texts = verified_claims.text
    article_embeddings = []
    for vclaim_text in tqdm(vclaim_texts[:3]):
      vclaim_text = nltk.sent_tokenize(vclaim_text)
      if len(vclaim_text) == 0:
        text_embedding = np.array([])
      else:
        tokenized_text = tokenizer(vclaim_text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
          text_embedding = model(**tokenized_text, output_hidden_states=True, return_dict=True).pooler_output
        article_embeddings.append(text_embedding.tolist())
    np.save(out_path, article_embeddings, allow_pickle=True)

  if 'transcript' in args.input:
    logger.info('Getting sentence embeddings')

    transcripts_path = os.path.join(args.out_dir, 'transcripts')

    if not os.path.exists(transcripts_path):
      logger.warning('Transcripts folder missing, creating one...')
      os.makedirs(transcripts_path)

    for transcript_name in tqdm(transcript_names[args.lower: args.upper]):
      out_path = os.path.join(transcripts_path, f'{transcript_name}.npy')

      if os.path.exists(out_path):
        logger.info(f'{transcript_name} present')
        continue
      else:
        transcript = transcripts[transcript_name]
        transcript = transcript.replace(np.nan, '', regex=True)
        sentences = transcript.sentence.tolist()
        logger.info(f'Processing debate sentence count: {len(sentences)}')

        batch_size = 200
        sentences = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]
        sentence_embeddings = []
        for sentence_batch in tqdm(sentences):
          inputs = tokenizer(sentence_batch, padding=True, truncation=True, return_tensors="pt")
          with torch.no_grad():
            sentence_batch_embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
          sentence_embeddings.extend(sentence_batch_embeddings.tolist())
        np.save(out_path, sentence_embeddings, allow_pickle=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get elasticsearch scores')
  parser.add_argument('--data-root', '-d', required=True, 
    help='Path to the dataset directory.')
  parser.add_argument('--out-dir', '-o', required=True, 
    help='Path to the output directory')
  parser.add_argument('--input', '-i', nargs='+', 
    default=['vclaim', 'title', 'text', 'transcript'], 
    choices=['vclaim', 'title', 'text', 'transcript'], 
    help='Path to the oconfig file')
  parser.add_argument('--concat-before', default=0, type=int,
    help='Number of sentences concatenated before the input sentence in a transcript')
  parser.add_argument('--concat-after', default=0, type=int,
    help='Number of sentences concatenated after the input sentence in a transcript')
  parser.add_argument('--lower', '-l', default=0, type=int)
  parser.add_argument('--upper', '-u', default=70, type=int)

  args = parser.parse_args()
  run(args)