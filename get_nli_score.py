import argparse
import numpy as np
import os
import pdb
import gc
import subprocess
from logger import logger
from tqdm import tqdm
from itertools import product

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from politifactDataloader import Dataset

def run(args):
  if not os.path.exists(args.data_root):
    logger.error("File not found")
    exit()
  if not os.path.exists(args.output):
    logger.warning("Output dir missing. Creating one...")
    os.makedirs(args.output)

  torch_device = torch.device('cuda')

  dataset = Dataset(args.data_root, concat_before=0, concat_after=0)

  verified_claims = dataset.verified_claims
  verified_claims = verified_claims.replace(np.nan, '', regex=True)
  verified_claims = verified_claims.vclaim

  transcripts = dataset.transcripts
  transcripts_names = [name for name in transcripts.keys()]
  transcripts_names.sort(key=(lambda name: len(transcripts[name])))

  max_length = 256

  hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
  # hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
  # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
  # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
  # hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

  tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
  model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

  model = model.to(torch_device)

  vclaim_scores_dim = args.elasticsearch_max_length

  logger.info(f'Processing debates {args.transcript_start} to {args.transcript_end}')
  for transcript_name in tqdm(transcripts_names[args.transcript_start:args.transcript_end]):

      transcript = transcripts[transcript_name]
      transcript = transcript.replace(np.nan, '', regex=True)
      transcript_opath = os.path.join(args.output, '%s.npz'%(transcript_name))

      elasticsearch_transcript = np.load(os.path.join(args.elasticsearch_dir, transcript_name+'.npz'), allow_pickle=True)
      vclaim_selected_idxs = elasticsearch_transcript[args.measure][0][:, :vclaim_scores_dim].astype('int')

      transcript_verified_claims = [verified_claims[sentence_claim_indxs].tolist() for sentence_claim_indxs in vclaim_selected_idxs]

      transcript_sentences = transcript.sentence.values.tolist()
      logger.info(f'Processing debate {transcript_name} sent. count: {len(transcript_sentences)}')

      batch_size=args.batch_size
      sentence_claims_pairs = list(zip(transcript_sentences, transcript_verified_claims))
      sentence_scores = []

      print("batch size: {}".format(batch_size))
      print("sentence_claims_pairs size: {}".format(len(sentence_claims_pairs)))

      for batch_number in tqdm(range(0, len(sentence_claims_pairs), batch_size)):
        logger.info(f'Processing batch number: {batch_number}')
        if os.path.exists(transcript_opath):
          logger.info(f'File for {transcript_name} already exists')
          continue
        else:
          batch_sentence_pairs = []
          for (sentence, sentence_claims) in sentence_claims_pairs[batch_number: batch_number+batch_size]:
            sentence_pairs = [list(pair) for pair in product(sentence_claims, [sentence])]
            batch_sentence_pairs.extend(sentence_pairs)
            del sentence_pairs

          print('batch_sentence_length', len(batch_sentence_pairs))

          encoded_results = tokenizer.batch_encode_plus(batch_sentence_pairs, max_length=max_length, return_token_type_ids=True, truncation=True, padding=True)
          del batch_sentence_pairs


          input_ids = torch.Tensor(encoded_results['input_ids']).long().to(torch_device)
          token_types = torch.Tensor(encoded_results['token_type_ids']).long().to(torch_device)
          att_mask = torch.Tensor(encoded_results['attention_mask']).long().to(torch_device)
          del encoded_results

          outputs = model(input_ids,
                          attention_mask=att_mask,
                          token_type_ids=token_types,
                          labels=None)
          # Since there was not enought memory, I'm computing them into batches
          # output_list = []
          # pair_batch_size = 20
          # for encoded_pair_indx in range(0, len(input_ids),pair_batch_size):
          #   if encoded_pair_indx % 1000 == 0:
          #     logger.debug(f'Processing sentence {encoded_pair_indx}')
          #   outputs = model(input_ids[encoded_pair_indx:encoded_pair_indx+pair_batch_size],
          #                   attention_mask=att_mask[encoded_pair_indx:encoded_pair_indx+pair_batch_size],
          #                   token_type_ids=token_types[encoded_pair_indx:encoded_pair_indx+pair_batch_size],
          #                   labels=None)
          #   output_list.extend(outputs[0].tolist())
          #   del outputs

          del input_ids, token_types, att_mask
          # output_list = torch.Tensor(output_list)
          # predicted_probability = torch.softmax(output_list, dim=1)
          predicted_probability = torch.softmax(outputs[0], dim=1)
          predicted_probability = predicted_probability.tolist()
          predicted_probability = [predicted_probability[i:i+vclaim_scores_dim] for i in range(0,len(predicted_probability), vclaim_scores_dim)]
          # del output_list

          sentence_scores.extend(predicted_probability)
          del predicted_probability
      sentence_scores = clear_invalid_claims_scores(sentence_scores, vclaim_selected_idxs)
      np.savez(transcript_opath, scores=sentence_scores, allow_pickle=True)

def clear_invalid_claims_scores(sentence_scores, vclaim_indxs):
  transcript_scores = []
  for sent_scores, claim_indxs in list(zip(sentence_scores,vclaim_indxs)):
    processed_scores = []
    for scores, claim_indx in list(zip(sent_scores, claim_indxs)):
      if claim_indx != 0:
        processed_scores.append(scores)
      else:
        empty_scores = np.zeros(3).tolist()
        processed_scores.append(empty_scores)
    transcript_scores.append(processed_scores)
  return transcript_scores


def get_claims_from_idxs(claims_idxs, claims_list):
  return [claims_list[idx] for idx in get_unique_claim_idxs(claims_idxs)]

def get_unique_claim_idxs(claims_idxs):
  claims_idxs, idxs = np.unique(claims_idxs, return_index=True)
  claims_idxs = list(zip(claims_idxs, idxs))
  claims_idxs.sort(key=lambda elem: elem[1])
  return [elem[0] for elem in claims_idxs]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get NLI score values')
  parser.add_argument('--data-root', '-d', required=True)
  parser.add_argument('--output', '-o', required=True)
  parser.add_argument('--elasticsearch-dir', '-e', required=True)
  parser.add_argument('--elasticsearch-max-length', '-n', required=True, type=int, help='Dimension of the claims list retrieved from elasticsearch')
  parser.add_argument('--measure', '-m', default='vclaim')
  parser.add_argument('--transcript-start', '-start', default=0, type=int)
  parser.add_argument('--transcript-end', '-end', default=70, type=int)
  parser.add_argument('--batch-size', '-s', default=100, type=int)

  args = parser.parse_args()
  run(args)