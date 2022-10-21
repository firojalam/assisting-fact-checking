import argparse
import numpy as np
import os
import pdb
from logger import logger
from tqdm import tqdm
import torch

from itertools import product

from bert_score import BERTScorer
from politifactDataloader import Dataset

def run(args):
  if not os.path.exists(args.data_root):
    logger.error("File not found")
    exit()
  if not os.path.exists(args.output):
    logger.warning("Output dir missing. Creating one...")
    os.makedirs(args.output)

  dataset = Dataset(args.data_root, concat_before=args.concat_before, concat_after=args.concat_after)
  
  torch_device = 'cuda'

  scorer = BERTScorer(lang="en", rescale_with_baseline=args.rescale_with_baseline, device=torch_device)


  verified_claims = dataset.verified_claims
  verified_claims = verified_claims.replace(np.nan, '', regex=True)
  verified_claims = verified_claims.vclaim.values.tolist()

  transcripts = dataset.transcripts
  transcripts_names = [name for name in transcripts.keys()]
  transcripts_names.sort(key=(lambda name: len(transcripts[name])))

  vclaim_scores_dim = args.elasticsearch_max_length

  for transcript_name in tqdm(transcripts_names):
      transcript = transcripts[transcript_name]
      transcript = transcript.replace(np.nan, '', regex=True)
      transcript_opath = os.path.join(args.output, '%s.npz'%(transcript_name))
      
      #get top N vclaims for every sentence from elasticsearch 
      elasticsearch_transcript = np.load(os.path.join(args.elasticsearch_dir, transcript_name+'.npz'), allow_pickle=True)
      vclaim_seleced_idxs = elasticsearch_transcript[args.measure][0][:, :vclaim_scores_dim].astype('int')
      
      if os.path.exists(transcript_opath):
        logger.info(f'File for {transcript_name} already exists')
        continue
      else:
        transcript_sentences = transcript.sentence.values.tolist()

        transcript_sentences_length = len(transcript_sentences)
        logger.info(f'Calculating bert scores for {transcript_name}, sent. count: {transcript_sentences_length}')

        transcript_targets = []
        transcript_refs = []
        sentence_scores_lengths = []
        transcript_P = []
        transcript_R = []
        transcript_F1 = []
        for (sentence, claims_indxs) in list(zip(transcript_sentences, vclaim_seleced_idxs)):
          sentence_claims = get_claims_from_idxs(claims_indxs, verified_claims)
          sentence_scores_lengths.append(len(sentence_claims))
          if len(sentence_claims) != 0:
            sentence_pairs = product([sentence], sentence_claims)
            targets, refs = get_encode_lists(sentence_pairs)
            transcript_targets = np.append(transcript_targets, targets).tolist()
            transcript_refs = np.append(transcript_refs, refs).tolist()
        
        print(sentence_scores_lengths)
        P, R, F1 = scorer.score(transcript_targets, transcript_refs, verbose=args.verbose)
        P = P.tolist()
        R = R.tolist()
        F1 = F1.tolist()
        
        i = 0
        for sent_scores_count in sentence_scores_lengths:
          if sent_scores_count != 0:
            P_scores = P[i:i+sent_scores_count]
            R_scores = R[i:i+sent_scores_count]
            F1_scores = F1[i:i+sent_scores_count]

            i+=sent_scores_count
            
            # padd if needed
            padd_length = vclaim_scores_dim - sent_scores_count
            if padd_length != 0:
              empty_scores = np.zeros(padd_length).tolist()
              P_scores.extend(empty_scores)
              R_scores.extend(empty_scores)
              F1_scores.extend(empty_scores)

            transcript_P.append(P_scores)
            transcript_R.append(R_scores)
            transcript_F1.append(F1_scores)
          else:
            empty_scores = np.zeros(vclaim_scores_dim).tolist()
            transcript_P.append(empty_scores)
            transcript_R.append(empty_scores)
            transcript_F1.append(empty_scores)

          # transcript_targets = np.append(transcript_targets, targets).tolist()
          # transcript_refs = np.append(transcript_refs, refs).tolist()

        # P, R, F1 = scorer.score(transcript_targets, transcript_refs, verbose=args.verbose)

        # sorted_P = [P[i:i+vclaim_scores_dim].tolist() for i in range(0,len(P),vclaim_scores_dim)]
        # sorted_R = [R[i:i+vclaim_scores_dim].tolist() for i in range(0,len(R),vclaim_scores_dim)]
        # sorted_F1 = [F1[i:i+vclaim_scores_dim].tolist() for i in range(0,len(F1),vclaim_scores_dim)]

        np.savez(transcript_opath, precision=transcript_P, recall=transcript_R, f1=transcript_F1, allow_pickle=True)

def get_encode_lists(sentence_pairs):
  targets = []
  refs = []

  for (target, ref) in sentence_pairs:
    targets.append(target)
    refs.append(ref)

  return targets, refs

def get_claims_from_idxs(claims_idxs, claims_list):
  vclaims = []
  for claim_indx in claims_idxs:
    if claim_indx != 0:
      vclaims.append(claims_list[claim_indx])
  return vclaims

# def get_claims_from_idxs(claims_idxs, claims_list):
#   vlaims_list = [claims_list[idx] for idx in get_unique_claims_idxs(claims_idxs)]
#   results_padd_length = 100 - len(vlaims_list)

#   if results_padd_length != 0:
#     return np.pad(vlaims_list, (0,results_padd_length), mode='constant', constant_values='').tolist()
#   else:
#     return vlaims_list

# def get_unique_claims_idxs(claims_idxs):
#   claims_idxs, idxs = np.unique(claims_idxs, return_index=True)
#   claims_idxs = list(zip(claims_idxs, idxs))
#   claims_idxs.sort(key=lambda elem: elem[1])
#   return [elem[0] for elem in claims_idxs]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get BERT score values')
  parser.add_argument('--data-root', '-d', required=True, help='Path to the dataset directory.')
  parser.add_argument('--output', '-o', required=True, help='Path where files with debate scores will be stored')
  parser.add_argument('--elasticsearch-dir', '-e', required=True)
  parser.add_argument('--elasticsearch-max-length', '-n', required=True, type=int, help='Dimension of the claims list retrieved from elasticsearch')
  parser.add_argument('--measure', '-m', default='vclaim')
  parser.add_argument('--rescale-with-baseline', default=True, help='If this value is True, all scores will be rescaled to be more human-readable')
  parser.add_argument('--verbose', default=True)
  parser.add_argument('--concat-before', default=0, type=int)
  parser.add_argument('--concat-after', default=0, type=int)

  args = parser.parse_args()
  run(args)