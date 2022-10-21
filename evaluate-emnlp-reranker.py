import re
import sys
import pickle
import argparse
import pdb
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from elasticsearch import Elasticsearch
import imblearn
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn import datasets

from politifactDataloader import Dataset
from logger import logger
import utils

np.random.seed(0)

data_dir = '../data/politifact'
elasticsearch_dir = f'{data_dir}/elasticsearch'
man_dir = '../verdict-annotations'
ranksvm_dir = f'{data_dir}/ranksvm_bs'
bert_scores_dir = f'{data_dir}/bert-scores-vclaim'
nli_scores_dir = f'{data_dir}/nli_scores/'
cimsce_embeddings_dir = f'{data_dir}/cimsce'
sbert_embeddings_dir = f'{data_dir}/SBERT.large.embeddings'

TRANSCRIPTS = ['20170803_Trump_WV', '20180612_Trump_Singapore',
  '20170822_Trump_phoenix', '20180615_Trump_lawn', 
  '20180426_Trump_Fox_Friends', '20180628_Trump_NorthDakota', 
  '20180525_Trump_Naval']

dataset = Dataset(data_dir)
verifiedClaims = dataset.verified_claims
print(verifiedClaims)
load = True

top_n = int(sys.argv[-1])

# elasticesearch default params
elasticsearch_top_n = 100
elasticsearch_measure= 'text'

threshs_MAPs = [1, 3]

def load_predictions(fpath):
  scores = []
  with open(fpath) as f:
    for line in tqdm(f.readlines()):
      scores.append(float(line))
  return np.array(scores)

def dump_svmlight(X, y, qid, out_fpath):
  y = np.squeeze(y)
  X = np.squeeze(X)
  qid = np.squeeze(qid)
  print('dumping', y.shape, X.shape, qid.shape, out_fpath)
  datasets.dump_svmlight_file(X, y, out_fpath, zero_based=False, query_id=qid)

def get_labels(transcript, man_annotation):
  labels = np.zeros(len(transcript))
  for i, row in man_annotation.iterrows():
    if not row.verdict in ['unknown', 'repeated', 'not-claim']:
      labels[row.line_number-1] = 1
  return labels

def load_input(elasticsearch_transcript, sbert_transcript, verifiedClaims):
  elasticsearch_scores = []
  for measure in ['vclaim', 'title', 'text']:
    idx = elasticsearch_transcript[measure][0].astype('int')
    small_scores = elasticsearch_transcript[measure][1]
    scores = np.zeros((len(small_scores), len(verifiedClaims)))
    for sent_idx, (vclaim_idxs, vclaim_scores) in enumerate(zip(idx, small_scores)):
      scores[sent_idx][vclaim_idxs] = vclaim_scores
    elasticsearch_scores.append(np.expand_dims(scores, axis=2))
    print(elasticsearch_scores[-1].shape)
  print(sbert_transcript.shape)
  return np.concatenate(elasticsearch_scores + [sbert_transcript], axis=2)


def get_map(predictions, labels):
  predictions = predictions.squeeze()
  order = np.argsort(predictions)[::-1]
  predictions = predictions[order]
  labels = labels.squeeze()[order]
  total = labels.sum()
  cur = 0
  score = 0
  for rank, (p, l) in enumerate(zip(predictions, labels)):
    if total == cur:
      break
    if l:
      cur += 1
      score += cur/(rank+1)
  print('get_map', score, total, score/total, predictions.shape, labels.shape)
  return score/total

def get_map_n(predictions, labels, vclaim_scores, vclaims_labels, threshs, n=0):
  results = []
  predictions = predictions.squeeze()
  order = np.argsort(predictions)[::-1]
  predictions = predictions[order]
  labels = labels.squeeze()[order]
  vclaims_labels = vclaims_labels.squeeze()[order]
  vclaim_scores = vclaim_scores.squeeze()[order]
  total = labels.sum()
  for thresh in threshs:
    cur = 0
    cur2 = 0
    score = 0
    for rank, (p, l, vp, vl) in enumerate(zip(predictions, labels, vclaim_scores, vclaims_labels)):
      if total == cur2:
        break
      if l:
        order = np.argsort(vp)[::-1]
        vl = vl[order]
        if sum(vl[:thresh]):
          cur += 1
          score += cur/(rank+1)
        else:
          cur += n
          score += cur/(rank+1)
        cur2 += 1
    results.append(score/total)
  return results

def get_map_harsh(predictions, labels, vclaim_scores, vclaims_labels, threshs):
  results = []
  predictions = predictions.squeeze()
  order = np.argsort(predictions)[::-1]
  predictions = predictions[order]
  labels = labels.squeeze()[order]
  vclaims_labels = vclaims_labels.squeeze()[order]
  vclaim_scores = vclaim_scores.squeeze()[order]
  total = labels.sum()
  for thresh in threshs:
    cur = 0
    cur2 = 0
    score = 0
    for rank, (p, l, vp, vl) in enumerate(zip(predictions, labels, vclaim_scores, vclaims_labels)):
      if total == cur2:
        break
      order = np.argsort(vp)[::-1]
      vl = vl[order]
      if l and sum(vl[:thresh]):
        cur += 1
        score += cur/(rank+1)
      if l:
        cur2 += 1
    results.append(score/total)
  return results


def compute_MAP(labels, scores, top_k=-1):
  if len(labels) != len(scores):
    logger.error(
    "Failed computing MAP because leght of labels (%d) and scores (%d) are different"%(len(labels), len(scores)))
    return -1
  if top_k < 0 or top_k > labels.shape[1]:
    top_k = labels.shape[1]
  average_precision_scores = []
  for i, (label, score) in enumerate(zip(labels, scores)):
    sorted_indices = score.argsort()[::-1]
    score = score[sorted_indices]
    label = label[sorted_indices]
    label = label.astype(int)
    score[top_k:] = 0
    if sum(label) == 0:
      logger.error("Found something when computing MAP no labels (%d)"%sum(label))
      average_precision_score = 0
    else:
      average_precision_score = metrics.average_precision_score(label, score)
  average_precision_scores.append(average_precision_score)
  return np.mean(average_precision_scores), average_precision_scores


def get_map_inner(predictions, labels, vclaim_scores, vclaims_labels):
  p = vclaim_scores[np.where(labels)[0]]
  l = vclaims_labels[np.where(labels)[0]]
  return compute_MAP(l, p)[0]

def evaluate(predictions, labels, vclaim_scores, vclaims_labels):
  inner_map = get_map_inner(predictions, labels, vclaim_scores, vclaims_labels)
  MAP = get_map(predictions, labels)
  MAP_1 = get_map_n(predictions, labels, vclaim_scores, vclaims_labels, threshs_MAPs, n=0)
  MAP_0_5 = get_map_n(predictions, labels, vclaim_scores, vclaims_labels, threshs_MAPs, n=0.5)
  MAP_harsh = get_map_harsh(predictions, labels, vclaim_scores, vclaims_labels, threshs_MAPs)
  return MAP, MAP_1, MAP_0_5, MAP_harsh, inner_map

def get_vclaim_indxs_from_elasticsearch(elasticsearch_transcript, measure=elasticsearch_measure, esearch_top_n = elasticsearch_top_n):
  return elasticsearch_transcript[measure][0][:, :esearch_top_n].astype('int')

def load_input(elasticsearch_transcript, svm_preds, verifiedClaims):
  vclaim_seleced_idxs = get_vclaim_indxs_from_elasticsearch(elasticsearch_transcript)
  # svm_preds = svm_preds.reshape((-1, 100))
  scores = np.zeros((len(vclaim_seleced_idxs), len(verifiedClaims)))
  for i, (svm_pred, idx) in enumerate(zip(svm_preds, vclaim_seleced_idxs)):
    scores[i][idx[:len(svm_pred)]] = svm_pred
  return np.expand_dims(scores, 2)

def load_transcript_scores(transcript_name, scores_path, scores_file_ext, with_pickle=True):
  transcript_scores_path = os.path.join(scores_path, transcript_name+scores_file_ext)
  if not os.path.exists(transcript_scores_path):
    logger.error(f'Missing scores for {transcript_name} in {transcript_scores_path}')
    exit()
  scores = np.load(transcript_scores_path, allow_pickle=with_pickle)
  return scores

def get_bert_scores(transcript_name, elasticsearch_transcript, verifiedClaims):
  bert_scores = load_transcript_scores(transcript_name, bert_scores_dir, '.npz')
  bert_scores_f1 = bert_scores['f1']
  input = load_input(elasticsearch_transcript, bert_scores_f1, verifiedClaims)
  return input

def get_elsaticserch(transcript_name, elasticsearch_transcript, verifiedClaims):
  elasticsearch_transcript = load_transcript_scores(transcript_name, elasticsearch_dir, '.npz')
  scores = []
  for measure in ['vclaim', 'title', 'text']:
    if measure == 'all':
      pass
    idx = elasticsearch_transcript[measure][0].astype('int')
    small_scores = elasticsearch_transcript[measure][1]
    
    score = np.zeros((len(small_scores), len(verifiedClaims)))
    for sent_idx, (vclaim_idxs, vclaim_scores) in enumerate(zip(idx, small_scores)):
      score[sent_idx][vclaim_idxs] = vclaim_scores
    score = np.expand_dims(score, axis=2)
    print('elasticsearch scores shape: ', measure, score.shape)
    scores.append(score)
  return np.concatenate(scores, axis=2)

def get_nli_scores(transcript_name, elasticsearch_transcript, verifiedClaims):
  nli_scores = load_transcript_scores(transcript_name, nli_scores_dir, '.npz')
  nli_scores = nli_scores['scores']
  nli_scores_dim = 3
  vclaim_selected_indxs = get_vclaim_indxs_from_elasticsearch(elasticsearch_transcript)
  scores = np.zeros((len(vclaim_selected_indxs), len(verifiedClaims), nli_scores_dim))
  for i, (sentence_scores, claim_indxs) in enumerate(zip(nli_scores, vclaim_selected_indxs)):
    scores[i][claim_indxs] = sentence_scores
  print('NLI scores shape: ', scores.shape)
  return scores

def get_cimsce_embeddings(transcript_name, elasticsearch_transcript, verifiedClaims):
  vclaim_cimsce_embeddings_path = os.path.join(cimsce_embeddings_dir, 'vclaim.npy')
  vclaim_title_cimsce_embeddings_path = os.path.join(cimsce_embeddings_dir, 'vclaim.title.npy')
  if not os.path.exists(vclaim_cimsce_embeddings_path):
    logger.error(f'Missing cimsce vclaim embeddings in {vclaim_cimsce_embeddings_path}')
  if not os.path.exists(vclaim_title_cimsce_embeddings_path):
    logger.error(f'Missing cimsce vclaim titile embeddings in {vclaim_title_cimsce_embeddings_path}')
  
  transcript_cimsce_embeddings_dir = os.path.join(cimsce_embeddings_dir, 'transcripts')
  cimsce_transcript_embeddings = load_transcript_scores(transcript_name, transcript_cimsce_embeddings_dir, '.npy')
  cimsce_vclaims_embeddings = np.load(vclaim_cimsce_embeddings_path, allow_pickle=True)
  cimsce_vclaim_title_embeddings = np.load(vclaim_title_cimsce_embeddings_path, allow_pickle=True)
  cimsce_scores = []
  vclaim_selected_indxs = get_vclaim_indxs_from_elasticsearch(elasticsearch_transcript)
  for claim_embeddings in [cimsce_vclaims_embeddings, cimsce_vclaim_title_embeddings]:
    scores = np.zeros((len(cimsce_transcript_embeddings), len(verifiedClaims)))
    for i, (sentence_embeddings, claim_indxs) in enumerate(zip(cimsce_transcript_embeddings, vclaim_selected_indxs)):
      scores[i][claim_indxs] = metrics.pairwise.cosine_similarity([sentence_embeddings], claim_embeddings[claim_indxs])
    scores = np.expand_dims(scores, axis=2)
    print('Cimsce shape:', scores.shape)
    cimsce_scores.append(scores)
  return np.concatenate(cimsce_scores, axis=2)


def get_cimsce_text_embeddings(transcript_name, elasticsearch_transcript, verifiedClaims, top_k=4):
  transcript_cimsce_embeddings_dir = os.path.join(cimsce_embeddings_dir, 'transcripts')
  cimsce_transcript_embeddings = load_transcript_scores(transcript_name, transcript_cimsce_embeddings_dir, '.npy')
  
  vclaim_text_embeddings_path = os.path.join(cimsce_embeddings_dir, 'vclaim.text.npy')
  if not os.path.exists(vclaim_text_embeddings_path):
    logger.error(f'Missing cimsce vclaim embeddings in {vclaim_text_embeddings_path}')
    exit()
  sbert_vclaims_text = np.load(vclaim_text_embeddings_path, allow_pickle=True)
  rerank_transcript = []

  sbert_vclaims_text_scores = np.zeros((len(cimsce_transcript_embeddings), top_k, len(verifiedClaims)))
  print(sbert_vclaims_text.shape, sbert_vclaims_text_scores.shape, cimsce_transcript_embeddings.shape)
  for vclaim_id, sbert_embeddings in enumerate(tqdm(sbert_vclaims_text)):
    if not len(sbert_embeddings):
      continue
    # sbert_embeddings = sbert_embeddings.squeeze()
    vclaim_text_score = metrics.pairwise.cosine_similarity(cimsce_transcript_embeddings, sbert_embeddings)
    vclaim_text_score = np.sort(vclaim_text_score)
    n = min(top_k, len(sbert_embeddings))
    sbert_vclaims_text_scores[:, :n, vclaim_id] = vclaim_text_score[:, -n:]
  
  sbert_vclaims_text_scores = np.transpose(sbert_vclaims_text_scores, (0, 2, 1))
  print('Cimsce text shape:', sbert_vclaims_text_scores.shape)
  return sbert_vclaims_text_scores



def get_sbert_embeddings(transcript_name, elasticsearch_transcript, verifiedClaims):
  transcript_sbert_embeddings = load_transcript_scores(transcript_name, sbert_embeddings_dir, '.npy')
  vclaim_embeddings_path = os.path.join(data_dir, 'vclaim.npy')
  vclaim_title_embeddings_path = os.path.join(data_dir, 'vclaim.title.npy')
  if not os.path.exists(vclaim_embeddings_path):
    logger.error(f'Missing sbert vclaim embeddings in {vclaim_embeddings_path}')
    exit()
  if not os.path.exists(vclaim_title_embeddings_path):
    logger.error(f'Missing sbert vclaim title embeddings in {vclaim_title_embeddings_path}')
    exit()
  vclaim_embeddings = np.load(vclaim_embeddings_path, allow_pickle=True)
  vclaim_title_embeddings = np.load(vclaim_title_embeddings_path, allow_pickle=True)

  vclaim_selected_indxs = get_vclaim_indxs_from_elasticsearch(elasticsearch_transcript)
  sbert_scores = []
  for claim_embeddings in [vclaim_embeddings, vclaim_title_embeddings]:
    scores = np.zeros((len(transcript_sbert_embeddings), len(verifiedClaims)))
    for i, (sentence_embeddings, claim_indxs) in enumerate(zip(transcript_sbert_embeddings, vclaim_selected_indxs)):
      scores[i][claim_indxs] = metrics.pairwise.cosine_similarity([sentence_embeddings], claim_embeddings[claim_indxs])
    scores = np.expand_dims(scores, axis=2)
    print('Sbert shape:', scores.shape)
    sbert_scores.append(scores)
  return np.concatenate(sbert_scores, axis=2)

def get_sbert_text_embeddings(transcript_name, elasticsearch_transcript, verifiedClaims, top_k=4):
  transcript_sbert_embeddings = load_transcript_scores(transcript_name, sbert_embeddings_dir, '.npy')
  
  vclaim_text_embeddings_path = os.path.join(data_dir, 'vclaim.text.npy')
  if not os.path.exists(vclaim_text_embeddings_path):
    logger.error(f'Missing sbert vclaim embeddings in {vclaim_text_embeddings_path}')
    exit()
  sbert_vclaims_text = np.load(vclaim_text_embeddings_path, allow_pickle=True)
  sbert_vclaims_text_scores = np.zeros((len(transcript_sbert_embeddings), top_k, len(verifiedClaims)))
  for vclaim_id, sbert_embeddings in enumerate(tqdm(sbert_vclaims_text)):
    if not len(sbert_embeddings):
      continue
    # sbert_embeddings = sbert_embeddings.squeeze()
    vclaim_text_score = metrics.pairwise.cosine_similarity(transcript_sbert_embeddings, sbert_embeddings)
    vclaim_text_score = np.sort(vclaim_text_score)
    n = min(top_k, len(sbert_embeddings))
    sbert_vclaims_text_scores[:, :n, vclaim_id] = vclaim_text_score[:, -n:]
  sbert_vclaims_text_scores = np.transpose(sbert_vclaims_text_scores, (0, 2, 1))
  print('Sbert text shape:', sbert_vclaims_text_scores.shape)
  return sbert_vclaims_text_scores

def get_label_input_features(transcript_name, elasticsearch_transcript, verifiedClaims):
  transcript = dataset.transcripts[transcript_name]
  vclaim_labels = []
  for i, row in verifiedClaims.iterrows():
    s = re.sub(r'[^a-zA-Z ]+', '', row.truth_meter)
    if s.lower().replace(' ', '') == 'true':
      vclaim_labels.append(5)
    elif s.lower().replace(' ', '') == 'mostlytrue':
      vclaim_labels.append(4)
    elif s.lower().replace(' ', '') == 'halftrue':
      vclaim_labels.append(3)
    elif s.lower().replace(' ', '') == 'mostlyfalse':
      vclaim_labels.append(2)
    elif s.lower().replace(' ', '') == 'false':
      vclaim_labels.append(1)
    elif s.lower().replace(' ', '') == 'pantsonfire':
      vclaim_labels.append(0)
    else:
      vclaim_labels.append(-1)
  vclaim_labels = np.array(vclaim_labels).reshape((1, -1))
  scores = []
  for i in range(len(transcript)):
    scores.append(vclaim_labels)
  scores = np.concatenate(scores, axis=0)
  print('label shape:', scores.shape)
  return np.expand_dims(scores, axis=2)

def get_label_hot_input_features(transcript_name, elasticsearch_transcript, verifiedClaims):
  transcript = dataset.transcripts[transcript_name]
  vclaim_labels = []
  for i, row in verifiedClaims.iterrows():
    s = re.sub(r'[^a-zA-Z ]+', '', row.truth_meter)
    if s.lower().replace(' ', '') == 'true':
      vclaim_labels.append([0, 0, 0, 0, 0, 1])
    elif s.lower().replace(' ', '') == 'mostlytrue':
      vclaim_labels.append([0, 0, 0, 0, 1, 0])
    elif s.lower().replace(' ', '') == 'halftrue':
      vclaim_labels.append([0, 0, 0, 1, 0, 0])
    elif s.lower().replace(' ', '') == 'mostlyfalse':
      vclaim_labels.append([0, 0, 1, 0, 0, 0])
    elif s.lower().replace(' ', '') == 'false':
      vclaim_labels.append([0, 1, 0, 0, 0, 0])
    elif s.lower().replace(' ', '') == 'pantsonfire':
      vclaim_labels.append([1, 0, 0, 0, 0, 0])
    else:
      vclaim_labels.append([0, 0, 0, 0, 0, 0])
  vclaim_labels = np.array(vclaim_labels).reshape((1, len(verifiedClaims), -1))
  scores = []
  for i in range(len(transcript)):
    scores.append(vclaim_labels)
  scores = np.concatenate(scores, axis=0)
  print('hot label shape:', scores.shape)
  return scores

def get_label_multi_hot_input_features(transcript_name, elasticsearch_transcript, verifiedClaims):
  transcript = dataset.transcripts[transcript_name]
  vclaim_labels = []
  for i, row in verifiedClaims.iterrows():
    s = re.sub(r'[^a-zA-Z ]+', '', row.truth_meter)
    if s.lower().replace(' ', '') == 'true':
      vclaim_labels.append([1, 1, 1, 1, 1, 1])
    elif s.lower().replace(' ', '') == 'mostlytrue':
      vclaim_labels.append([1, 1, 1, 1, 1, 0])
    elif s.lower().replace(' ', '') == 'halftrue':
      vclaim_labels.append([1, 1, 1, 1, 0, 0])
    elif s.lower().replace(' ', '') == 'mostlyfalse':
      vclaim_labels.append([1, 1, 1, 0, 0, 0])
    elif s.lower().replace(' ', '') == 'false':
      vclaim_labels.append([1, 1, 0, 0, 0, 0])
    elif s.lower().replace(' ', '') == 'pantsonfire':
      vclaim_labels.append([1, 0, 0, 0, 0, 0])
    else:
      vclaim_labels.append([0, 0, 0, 0, 0, 0])
  vclaim_labels = np.array(vclaim_labels).reshape((1, len(verifiedClaims), -1))
  scores = []
  for i in range(len(transcript)):
    scores.append(vclaim_labels)
  scores = np.concatenate(scores, axis=0)
  print('multi-hot label shape:', scores.shape)
  return scores

def get_label_halftrue_input_features(transcript_name, elasticsearch_transcript, verifiedClaims):
  transcript = dataset.transcripts[transcript_name]
  vclaim_labels = []
  for i, row in verifiedClaims.iterrows():
    s = re.sub(r'[^a-zA-Z ]+', '', row.truth_meter)
    if s.lower().replace(' ', '') == 'halftrue':
      vclaim_labels.append(1)
    else:
      vclaim_labels.append(0)
  vclaim_labels = np.array(vclaim_labels).reshape((1, -1))
  scores = []
  for i in range(len(transcript)):
    scores.append(vclaim_labels)
  scores = np.concatenate(scores, axis=0)
  print('label shape:', scores.shape)
  return np.expand_dims(scores, axis=2)


input = []
labels = []
vclaim_labels = []
elasticsearch_idxs = []
if not load:
  for i, transcript_name in enumerate(TRANSCRIPTS):
    transcript = dataset.transcripts[transcript_name]
    man_annotation = pd.read_csv(os.path.join(man_dir, transcript_name+'.tsv'), 
                   sep='\t')
    elasticsearch_transcript = np.load(os.path.join(elasticsearch_dir, transcript_name+'.npz'), allow_pickle=True)
    vclaim_selected_indxs = get_vclaim_indxs_from_elasticsearch(elasticsearch_transcript)
    # svm_preds = load_predictions(f'{ranksvm_dir}_{i}/transcript.qid.predict')
    X = []
    if 'bert-scores' in sys.argv:
      # add bert scores
      X.append(get_bert_scores(transcript_name, elasticsearch_transcript, verifiedClaims))

    if 'nli-scores' in sys.argv:
      # add nli scores
      X.append(get_nli_scores(transcript_name, elasticsearch_transcript, verifiedClaims))

    if 'cimsce' in sys.argv:
      # add cimsce embeddings
      X.append(get_cimsce_embeddings(transcript_name, elasticsearch_transcript, verifiedClaims))

    if 'cimsce-text' in sys.argv:
      # add cimsce vclaim and vclaim titile embeddings
      X.append(get_cimsce_text_embeddings(transcript_name, elasticsearch_transcript, verifiedClaims))

    if 'sbert' in sys.argv:
      # add sbert vclaim and vclaim titile embeddings
      X.append(get_sbert_embeddings(transcript_name, elasticsearch_transcript, verifiedClaims))

    if 'sbert-text' in sys.argv:
      # add sbert vclaim and vclaim titile embeddings
      X.append(get_sbert_text_embeddings(transcript_name, elasticsearch_transcript, verifiedClaims))

    if 'elasticsearch' in sys.argv:
      # add sbert vclaim and vclaim titile embeddings
      X.append(get_elsaticserch(transcript_name, elasticsearch_transcript, verifiedClaims))

    if 'label' in sys.argv:
      # add sbert vclaim and vclaim titile embeddings
      X.append(get_label_input_features(transcript_name, elasticsearch_transcript, verifiedClaims))

    if 'labelhot' in sys.argv:
      # add sbert vclaim and vclaim titile embeddings
      X.append(get_label_hot_input_features(transcript_name, elasticsearch_transcript, verifiedClaims))

    if 'labelmultihot' in sys.argv:
      # add sbert vclaim and vclaim titile embeddings
      X.append(get_label_multi_hot_input_features(transcript_name, elasticsearch_transcript, verifiedClaims))

    if 'labelhalftrue' in sys.argv:
      # add sbert vclaim and vclaim titile embeddings
      X.append(get_label_halftrue_input_features(transcript_name, elasticsearch_transcript, verifiedClaims))
    # concatenating inputs at the end
    X = np.concatenate(X, axis=2)
    print('End input shape:', X.shape)
    input.append(X)
    labels.append(get_labels(transcript, man_annotation))
    elasticsearch_idxs.append(vclaim_selected_indxs)

  print('dump')
  with open(f'max-input-labelhalftrue', 'wb') as f:
    pickle.dump(input, f)
  print('dump')
  with open(f'max-labels-labelhalftrue', 'wb') as f:
    pickle.dump(labels, f)
  print('dump')
  with open(f'max-elastlicsearch_idxs-labelhalftrue', 'wb') as f:
    pickle.dump(elasticsearch_idxs, f)
else:
  print('load')
  with open(f'max-input-labelhalftrue', 'rb') as f:
    input = pickle.load(f)
  # print('load')
  # with open(f'max-labels-labelhalftrue', 'rb') as f:
  #   labels = pickle.load(f)
  print('load')
  with open(f'max-elastlicsearch_idxs-labelhalftrue', 'rb') as f:
    elasticsearch_idxs = pickle.load(f)

labels = []
for transcript_name in TRANSCRIPTS:
  transcript = dataset.transcripts[transcript_name]
  l = np.zeros((len(transcript), len(verifiedClaims)))
  print(l.shape)
  for i in range(len(transcript)):
    kk = transcript.iloc[i].vclaims
    kk2 = transcript.iloc[i].vclaims_man
    l[i][kk] = 1
    l[i][kk2] = 1
  print(l.shape)
  vclaim_labels.append(l)
  man_annotation = pd.read_csv(os.path.join(man_dir, transcript_name+'.tsv'), 
                   sep='\t')
  labels.append(np.array(get_labels(transcript, man_annotation)))
  print('labels', labels[-1].shape, sum(labels[-1]))


def get_input(scores, idxs, top_n=1):
  print(scores.shape, idxs.shape)
  ss = []
  for s, idx in zip(scores, idxs):
    ss.append(s[idx[:top_n]].reshape((1, -1)))
  ss = np.concatenate(ss, axis=0)
  return ss, scores[:, :, 18]

inputs_final = []
vclaim_scores_final = []
for (input, idxs) in tqdm(zip(input, elasticsearch_idxs), total=len(input)):
  input, vclaim_scores = get_input(input, idxs, top_n=top_n)
  inputs_final.append(input)
  vclaim_scores_final.append(vclaim_scores)

splits = []
for i in range(len(inputs_final)):
  input_test = [inputs_final[i]]
  labels_test = [labels[i]]
  vclaim_labels_test = [vclaim_labels[i]]
  print(0, labels_test[0].shape)
  vclaim_elasticsearch_test = [elasticsearch_idxs[i]]
  vclaim_scores_test = [vclaim_scores_final[i]]
  input_train = inputs_final[:i] + inputs_final[i+1:]
  labels_train = labels[:i] + labels[i+1:]
  vclaim_labels_train = vclaim_labels[:i] + vclaim_labels[i+1:]
  vclaim_elasticsearch_train = elasticsearch_idxs[:i] + elasticsearch_idxs[i+1:]
  vclaim_scores_train = vclaim_scores_final[:i] + vclaim_scores_final[i+1:]
  splits.append((zip(input_train, labels_train, vclaim_labels_train, vclaim_elasticsearch_train, vclaim_scores_train), zip(input_test, labels_test, vclaim_labels_test, vclaim_elasticsearch_test, vclaim_scores_test)))

out = ''

out_MAP = []
out_MAP_0 = [[] for _ in range(len(threshs_MAPs))]
out_MAP_0_5 = [[] for _ in range(len(threshs_MAPs))]
out_MAP_harsh = [[] for _ in range(len(threshs_MAPs))]
out_MAP_inner = []
ranksvm_outer_dir = f'{sys.argv[-2]}/ranksvm-top-{top_n}'

for i, (train, test) in tqdm(enumerate(splits), total=len(splits)):

  for (input, label, vclaim_label, idxs, vclaim_scores) in test:
    # input, vclaim_scores = get_input(input, idxs, top_n=top_n)
    input = input.reshape((len(label), -1))
    qid = np.zeros((len(label), 1))
    print(11, label.sum(), len(label), label.shape, input.shape)
    predictions_fp = os.path.join(ranksvm_outer_dir, f'test-{top_n}-{i+1}.qid.predict')
    print('loading from', predictions_fp)
    predictions = load_predictions(predictions_fp)
    print(predictions.shape, label.shape)
    MAP, MAP_0, MAP_0_5, MAP_harsh, MAP_inner  = evaluate(predictions, label, vclaim_scores, vclaim_label)
    out_MAP.append(MAP)
    out_MAP_inner.append(MAP_inner)
    out += f'{MAP}\t'
    out += f'{MAP_inner}\t'
    for map_t, out_map_t in zip((MAP_0, MAP_0_5, MAP_harsh), (out_MAP_0, out_MAP_0_5, out_MAP_harsh)):
      for ii in range(len(threshs_MAPs)):
        out += f'{map_t[ii]}\t'
        out_map_t[ii].append(map_t[ii])
    out += '\n'
out += f'{np.mean(out_MAP)}\t'
out += f'{np.mean(out_MAP_inner)}\t'
for out_map_t in (out_MAP_0, out_MAP_0_5, out_MAP_harsh):
  for ii in range(len(threshs_MAPs)):
    out += f'{np.mean(out_map_t[ii])}\t'
out += '\n'
print('='*50)
print(out)
print('='*50)

with open('results-final.tsv', 'a') as f:
  f.write(f'ranksvm-{top_n}-{sys.argv[-2]}\n')
  f.write(out)
