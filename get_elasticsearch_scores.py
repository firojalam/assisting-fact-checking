import argparse
import pdb
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from elasticsearch import Elasticsearch

from politifactDataloader import Dataset
from logger import logger

TRANSCRIPTS = ['20170803_Trump_WV', '20180612_Trump_Singapore', 
'20170822_Trump_phoenix', '20180615_Trump_lawn', 
'20180426_Trump_Fox_Friends', '20180628_Trump_NorthDakota', 
'20180525_Trump_Naval']


'''
  Load the verified claims into the elasticsearch server
'''
def build_dataset(es, verifiedClaims, index='politifact'):
  # Create a json object of the verified claims
  verifiedClaims = json.loads(verifiedClaims.to_json(orient='records'))

  try:
    es.indices.delete(index=index)
  except:
    pass

  # Add all verified claims to the 
  for i, verifiedClaim in enumerate(tqdm(verifiedClaims)):
    if not es.exists(index=index, id=verifiedClaim['vclaim_id']):
      es.create(index=index, id=verifiedClaim['vclaim_id'], body=verifiedClaim)


'''
  Get elasticsearch/BM25 scores of a claim agaisnt verified claims. 
'''
def get_score(es, claim, search_key='all', size=10000, search_index='politifact'):
  verifiedClaims_len = 16636 #es.indices.stats()['indices'][search_index]['total']['docs']['count']
  query_score = np.zeros(verifiedClaims_len)
  indices = np.zeros(size)
  try:
    if search_key == 'all':
      response = es.search(index=search_index, q=claim, size=size)
    else:
      response = es.search(index=search_index, body={
                 "query": {"match": {search_key: claim}}}, size=size)
    # pdb.set_trace()
    results = response['hits']['hits']
    df = pd.DataFrame(results)
    indices = df._id.astype('int32').values
    query_score[indices] = df._score
  except:
    logger.error("No elasticsearch results for (%s)" % (claim))
    pass
  return query_score, indices

'''
  Get elasticsearch/BM25 scores of the claims agaisnt the verifiedclaims.
''' 
def get_scores(es, claims, verifiedClaims,  
    search_key='all', size=10000, search_index='politifact', vclaims_pair=[]):
  scores = np.zeros((len(claims), len(verifiedClaims)))
  indices = np.zeros((len(claims), size))
  logger.info("Get RM5 scores for (%d) claims" % (len(claims)))
  ss = 0
  for i, (claim) in enumerate(tqdm(claims)):
    if len(vclaims_pair) != 0 and len(vclaims_pair[i]) == 0:
      continue
    # print(i)
    ss += 1
    score, index = get_score(es, claim, search_key=search_key, size=size, search_index=search_index)
    # if sum(score) == 0:
    #   logger.error(f"{i}:{claim}\n{vclaims_pair[i]}")
    scores[i] = score
    indices[i, :len(index)] = index
  logger.info(f"total done {ss}")
  return scores, indices

def create_connection():
  logger.debug("Start ElasticSearch listener")
  es = Elasticsearch()
  logger.debug("Elasticsearch connected")
  return es


def run(args):
  if not os.path.exists(args.data_root):
    logger.error("Data directory (%s) doesnt exist"%args.data_root)
    exit()
  
  if not os.path.exists(args.out_dir):
    logger.warning("Output directory (%s) doesnt exist"%args.out_dir)
    os.makedirs(args.out_dir)

  dataset = Dataset(args.data_root, args.coref_ext, concat_before=args.concat_before, 
    concat_after=args.concat_after)

  # Create a socket for elasticsearch
  es = create_connection()

  # Load the verified claims into the elasticsearch server
  verifiedClaims = dataset.verified_claims
  if args.load:
    logger.info("Load verifiedClaims into elasticsearch")
    build_dataset(es, verifiedClaims, index=args.index)

  search_keys = ['all', 'vclaim', 'title', 'text']
  transcript_names = list(dataset.transcripts.keys())
  transcript_names.sort()
  for transcript_name in TRANSCRIPTS[args.lower:args.upper]:
    logger.info(f"Obtain es score for {transcript_name}")
    if os.path.exists(os.path.join(args.out_dir, '%s'%(transcript_name))):
      logger.warning(f"Skipping transcript ({transcript_name}) as the scores already exists.")
      continue
    transcript = dataset.transcripts[transcript_name]
    vclaims_pair_ = []
    if args.positives:
      vclaims_pair_ = [ a+b for (a, b) in zip(transcript.vclaims.tolist(), transcript.vclaims_man.tolist())]
    vclaims_pair = vclaims_pair_[:]
    print(len(vclaims_pair), len(vclaims_pair_), len(transcript))
    # for i in range(len(vclaims_pair_)):
    #   if len(vclaims_pair_[i]):
    #     vclaims_pair[i-3:i+2] = [vclaims_pair_[i]] * len(vclaims_pair[i-3:i+2])
    print(len(vclaims_pair), len(vclaims_pair_), len(transcript))
    data = {}
    for search_key in search_keys:
      scores = np.zeros((2, len(transcript), args.nscores))
      scores_, indices = get_scores(es, transcript.sentence, verifiedClaims, 
        search_key=search_key, size=args.nscores, search_index=args.index, vclaims_pair=vclaims_pair)
      for sent_id, score in enumerate(scores_):
        order = np.argsort(score[np.where(score)])[::-1]
        # print(indices[sent_id], np.where(score)[0][order])
        # print(np.all(indices[sent_id] == np.where(score)[0][order]))
        scores[0, sent_id, :len(indices[sent_id])] = indices[sent_id].astype('int')
        scores[1, sent_id, :len(indices[sent_id])] = score[indices[sent_id].astype('int')]
      data[search_key] = scores
    np.savez(os.path.join(args.out_dir, '%s'%(transcript_name)), 
            all=data['all'], 
            vclaim=data['vclaim'], 
            title=data['title'],
            text=data['text']
            )

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get elasticsearch scores')
  parser.add_argument('--data-root', '-d', required=True, 
    help='Path to the dataset directory.')
  parser.add_argument('--out-dir', '-o', required=True, 
    help='Path to the output directory')
  parser.add_argument('--nscores', '-n', default=10, type=int,
    help='Number of retrieved verified claims scores returned per sentence.')
  parser.add_argument('--index', default='politifact',
    help='index used to put data in elasticsearch')
  parser.add_argument('--load', default=False, action='store_true',
    help='Reload the data into politifact if this flag is set.')
  parser.add_argument('--positives', default=False, action='store_true',
    help='If this flag is set only the sentences with a verified claim would be scored from the transcript')
  parser.add_argument('--lower', default=0, type=int,
    help='')
  parser.add_argument('--upper', default=70, type=int,
    help='')
  parser.add_argument('--concat-before', default=0, type=int,
    help='Number of sentences concatenated before the input sentence in a transcript')
  parser.add_argument('--concat-after', default=0, type=int,
    help='Number of sentences concatenated after the input sentence in a transcript')
  parser.add_argument('--coref-ext', default='', type=str,
    help='If using co-referenced resolved transcripts it gives the resolved coreference data. ({data}/transcripts{EXT} provide EXT)')

  args = parser.parse_args()
  run(args)
