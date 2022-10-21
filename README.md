# Assisting the Human Fact-Checkers: Detecting All Previously Fact-Checked Claims in a Document

To train the reranker we need to get several things:
  1. Elasticsearch scores between the input-claims (iclaims) and the verified-claims (vclaims) dataset. 
  2. SBERT embeddings of the vclaims and their article.
  3. SBERT embeddings of the iclaims.

After getting them we create the feature vectors for the ranksvm to have a better rerank system. 

__Table of Contents:__
- [Dataset](data/)
- [Fact-Checking-and-Verification-in-Debates](#fact-checking-and-verification-in-debates)
  - [Elasticsearch Scores](#elasticsearch-scores)
  - [SBERT Embeddings](#sbert-embeddings)
  - [Features/Scores for the rankSVM](#featuresscores-for-the-ranksvm)
  - [Training the rankSVM](#training-the-ranksvm)
- [Publication](#publication)
- [Credits](#credits)
- [Licensing](#licensing)

## Dataset
More details of the dataset can be found in [data directory](data/).


## Elasticsearch scores
To run elasticsearch you need to first runt he elasticsearch server before running the experiment script. 
```
elasticsearch -d # The flag d is to run the elasticsearch server in the background as a demeon process
```

To install elasticsearch, check this [link](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).


After having the elasticsearch server running, to run the experiments you should run
```
data_dir="../data/politifact" #This is the directory with that contains the dataset. 
elasticsearch_score_dir="$data_dir/elasticsearch.scores.100" #THe directory where the data will be saved

python get_elasticsearch_scores.py \
		-d $data_dir \
		-o $elasticsearch_score_dir \
		-n 100 \ 
		--index politifact \
		--coref-ext $coref_ext \
    --load \ # Run thisflag only once to load the data in the elasticsearch server
		--lower 0 --upper 70 --positives 
```

Usage of the running script
```
usage: get_elasticsearch_scores.py [-h] --data-root DATA_ROOT --out-dir
                                   OUT_DIR [--nscores NSCORES] [--index INDEX]
                                   [--load] [--positives] [--lower LOWER]
                                   [--upper UPPER]
                                   [--concat-before CONCAT_BEFORE]
                                   [--concat-after CONCAT_AFTER]
                                   [--coref-ext COREF_EXT]

Get elasticsearch scores

optional arguments:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT, -d DATA_ROOT
                        Path to the dataset directory.
  --out-dir OUT_DIR, -o OUT_DIR
                        Path to the output directory
  --nscores NSCORES, -n NSCORES
                        Number of retrieved vclaims scores returned
                        per iclaim.
  --index INDEX         index used to put data in elasticsearch
  --load                Reload the data into elasticsearch if this flag is set.
  --positives           If this flag is set only the sentences with a vclaims
                        would be scored from the transcript
  --lower LOWER         To run the code over batches, the code would run the 
                        trasncripts[lower:upper]
  --upper UPPER         To run the code over batches, the code would run the 
                        trasncripts[lower:upper]
  --concat-before CONCAT_BEFORE
                        Number of sentences concatenated before the input
                        sentence in a transcript
  --concat-after CONCAT_AFTER
                        Number of sentences concatenated after the input
                        sentence in a transcript
  --coref-ext COREF_EXT
                        If using co-referenced resolved transcripts it gives
                        the resolved coreference data.
                        ({data}/transcripts{COREF_EXT})
```

## SBERT Embeddings
The second step is to get the sentenceBERT (SBERT) embeddings of the following:
  1. vclaim (vclaim)
  2. vclaim-article (title)
  3. vclaim-artcile-title (text)
  4. iclaims (transcript)
  
To run the experiment, 
```
data_dir="../data/politifact"
bert_embedding_dir="$data_dir/SBERT.large.embeddings"

sbert_config="config/bert-specs.json"
python get_bert_embeddings.py \
		-d $data_dir \
		-o $bert_embedding_dir \
		-c $sbert_config \
		--coref-ext $coref_ext \
		-i transcript vclaim title text

```

Usage of the script, 
```
usage: get_bert_embeddings.py [-h] --data-root DATA_ROOT --out-dir OUT_DIR
                              --config CONFIG
                              [--input {vclaim,title,text,transcript} [{vclaim,title,text,transcript} ...]]
                              [--coref-ext COREF_EXT]
                              [--concat-before CONCAT_BEFORE]
                              [--concat-after CONCAT_AFTER]

Get elasticsearch scores

optional arguments:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT, -d DATA_ROOT
                        Path to the dataset directory.
  --out-dir OUT_DIR, -o OUT_DIR
                        Path to the output directory
  --config CONFIG, -c CONFIG
                        Path to the config file
  --input {vclaim,title,text,transcript} [{vclaim,title,text,transcript} ...], -i {vclaim,title,text,transcript} [{vclaim,title,text,transcript} ...]
                        What dataentry you want to get the SBERT embeddings for. 
  --coref-ext COREF_EXT
                        If using co-referenced resolved transcripts it gives
                        the resolved coreference data.
                        ({data}/transcripts{EXT} provide EXT)
  --concat-before CONCAT_BEFORE
                        Number of sentences concatenated before the input
                        sentence in a transcript
  --concat-after CONCAT_AFTER
                        Number of sentences concatenated after the input
                        sentence in a transcript
```

## Bert score between iclaim and vclaim
Before computing the bert score you need to install the tool by running the following:

```
pip install bert_score
```

To compute the bert score for the iclaims based on the vclaims, you need to run the following scripT:
```
data_dir="../data/politifact"
output_dir="../data/politifact/bert_scores"
python get_bert_score_rankings.py \
    -d $data_dir \
    -o $output_dir \
    --rescale-with-baseline True \
    --verbose True
```
Usage of the script,
```
usage: get_bert_score_rankings.py [-h] --data-root DATA_ROOT --output OUTPUT_DIR
                                  [--rescale-with-baseline] [--verbose]
--data-root DATA_ROOT       Root folder where the dataset is located
--output OUTPUT_DIR         Folder where the scores are stored
--rescale-with-baseline     [TRUE|FALSE] Determines whether the scores should be rescaled or not. Rescaled scores
                            tend to be more human-readable
--verbose                   [TRUE|FALSE]Determines whether the scorer outputs will be shown.
```


## Publication:
Please cite the following paper.

*Shaden Shaar, Nikola Georgiev, Firoj Alam, Giovanni Da San Martino, Aisha Mohamed, Preslav Nakov, "Assisting the Human Fact-Checkers: Detecting All Previously Fact-Checked Claims in a Document", Findings of EMNLP 2022,  [download](https://arxiv.org/pdf/2109.07410.pdf).*


```bib

@inproceedings{assisting-fact-checkers:2022,
author = {Shaden Shaar, Nikola Georgiev, Firoj Alam, Giovanni Da San Martino, Aisha Mohamed, Preslav Nakov},
title = {Assisting the Human Fact-Checkers: Detecting All Previously Fact-Checked Claims in a Document},
booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2022},
series = {EMNLP~'22},
address = {Abu Dhabi, UAE},
year = {2022},
}
```

## Credits
* Shaden Shaar, Cornell University, United States
* Nikola Georgiev, Sofia University, Bulgaria
* Firoj Alam, Qatar Computing Research Institute, HBKU, Qatar
* Giovanni Da San Martino, University of Padova, Italy
* Aisha Mohamed, University of Wisconsin-Madison, United States
* Preslav Nakov, Mohamed bin Zayed University of Artificial Intelligence, UAE


## Licensing

This dataset is published under CC BY-NC-SA 4.0 license, which means everyone can use this dataset for non-commercial research purpose: https://creativecommons.org/licenses/by-nc/4.0/.