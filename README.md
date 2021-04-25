# Automatic Multi-Doc Summarization using LSI

## Requirements

* Python3 (3.6+ preferred)
* Anaconda3 (optional, but helpful for setup)

## Setup

Run the setup script provided (```cd scripts/ && ./setup.sh```) or follow the steps below to create the conda environment with dependencies installed, install auxiliary resources for dependencies, download a minimal dataset, and perform preprocessing.

```bash
# from root dir:

# create environment from file and activate it
# if you run into issues, using environment_verbose.yml may help for finding missing packages
conda env create --file environment.yml

conda activate autosumm

# download a small set of research papers
# additional command line args available; to view, run as: python arxiv_fetch.py -h
# skip this step if you'd like to use the provided data
python arxiv_fetch.py

# download and install spaCy language model
python -m spacy download en_core_web_sm

# preprocess the downloaded documents
# skip this step if you'd like to use the provided data
cd summarize/ && python preprocess.py
```

**NOTE:** You may skip most of the above steps if you only seek to run experiments on the data I used (the documents in `sentences/`). You will only need to create the `conda` environment and download the `spaCy` language model as shown above, then skip to the instructions to run experiments below.

**NOTE:** in these experiments, a set of documents with topic tag `"text summarization"` are downloaded from arXiv, for LSI training. Three documents (IDs 386, 252, and 113 in this `sentences/`) are used for testing summary generation. You can tweak the number of papers you'd like to download and the general topic \[tag\] of the papers by manually running `arxiv_fetch.py` in `scripts/` with the optional flags (run `python arxiv_fetch.py -h` to view).

## Running Experiments

Specify the documents you'd like to summarize by their document ID in `summarize/targets.txt` (see example in that file). Document IDs can be found in `summarize/ids.txt`, which is generated after running the arXiv downloading script.

Run the entire pipeline:

```bash
# from root dir
cd summarize/

# run the LSI model experiments
# default: extracts 200 latent topics from corpus, summarizes only the documents specified in targets.txt
python pipeline_lsi.py

# run the TF-IDF summarizer experiments
python pipeline_tfsumm.py
```

### Pipeline Overview

Below is a diagram of the complete summarization pipeline. See `summarize/pipeline_lsi.py` for implementation.

![overview of the summarization pipeline](assets/lsi_tfidf_pipeline.png "Overview of the summarization pipeline")

## Preliminary Results

Here are the results from a sample run the LSI method (`Overall` denotes performance of the generated multi-document summary, and `Average` is the average over the single-document summaries):

|     Doc     | ROUGE-1 Precision | ROUGE-L Precision | ROUGE-1 Recall | ROUGE-L Recall |
|:-----------:|:-----------------:|:-----------------:|:--------------:|:--------------:|
|  0 (target) |       0.581       |       0.346       |      0.474     |      0.282     |
|  1 (target) |       0.521       |       0.192       |      0.364     |      0.134     |
|  2 (target) |       0.602       |       0.305       |      0.485     |      0.246     |
|   Average   |       0.568       |       0.281       |      0.441     |      0.221     |
|   Overall   |       0.905       |       0.631       |      0.035     |      0.024     |

And the results of the TF-IDF summarizer on the same documents:

|     Doc     | ROUGE-1 Precision | ROUGE-L Precision | ROUGE-1 Recall | ROUGE-L Recall |
|:-----------:|:-----------------:|:-----------------:|:--------------:|:--------------:|
|  0 (target) |       0.226       |       0.124       |      0.462     |      0.255     |
|  1 (target) |       0.234       |       0.119       |      0.428     |      0.217     |
|  2 (target) |       0.195       |       0.103       |      0.419     |      0.221     |
|   Average   |       0.218       |       0.115       |      0.436     |      0.231     |
|   Overall   |       0.583       |       0.345       |      0.102     |      0.060     |
