import nltk
from gensim.summarization import summarize
from rouge_score import rouge_scorer
import collections
import math

import rouge_score

nltk.download("wordnet")
nltk.download("stopwords")
from nltk.corpus import wordnet, stopwords
from nltk.stem.porter import PorterStemmer

def lem_and_stem(word, stemmer = PorterStemmer()):

    lemma = wordnet.morphy(word)
    lemma = lemma if lemma else word

    root = stemmer.stem(lemma)

    return root if root else lemma

def word_frequencies(bag: list):

    word_map = collections.defaultdict(float)

    for word in bag:

        word_root = lem_and_stem(word)
        
        word_map[word_root] += 1

    return word_map

def get_tf_matrix(sentences: list):

    tf_matrix = {}

    for sentence in sentences:
        tokens = [word for word in sentence.split() if word not in stopwords.words("english")]
        tf_table = word_frequencies(tokens)

        for word in tf_table:
            tf_table[word] /= len(tokens)

        tf_matrix[sentence] = tf_table

    return tf_matrix

def get_idf_freqs(tf_matrix):

    idf_freqs = collections.defaultdict(float)

    for sentence, freqs in tf_matrix.items():

        for term in freqs:

            idf_freqs[term] += 1

    return idf_freqs

def get_idf_matrix(tf_matrix, docs_per_term, total_docs):

    idf_matrix = {}

    for sent, freqs in tf_matrix.items():

        idf_table = {}

        for term in freqs:

            idf_table[term] = math.log10(total_docs / docs_per_term[term])

        idf_matrix[sent] = idf_table

    return idf_matrix

def tf_idf_matrix(tf_matrix, idf_matrix):

    tf_idf = {}

    for (sent1, f1), (sent2, f2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (term1, val1), (term2, val2) in zip(f1.items(), f2.items()):
            tf_idf_table[term1] = val1 * val2

        tf_idf[sent1] = tf_idf_table

    return tf_idf

def sentence_score(tf_idf):

    score = {}

    for sent, freqs in tf_idf.items():

        total_score = 0

        sent_len = len(freqs)

        for term, term_score in freqs.items():
            total_score += term_score

        score[sent] = total_score / sent_len

    return score

def avg_sent_score(sent_val):

    avg = sum([sent_val[key] for key in sent_val]) / len(sent_val)

    return avg

def topk_sents(sents, k: int = 5):

    ranked = sorted(sents, key = sents.get, reverse = True)

    return [s for s in ranked[:k]]


def tfidf_summarize(target_sents):

    tf_matrix = get_tf_matrix(target_sents)
        
    docs_per_term = get_idf_freqs(tf_matrix)
    idf_matrix = get_idf_matrix(tf_matrix, docs_per_term, total_docs)

    tf_idf = tf_idf_matrix(tf_matrix, idf_matrix)

    sent_scores = sentence_score(tf_idf)
    ranked_sents = topk_sents(sent_scores, k = int(0.05 * total_docs))

    # generate summary using these top-k sentences
    article_summary = '. '.join(ranked_sents)
    summaries.append(article_summary)

    return article_summary

if __name__ == "__main__":

    targets = []

    with open("targets.txt", 'r') as file:
        targets = [int(target) for target in file.readline().split()]

    overall_text = []
    overall_sents = []

    summaries = []
    scores = []

    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer = True)

    for i, target in enumerate(targets):

        with open("../sentences/{}.sentences".format(target), 'r') as file:
            target_sents = file.readlines()

        target_sents = [sent.strip() for sent in target_sents if len(sent.split()) > 10]
        total_docs = len(target_sents)

        #overall_sents.append(target_sents)

        print("total sentences: ", total_docs)

        article_summary = tfidf_summarize(target_sents)

        # print(article_summary)
        # print('\n')
        summaries.append(article_summary)

        # for sent in ranked_sents:
        #     print("{} | [ {} ]".format(sent, sent_scores[sent]))

        overall_sents.extend(target_sents)
        article_text = '. '.join(target_sents)
        overall_text.append(article_text)

        tr_summary = summarize(article_text)

        score = rouge.score(article_summary, tr_summary)
        scores.append(score)

    overall_text = '. '.join(overall_text)
    tr_overall_summary = summarize(overall_text)
    tfidf_overall_summary = tfidf_summarize(overall_sents)
    overall_score = rouge.score(tfidf_overall_summary, tr_overall_summary)
    
    print()

    for i, score in enumerate(scores):
        print("Scores for document ", i)
        print("R-1 Precision: {:.3f} | R-L Precision: {:.3f}".format(score["rouge1"].precision, score["rougeL"].precision))
        print("R-1 Recall:    {:.3f} | R-L Recall:    {:.3f}".format(score["rouge1"].recall, score["rougeL"].recall))
        print()

    print("Overall scores: ")
    print("R-1 Precision: {:.3f} | R-L Precision: {:.3f}".format(overall_score["rouge1"].precision, overall_score["rougeL"].precision))
    print("R-1 Recall:    {:.3f} | R-L Recall:    {:.3f}".format(overall_score["rouge1"].recall, overall_score["rougeL"].recall))