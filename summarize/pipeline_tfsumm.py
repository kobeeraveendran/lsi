import nltk
import collections
import math

nltk.download("wordnet")
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer

def lem_and_stem(word, stemmer = PorterStemmer()):

    lemma = wordnet.morphy(word)
    lemma = lemma if lemma else word

    root = stemmer.stem(lemma)

    return root if root else lemma

def word_frequencies(bag: list):

    word_map = collections.defaultdict(float)

    for word in bag:

        word = lem_and_stem(word)
        word_map[word] += 1

    return word_map

def get_tf_matrix(sentences: list):

    tf_matrix = {}

    for sentence in sentences:
        tokens = sentence.split()
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

    ranked = sorted(sents, key = sents.get)

    return ranked[-k:]

if __name__ == "__main__":

    pass
