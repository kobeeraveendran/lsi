import nltk
from gensim import corpora
from gensim.models import LsiModel, coherencemodel, lsimodel
from gensim.models import CoherenceModel
from gensim.summarization import summarize
from nltk.corpus.reader.nombank import NombankPointer
from rouge_score import rouge_scorer

nltk.download("wordnet")
nltk.download("stopwords")
from nltk.corpus import wordnet, stopwords
from nltk.stem.porter import PorterStemmer


def lem_and_stem(word, stemmer = PorterStemmer()):

    lemma = wordnet.morphy()
    lemma = lemma if lemma else word

    root = stemmer.stem(lemma)

    return root if root else lemma

def term_freqs(docs):

    corpus_dict = corpora.Dictionary(docs)
    doc_term_matrix = [corpus_dict.doc2bow(doc) for doc in docs]

    return corpus_dict, doc_term_matrix

# some steps for topic extraction, and references for LSI model creation are adapted from the following sources:
# https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python
# https://github.com/yeedas/Abstractive_Summary_of_Transcriptions/blob/master/Summarization_using_Latent_Semantic_Analysis.ipynb
def lsi(corpus_dict, doc_term_matrix, topics):

    #corpus_dict, doc_term_matrix = term_freqs(docs)

    lsi_model = LsiModel(doc_term_matrix, num_topics = topics, id2word = corpus_dict)
    #print(lsi_model.print_topics(num_topics = topics, num_words = words))
    return lsi_model

# would have been desirable for better results, but takes too long to run for complex articles, which 
# can have hundreds of latent topics

# def optimal_lsi(corpus_dict, doc_term_matrix, sents, max_topics = 300):

#     coherence_vals = []
#     model_list = []

#     print("Calculating optimal number of topics...", end = '\r')

#     for n_topics in range(2, max_topics):
#         model = lsi(corpus_dict, doc_term_matrix, topics = n_topics)
#         model_list.append(model)

#         coherence_model = CoherenceModel(model = model, texts = sents, dictionary = corpus_dict, coherence = "c_v")
#         coherence_vals.append(coherence_model.get_coherence())

#         print("Calculating optimal number of topics... {}/{} tested".format(n_topics, max_topics), end = '\r')

#     print("Calculating optimal number of topics... Complete")

#     max_index = max(range(len(coherence_vals)), key = lambda x: coherence_vals[x])
#     best_model = model_list[max_index]
#     optimal_topics = max_index * 2 + 2

#     return best_model, optimal_topics

def vec_sort(corpus_lsi, num_topics):

    vecs = list(map(lambda i: list(), range(num_topics)))

    for i, doc_vec in enumerate(corpus_lsi):
        for sc in doc_vec:
            vecs[sc[0]].append((i, abs(sc[1])))

    vecs = list(map(lambda x: sorted(x, key = lambda y: y[1], reverse = True), vecs))

    return vecs

def sent_rank(summary_size, topics, sorted_vecs):

    top_sents = []
    sent_nums = []
    sent_indices = set()

    sent_count = 0

    
    for j in range(topics):

        vecs = sorted_vecs[j]
        sent_index = vecs[0][0]

        if sent_index not in sent_indices:
            top_sents.append(vecs[i])
            sent_nums.append(sent_index)
            sent_indices.add(sent_index)

            sent_count += 1
            if sent_count == summary_size:
                break

    print(summary_size, len(sent_nums))

    return sent_nums, top_sents


def lsi_summarize(prep_sents):

    corpus_dict, doc_term_matrix = term_freqs(prep_sents)

    lsi_model = lsi(corpus_dict, doc_term_matrix, topics = 200)
    lsi_corpus = lsi_model[doc_term_matrix]

    vecs = vec_sort(lsi_corpus, num_topics = 200)

    num_sents = int(0.05 * len(prep_sents))

    sent_nums, top_sents = sent_rank(num_sents, topics = 200, sorted_vecs = vecs)

    #print("sent nums: ", sent_nums)
    #print("top sents: ", top_sents)

    summary = []

    topk_sents = set(sent_nums[:num_sents])

    cleaned_target_sents = []

    for i, sentence in enumerate(prep_sents):
        cleaned_target_sents.append(' '.join(sentence))

    overall_sents.extend(prep_sents)
    cleaned_orig_text = '. '.join(cleaned_target_sents)

    summary = []

    for i, sentence in enumerate(cleaned_target_sents):

        if i in topk_sents:
            summary.append(sentence)

    # print("original: \n")
    # print(' '.join(target_sents))
    
    summary = '. '.join(summary)

    return summary, cleaned_target_sents, cleaned_orig_text

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

        print("Document", i)
        print("total docs: ", total_docs)

        #print(target_sents[:10])

        prep_sents = []
        for target in target_sents:
            curr_sent = []

            for word in target.split():
                if len(word) < 2 and word != 'a' and word != 'i' and word not in stopwords.words("english"):
                    continue
                curr_sent.append(word)

            if curr_sent:
                prep_sents.append(curr_sent)

        summary, _, cleaned_orig_text = lsi_summarize(prep_sents)
        overall_text.append(cleaned_orig_text)
        overall_sents.extend(prep_sents)

        print("Summary:\n")
        print(summary)
        print('\n')

        #print(summary)
        summaries.append(summary)

        tr_summary = summarize(cleaned_orig_text, ratio = 0.05)

        score = rouge.score(summary, tr_summary)
        scores.append(score)

    overall_text = '. '.join(overall_text)
    tr_overall_summary = summarize(overall_text, ratio = 0.005)
    lsi_overall_summary, _, _ = lsi_summarize(overall_sents)
    overall_score = rouge.score(lsi_overall_summary, tr_overall_summary)

    print("Overall summary:")
    print(lsi_overall_summary)
    print('\n')

    for i, score in enumerate(scores):
        print("Scores for document ", i)
        print("R-1 Precision: {:.3f} | R-L Precision: {:.3f}".format(score["rouge1"].precision, score["rougeL"].precision))
        print("R-1 Recall:    {:.3f} | R-L Recall:    {:.3f}".format(score["rouge1"].recall, score["rougeL"].recall))
        print()

    print("Overall scores: ")
    print("R-1 Precision: {:.3f} | R-L Precision: {:.3f}".format(overall_score["rouge1"].precision, overall_score["rougeL"].precision))
    print("R-1 Recall:    {:.3f} | R-L Recall:    {:.3f}".format(overall_score["rouge1"].recall, overall_score["rougeL"].recall))
