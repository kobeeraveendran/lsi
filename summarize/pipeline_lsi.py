import nltk
from gensim import corpora
from gensim.models import LsiModel, coherencemodel, lsimodel
from gensim.models import CoherenceModel

# nltk.download("wordnet")
# nltk.download("stopwords")
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

def lsi(corpus_dict, doc_term_matrix, topics):

    #corpus_dict, doc_term_matrix = term_freqs(docs)

    lsi_model = LsiModel(doc_term_matrix, num_topics = topics, id2word = corpus_dict)
    #print(lsi_model.print_topics(num_topics = topics, num_words = words))
    return lsi_model

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

def take_next(item):
    return item[1]

def vec_sort(corpus_lsi, num_topics):

    vecsSort = list(map(lambda i: list(), range(num_topics)))
    for i,docv in enumerate(corpus_lsi):
        for sc in docv:
            isent = (i, abs(sc[1]))
            vecsSort[sc[0]].append(isent)
    vecsSort = list(map(lambda x: sorted(x,key= take_next, reverse=True), vecsSort))

    return vecsSort

    # vecs = list(map(lambda i: list(), range(2)))

    # for i, doc_vec in enumerate(corpus_lsi):
    #     for scalar in doc_vec:
    #         vecs[scalar[0]].append((i, abs(scalar[1])))

    # vecs = list(map(lambda x: sorted(x, key = lambda y: y[1], reverse = True), vecs))

    # return vecs

def sent_rank(summary_size, topics, sorted_vecs):

    # top_sents = []
    # sent_nums = []
    # sent_indices = set()

    # sent_count = 0

    # for i in range(summary_size):
    #     for j in range(topics):

    #         sent_index = vecs[j][i][0]
    #         if sent_index not in sent_indices:
    #             top_sents.append(vecs[j][i])
    #             sent_nums.append(sent_index)
    #             sent_indices.add(sent_index)

    #             if sent_count == summary_size:
    #                 break

    # return sent_nums, top_sents

    topSentences = []
    sent_no = []
    sentInd = set()
    sCount = 0

    #print(len(sorted_vecs))
    #print(len(sorted_vecs[0]))
    
    for i in range(summary_size):
        for j in range(topics):
            vecs = sorted_vecs[j]

            if len(vecs) <= i:
                continue

            #print("vecs length: ", len(vecs))
            #print('i: ', i)
            si = vecs[i][0]
            #print("si: ", si)
            if si not in sentInd:
                sent_no.append(si)
                topSentences.append(vecs[i])
                sentInd.add(si)
                sCount +=1
                if sCount == summary_size:
                    break

    return sent_no, topSentences


if __name__ == "__main__":

    targets = []

    with open("targets.txt", 'r') as file:
        targets = [int(target) for target in file.readline().split()]

    for i, target in enumerate(targets):

        with open("../sentences/{}.sentences".format(target), 'r') as file:
            target_sents = file.readlines()

        target_sents = [sent.strip() for sent in target_sents if len(sent.split()) > 10]
        total_docs = len(target_sents)
        print("\n\ntotal docs: ", total_docs)

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

        corpus_dict, doc_term_matrix = term_freqs(prep_sents)

        lsi_model = lsi(corpus_dict, doc_term_matrix, topics = 200)
        lsi_corpus = lsi_model[doc_term_matrix]

        vecs = vec_sort(lsi_corpus, num_topics = 200)

        sent_nums, top_sents = sent_rank(8, topics = 200, sorted_vecs = vecs)

        #print("sent nums: ", sent_nums)
        #print("top sents: ", top_sents)

        summary = []
        index = 0

        topk_sents = set(sent_nums[:8])

        cleaned_target_sents = []

        for i, sentence in enumerate(prep_sents):
            cleaned_target_sents.append(' '.join(sentence))

        for i, sentence in enumerate(cleaned_target_sents):

            if i in topk_sents:
                summary.append(sentence)

        # print("original: \n")
        # print(' '.join(target_sents))

        print("summary: \n")
        print('. '.join(summary))