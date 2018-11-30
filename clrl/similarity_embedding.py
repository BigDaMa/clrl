'''
- Loads the word embedding
- Includes all word embedding similarity functions
'''

import numpy as np
import pandas as pd
import io
import pickle
import collections

from scipy.optimize import linear_sum_assignment
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words
from babylon.fasttext import FastVector
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from nltk.corpus import wordnet as wn
from SIF.src import data_io, SIF_embedding
from SIF.src import params


class Similarity_Embedding():

    def __init__(self):
        self.embedding = ""

        self.en_dictionary = {}
        self.de_dictionary = {}
        self.dim = 0

        # For tfidf
        self.en_word2weight = {}
        self.de_word2weight = {}
        # self.en_stop_words = get_stop_words('en')
        # self.de_stop_words = get_stop_words('de')

        # For sif
        self.en_words = {}
        self.en_We = []
        self.en_weight4ind = {}
        self.de_words = {}
        self.de_We = []
        self.de_weight4ind = {}
        self.parameters = params.params()
        self.parameters.rmpc = 1

        # For senses
        self.en_senses = {}
        self.de_senses = {}



    # Loads word embedding vec file to a dictionary type
    # Prepares necessary data for SIF feature
    def load_vec(self, emb_path):
        fin = io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        words = {}
        We = []
        for (i, line) in enumerate(fin):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            data[word] = vect

            # This is for sif
            words[word] = i
            We.append(vect)

        We = np.array(We)
        return data, words, We

    # Loads word embedding vec file to a dictionary type
    # Prepares necessary data for SIF feature
    def load_vec_multivecs(self, emb_path):
        fin = io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        data = {}
        words = {}
        We = []
        for (i, line) in enumerate(fin):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            data[word] = vect

            # This is for sif
            words[word] = i
            We.append(vect)

        We = np.array(We)
        return data, words, We

    # Preparation for SIF
    def prepare_sif(self, lang):

        path = "/home/oyku/myversion/SIF/myfiles/"
        en_weightfile = path + 'en_full_freq.txt'  # word - frequency
        de_weightfile = path + lang + '_full_freq.txt'  # word - frequency
        weightpara = 1e-3  # the parameter in the SIF weighting

        en_word2weight = data_io.getWordWeight(en_weightfile, weightpara)
        self.en_weight4ind = data_io.getWeight(self.en_words, en_word2weight)

        de_word2weight = data_io.getWordWeight(de_weightfile, weightpara)
        self.de_weight4ind = data_io.getWeight(self.de_words, de_word2weight)

    # Preparation for TFIDF
    def prepare_tfidf(self, dataset):
        corres = set()

        for col in list(dataset):
            field = col[7:]
            if "ltable_" + field in list(dataset) and "rtable_" + field in list(dataset) and field != "id":
                corres.add(field)

        # For English data
        cols = [col for col in list(dataset) if "ltable_" in col and col[7:] in corres]
        en = dataset[cols]
        new_en = en.drop_duplicates(subset=cols)

        en_whole = new_en[list(new_en)[0]]
        for col in list(new_en)[1:]:
            en_whole = en_whole.append(new_en[col])

        en_tfidf = TfidfVectorizer(analyzer='word')
        en_tfidf.fit(en_whole.values.astype('U'))
        max_idf = max(en_tfidf.idf_)
        self.en_word2weight = defaultdict(
            lambda: max_idf,
            [(w, en_tfidf.idf_[i]) for w, i in en_tfidf.vocabulary_.items()])

        # For German data
        cols = [col for col in list(dataset) if "rtable_" in col and col[7:] in corres]
        de = dataset[cols]
        new_de = de.drop_duplicates(subset=cols)

        de_whole = new_de[list(new_de)[0]]
        for col in list(new_de)[1:]:
            de_whole = de_whole.append(new_de[col])

        de_tfidf = TfidfVectorizer(analyzer='word')
        de_tfidf.fit(de_whole.values.astype('U'))
        max_idf = max(de_tfidf.idf_)
        self.de_word2weight = defaultdict(
            lambda: max_idf,
            [(w, de_tfidf.idf_[i]) for w, i in de_tfidf.vocabulary_.items()])


    # This is the preparation of the class, initiates other functions
    def prepare(self, embedding, lang, dataset, oov_pickle_en, oov_pickle_de, sense_pickle_en, sense_pickle_de):
        self.embedding = embedding
        path = "/home/oyku/embeddings/"

        if embedding == "fasttext":
            # Only in FastText, we parametrize the other language
            # Because this is the only embedding we test with other languages
            EN_PATH = path + "fasttext/wiki.en.align.vec"
            DE_PATH = path + "fasttext/wiki." + lang + ".align.vec"

            self.en_dictionary, self.en_words, self.en_We = self.load_vec(EN_PATH)
            self.de_dictionary, self.de_words, self.de_We = self.load_vec(DE_PATH)

        # Babylon word embeddings need to be aligned
        elif embedding == "babylon":
            EN_PATH = path + "babylon/wiki.en.vec"
            DE_PATH = path + "babylon/wiki.de.vec"

            self.en_dictionary = FastVector(vector_file=EN_PATH)
            self.de_dictionary = FastVector(vector_file=DE_PATH)

            align_path = "/home/oyku/myversion/babylon/alignment_matrices/"
            self.en_dictionary.apply_transform(align_path + 'en.txt')
            self.de_dictionary.apply_transform(align_path + 'de.txt')

        elif embedding == "fbmuse":
            EN_PATH = path + "fbmuse/wiki.multi.en.vec.txt"
            DE_PATH = path + "fbmuse/wiki.multi.de.vec.txt"

            self.en_dictionary, self.en_words, self.en_We = self.load_vec(EN_PATH)
            self.de_dictionary, self.de_words, self.de_We = self.load_vec(DE_PATH)


        elif embedding == "multicca":
            EN_PATH = path + "multicca/multicca_en.txt"
            DE_PATH = path + "multicca/multicca_de.txt"

            self.en_dictionary, self.en_words, self.en_We = self.load_vec_multivecs(EN_PATH)
            self.de_dictionary, self.de_words, self.de_We = self.load_vec_multivecs(DE_PATH)

        elif embedding == "multiskip":
            EN_PATH = path + "multiskip/multiskip_en.txt"
            DE_PATH = path + "multiskip/multiskip_de.txt"

            self.en_dictionary, self.en_words, self.en_We = self.load_vec_multivecs(EN_PATH)
            self.de_dictionary, self.de_words, self.de_We = self.load_vec_multivecs(DE_PATH)

        elif embedding == "multicluster":
            EN_PATH = path + "multicluster/multicluster_en.txt"
            DE_PATH = path + "multicluster/multicluster_de.txt"

            self.en_dictionary, self.en_words, self.en_We = self.load_vec_multivecs(EN_PATH)
            self.de_dictionary, self.de_words, self.de_We = self.load_vec_multivecs(DE_PATH)

        elif embedding == "translationInvariance":
            EN_PATH = path + "translationInvariance/translation_invariance_en.txt"
            DE_PATH = path + "translationInvariance/translation_invariance_de.txt"

            self.en_dictionary, self.en_words, self.en_We = self.load_vec_multivecs(EN_PATH)
            self.de_dictionary, self.de_words, self.de_We = self.load_vec_multivecs(DE_PATH)


        elif embedding == "bilbowa":
            EN_PATH = path + "bilbowa/en_de_vec_wiki_shelf"
            DE_PATH = path + "bilbowa/de_en_vec_wiki_shelf"

            self.en_dictionary, self.en_words, self.en_We = self.load_vec_bilbowa(EN_PATH)
            self.de_dictionary, self.de_words, self.de_We = self.load_vec_bilbowa(DE_PATH)

        elif "shuffle" in embedding:
            EN_PATH = path + "shuffle/" + embedding + ".vec"
            DE_PATH = path + "shuffle/" + embedding + ".vec"

            self.en_dictionary, self.en_words, self.en_We = self.load_vec(EN_PATH)
            self.de_dictionary, self.de_words, self.de_We = self.load_vec(DE_PATH)


        self.dim = len(self.en_dictionary["."])

        print(embedding.title() + " word embeddings are read")

        # Prepare for tf-idf
        self.prepare_tfidf(dataset)
        print("tfidf vector preparation is completed")

        # Prepare for SIF vector
        self.prepare_sif(lang)
        print("SIF vector preparation is completed")

        self.en_senses = pickle.load(open(sense_pickle_en, "rb"))
        self.de_senses = pickle.load(open(sense_pickle_de, "rb"))
        print("Senses are extracted")

    def cosine_similarity(self, ar1, ar2):
        result = 1 - spatial.distance.cosine(ar1, ar2)
        return result


    # Mean Vector Similarity (MeVS)
    def get_mean_sim(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        en_word_emb = np.mean(
            [self.en_dictionary[w] for w in en_tokens if w in self.en_dictionary]
            or [np.zeros(self.dim)],
            axis=0)
        de_word_emb = np.mean(
            [self.de_dictionary[w] for w in de_tokens if w in self.de_dictionary]
            or [np.zeros(self.dim)],
            axis=0)

        if np.count_nonzero(en_word_emb) == 0 or np.count_nonzero(de_word_emb) == 0:
            return -1

        score = self.cosine_similarity(en_word_emb, de_word_emb)
        return score


    # Maximum Vector Similarity (MaVS)    
    def get_max_sim(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        en_emb_vecs = [self.en_dictionary[w] for w in en_tokens if w in self.en_dictionary]
        de_emb_vecs = [self.de_dictionary[w] for w in de_tokens if w in self.de_dictionary]

        if np.count_nonzero(en_emb_vecs) == 0 or np.count_nonzero(de_emb_vecs) == 0:
            return -1

        maxscore = -1

        for en_emb in en_emb_vecs:
            for de_emb in de_emb_vecs:

                if np.count_nonzero(en_emb) != 0 and np.count_nonzero(de_emb) != 0:

                    score = self.cosine_similarity(en_emb, de_emb)

                    if score > maxscore:
                        maxscore = score

        return maxscore


        # Optimal Alignment Similarity (OAS)
    def get_optimal_alignment(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())


        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        diff = len(en_tokens) - len(de_tokens)
        length = max(len(en_tokens), len(de_tokens))
        mul = len(en_tokens) * len(de_tokens)
        total = len(en_tokens) + len(de_tokens)

        # Padding the short text with #
        if diff < 0:
            en_tokens = en_tokens + ["######"] * abs(diff)

        elif diff > 0:
            de_tokens = de_tokens + ["######"] * abs(diff)

        matrix = []
        for en_el in en_tokens:

            if en_el == "######" or en_el not in self.en_dictionary:
                vector = [-1] * length

            else:
                vector = []
                for de_el in de_tokens:

                    if de_el == "######" or de_el not in self.de_dictionary:
                        vector.append(-1)

                    else:
                        score = self.cosine_similarity(self.en_dictionary[en_el], self.de_dictionary[de_el])
                        vector.append(score)

            matrix.append(vector)

        m = np.matrix(matrix)
        m = -1 * m
        row_i, col_i = linear_sum_assignment(m)

        align_score = 0
        for i, j in zip(row_i, col_i):
            align_score += m[i, j] * (-1)

        result = float(align_score) * total / (2 * mul)
        return result


    # Maximum Alignment Similarity (MAS)
    def greedy_aligned_words(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        # for each English word
        sum_en = 0
        for en_token in en_tokens:
            sim, best_aligned = self.get_max_word_alignment(en_token, de_tokens, 0)  # word is in English
            sum_en += sim

        # for each German word
        sum_de = 0
        for de_token in de_tokens:
            sim, best_aligned = self.get_max_word_alignment(de_token, en_tokens, 1)  # word is in German
            sum_de += sim

        sim_en = float(sum_en) / len(en_tokens)
        sim_de = float(sum_de) / len(de_tokens)

        score = float(sim_en + sim_de) / 2
        return score

    # Gets the tfidf embedding and then again give an average embedding by mean
    # Then computes the score
    def get_tfidf_mean_sim(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        en_word_emb = np.mean(
            [self.en_dictionary[w] * self.en_word2weight[w] for w in en_tokens if
             w in self.en_dictionary]
            or [np.zeros(self.dim)],
            axis=0)

        de_word_emb = np.mean(
            [self.de_dictionary[w] * self.de_word2weight[w] for w in de_tokens if
             w in self.de_dictionary]
            or [np.zeros(self.dim)],
            axis=0)

        if np.count_nonzero(en_word_emb) == 0 or np.count_nonzero(de_word_emb) == 0:
            return -1

        score = self.cosine_similarity(en_word_emb, de_word_emb)
        return score



    def get_tfidf_max_sim(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        en_emb_vecs = [self.en_dictionary[w] * self.en_word2weight[w] for w in en_tokens if w in self.en_dictionary]
        de_emb_vecs = [self.de_dictionary[w] * self.de_word2weight[w] for w in de_tokens if w in self.de_dictionary]

        if np.count_nonzero(en_emb_vecs) == 0 or np.count_nonzero(de_emb_vecs) == 0:
            return -1

        maxscore = -1

        for en_emb in en_emb_vecs:
            for de_emb in de_emb_vecs:

                if np.count_nonzero(en_emb) != 0 and np.count_nonzero(de_emb) != 0:

                    score = self.cosine_similarity(en_emb, de_emb)

                    if score > maxscore:
                        maxscore = score

        return maxscore

    # Cares the word which is most important according to the tfidf (with higher weight)
    # Computes the score of most important words' embeddings
    def get_tfidf_max_weight(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())


        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        en_weight = -100
        en_important_token = ""

        for w in en_tokens:
            if w in self.en_dictionary:
                weight = self.en_word2weight[w]

                if weight > en_weight:
                    en_weight = weight
                    en_important_token = w

        de_weight = -100
        de_important_token = ""

        for w in de_tokens:
            if w in self.de_dictionary:
                weight = self.de_word2weight[w]

                if weight > de_weight:
                    de_weight = weight
                    de_important_token = w

        if de_important_token == "" or en_important_token == "":
            return -1

        en_word_emb = self.en_dictionary[en_important_token]
        de_word_emb = self.de_dictionary[de_important_token]

        if np.count_nonzero(en_word_emb) == 0 or np.count_nonzero(de_word_emb) == 0:
            return -1

        score = self.cosine_similarity(en_word_emb, de_word_emb)
        return score


    def get_vector_composition(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())


        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        union_list = collections.OrderedDict()
        for el in en_tokens:
            union_list[el] = 0

        for el in de_tokens:
            if el not in union_list:
                union_list[el] = 1

        en_vector = []
        de_vector = []

        for el, lang in union_list.items():
            en_max = -1
            de_max = -1

            if lang == 0:
                if el in self.en_dictionary:
                    for word in en_tokens:
                        if word in self.en_dictionary:
                            score = self.cosine_similarity(self.en_dictionary[el], self.en_dictionary[word])

                            if score > en_max:
                                en_max = score
                    en_vector.append(en_max)

                    for word in de_tokens:
                        if word in self.de_dictionary:
                            score = self.cosine_similarity(self.en_dictionary[el], self.de_dictionary[word])

                            if score > de_max:
                                de_max = score

                    de_vector.append(de_max)

                # Putting Nan for OOV words
                else:
                    en_vector.append(pd.np.NaN)
                    de_vector.append(pd.np.NaN)
            else:
                if el in self.de_dictionary:
                    for word in en_tokens:
                        if word in self.en_dictionary:
                            score = self.cosine_similarity(self.de_dictionary[el], self.en_dictionary[word])

                            if score > en_max:
                                en_max = score
                    en_vector.append(en_max)

                    for word in de_tokens:
                        if word in self.de_dictionary:
                            score = self.cosine_similarity(self.de_dictionary[el], self.de_dictionary[word])

                            if score > de_max:
                                de_max = score

                    de_vector.append(de_max)

                # Putting Nan for OOV words
                else:
                    en_vector.append(pd.np.NaN)
                    de_vector.append(pd.np.NaN)

        # Replacing Nans with mean
        en = np.where(np.isnan(en_vector), np.nanmean(en_vector), en_vector)
        de = np.where(np.isnan(de_vector), np.nanmean(de_vector), de_vector)

        return self.cosine_similarity(en, de)










    def weighted_greedy_aligned_words(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        # for each English word
        sum_en = 0
        sum_weights_en = 0
        for en_token in en_tokens:
            sim, best_aligned = self.get_max_word_alignment(en_token, de_tokens, 0)  # word is in English
            sum_en += sim * self.en_word2weight[en_token]
            sum_weights_en += self.en_word2weight[en_token]

        # for each German word
        sum_de = 0
        sum_weights_de = 0
        for de_token in de_tokens:
            sim, best_aligned = self.get_max_word_alignment(de_token, en_tokens, 1)  # word is in German
            sum_de += sim * self.de_word2weight[de_token]
            sum_weights_de += self.de_word2weight[de_token]

        sim_en = float(sum_en) / sum_weights_en
        sim_de = float(sum_de) / sum_weights_de

        score = float(sim_en + sim_de) / 2
        return score


    '''This feature is from the paper Cross-Language Record Linkage by Exploiting Semantic Matching of Textual Metadata
    to be used on translated pairs
    It is used as a baseline '''

    def record_linkage_baseline(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        # s2 is the translated one
        contribution = 0
        for de_token in de_tokens:
            maxscore = 0
            #Since German is translated now, we look for it English word embedding
            if de_token in self.en_dictionary:

                for en_token in en_tokens:
                    if en_token in self.en_dictionary:
                        score = self.cosine_similarity(self.en_dictionary[en_token], self.en_dictionary[de_token])

                        if maxscore < score:
                            maxscore = score
                contribution += maxscore

        return contribution

        # To be used in other fuctions  - returns the sense of a word with max similarity
    def get_wordnet_synset_max_sim(self, word, lang):

        wn_synset = ""

        if lang == 0:  # word is English
            if word in self.en_senses:
                score = 0
                synset = ""
                for key, value in self.en_senses[word].items():
                    if float(value) > score:
                        score = float(value)
                        synset = key

                if synset[3:].strip().isdigit():
                    wn_synset = wn.synset_from_pos_and_offset(synset[2], int(synset[3:].strip()))


        else:  # word is German
            if word in self.de_senses:
                score = 0
                synset = ""
                for key, value in self.de_senses[word].items():
                    if float(value) > score:
                        score = float(value)
                        synset = key

                if synset[3:].strip().isdigit():
                    wn_synset = wn.synset_from_pos_and_offset(synset[2], int(synset[3:].strip()))

        return wn_synset

        # To be used in other functions  - returns the word from other sentence with max alignment score
        def get_max_word_alignment(self, word, sentence, lang):

            maxscore = -1
            best_aligned = ""
            if lang == 0:  # word is in English, sentence is in German

                if word not in self.en_dictionary:
                    maxscore = 0
                    return maxscore, best_aligned

                for token in sentence:
                    if token in self.de_dictionary:
                        score = self.cosine_similarity(self.en_dictionary[word], self.de_dictionary[token])

                        if maxscore < score:
                            maxscore = score
                            best_aligned = token

            else:  # word is in German, sentence is in English

                if word not in self.de_dictionary:
                    maxscore = 0
                    return maxscore, best_aligned

                for token in sentence:
                    if token in self.en_dictionary:
                        score = self.cosine_similarity(self.de_dictionary[word], self.en_dictionary[token])

                        if maxscore < score:
                            maxscore = score
                            best_aligned = token

            return maxscore, best_aligned

    def aligned_words_senses_jaccard(self, s1, s2):

        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        # for each English word
        sum_en = 0
        for en_token in en_tokens:
            sim, best_aligned = self.get_max_word_alignment(en_token, de_tokens, 0)  # word is in English

            if en_token in self.en_senses and best_aligned in self.de_senses:
                en_sense_list = list(self.en_senses[en_token])
                de_sense_list = list(self.de_senses[best_aligned])

                # Jaccard similarity over list
                en_sim = len(set(en_sense_list).intersection(de_sense_list)) / len(
                    set(en_sense_list).union(de_sense_list))

                sum_en += en_sim

        # for each German word
        sum_de = 0
        for de_token in de_tokens:
            sim, best_aligned = self.get_max_word_alignment(de_token, en_tokens, 1)  # word is in German

            if de_token in self.de_senses and best_aligned in self.en_senses:
                en_sense_list = list(self.en_senses[best_aligned])
                de_sense_list = list(self.de_senses[de_token])

                de_sim = len(set(en_sense_list).intersection(de_sense_list)) / len(
                    set(en_sense_list).union(de_sense_list))

                sum_de += de_sim

        sim_en = float(sum_en) / len(en_tokens)
        sim_de = float(sum_de) / len(de_tokens)

        score = float(sim_en + sim_de) / 2
        return score


    def weighted_aligned_words_senses_jaccard(self, s1, s2):

        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        # for each English word
        sum_en = 0
        sum_weights_en = 0
        for en_token in en_tokens:
            sim, best_aligned = self.get_max_word_alignment(en_token, de_tokens, 0)  # word is in English

            if en_token in self.en_senses and best_aligned in self.de_senses:
                en_sense_list = list(self.en_senses[en_token])
                de_sense_list = list(self.de_senses[best_aligned])

                # Jaccard similarity over list
                en_sim = len(set(en_sense_list).intersection(de_sense_list)) / len(
                    set(en_sense_list).union(de_sense_list))

                sum_en += en_sim * self.en_word2weight[en_token]
            sum_weights_en += self.en_word2weight[en_token]

        # for each German word
        sum_de = 0
        sum_weights_de = 0
        for de_token in de_tokens:
            sim, best_aligned = self.get_max_word_alignment(de_token, en_tokens, 1)  # word is in German

            if de_token in self.de_senses and best_aligned in self.en_senses:
                en_sense_list = list(self.en_senses[best_aligned])
                de_sense_list = list(self.de_senses[de_token])

                de_sim = len(set(en_sense_list).intersection(de_sense_list)) / len(
                    set(en_sense_list).union(de_sense_list))

                sum_de += de_sim * self.de_word2weight[de_token]
            sum_weights_de += self.de_word2weight[de_token]

        sim_en = float(sum_en) / sum_weights_en
        sim_de = float(sum_de) / sum_weights_de

        score = float(sim_en + sim_de) / 2
        return score


    def aligned_words_senses_path_sim(self, s1, s2):

        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        # for each English word
        sum_en = 0
        for en_token in en_tokens:
            sim, best_aligned = self.get_max_word_alignment(en_token, de_tokens, 0)  # word is in English

            en_sense = self.get_wordnet_synset_max_sim(en_token, 0)
            de_sense = self.get_wordnet_synset_max_sim(best_aligned, 1)

            if not isinstance(en_sense, str) and not isinstance(de_sense, str):
                en_sim = en_sense.path_similarity(de_sense)

                if en_sim is not None:
                    sum_en += en_sim

        # for each German word
        sum_de = 0
        for de_token in de_tokens:
            sim, best_aligned = self.get_max_word_alignment(de_token, en_tokens, 1)  # word is in German

            en_sense = self.get_wordnet_synset_max_sim(best_aligned, 0)
            de_sense = self.get_wordnet_synset_max_sim(de_token, 1)

            if not isinstance(en_sense, str) and not isinstance(de_sense, str):
                de_sim = en_sense.path_similarity(de_sense)

                if de_sim is not None:
                    sum_de += de_sim

        sim_en = float(sum_en) / len(en_tokens)
        sim_de = float(sum_de) / len(de_tokens)

        score = float(sim_en + sim_de) / 2
        return score


    def weighted_aligned_words_senses_path_sim(self, s1, s2):

        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return pd.np.NaN

        # for each English word
        sum_en = 0
        sum_weights_en = 0
        for en_token in en_tokens:
            sim, best_aligned = self.get_max_word_alignment(en_token, de_tokens, 0)  # word is in English

            en_sense = self.get_wordnet_synset_max_sim(en_token, 0)
            de_sense = self.get_wordnet_synset_max_sim(best_aligned, 1)

            if not isinstance(en_sense, str) and not isinstance(de_sense, str):
                en_sim = en_sense.path_similarity(de_sense)

                if en_sim is not None:
                    sum_en += en_sim * self.en_word2weight[en_token]
            sum_weights_en += self.en_word2weight[en_token]

        # for each German word
        sum_de = 0
        sum_weights_de = 0
        for de_token in de_tokens:
            sim, best_aligned = self.get_max_word_alignment(de_token, en_tokens, 1)  # word is in German

            en_sense = self.get_wordnet_synset_max_sim(best_aligned, 0)
            de_sense = self.get_wordnet_synset_max_sim(de_token, 1)

            if not isinstance(en_sense, str) and not isinstance(de_sense, str):
                de_sim = en_sense.path_similarity(de_sense)

                if de_sim is not None:
                    sum_de += de_sim * self.de_word2weight[de_token]
            sum_weights_de += self.de_word2weight[de_token]

        sim_en = float(sum_en) / sum_weights_en
        sim_de = float(sum_de) / sum_weights_de

        score = float(sim_en + sim_de) / 2
        return score


    def get_sif(self, s1, s2):
        s1 = list(s1.lower())
        s2 = list(s2.lower())

        if len(s1) == 0 or len(s2) == 0:
            return pd.np.NaN

        ## English data
        en_x, en_m = data_io.sentences2idx(s1, self.en_words)
        en_w = data_io.seq2weight(en_x, en_m, self.en_weight4ind)
        en_embedding = SIF_embedding.SIF_embedding(self.en_We, en_x, en_w, self.parameters)
        en_embedding = en_embedding[0]

        ## German data
        de_x, de_m = data_io.sentences2idx(s2, self.de_words)
        de_w = data_io.seq2weight(de_x, de_m, self.de_weight4ind)
        de_embedding = SIF_embedding.SIF_embedding(self.de_We, de_x, de_w, self.parameters)
        de_embedding = de_embedding[0]

        if np.count_nonzero(en_embedding) == 0 or np.count_nonzero(de_embedding) == 0:
            return -1

        score = self.cosine_similarity(en_embedding, de_embedding)
        return score




    ''' Functions below are not used in the end'''

    # def get_tfidf_sum_sim(self, s1, s2):
    #     if self.embedding != "bilbowa":
    #         en_tokens = word_tokenize(s1.lower())
    #         de_tokens = word_tokenize(s2.lower())
    #
    #     else:
    #         en_tokens = word_tokenize(s1)
    #         de_tokens = word_tokenize(s2)
    #
    #     if len(en_tokens) == 0 or len(de_tokens) == 0:
    #         return pd.np.NaN
    #
    #     en_word_emb = np.sum(
    #         [self.en_dictionary[w] * self.en_word2weight[w] for w in en_tokens if
    #          w in self.en_dictionary]
    #         or [np.zeros(self.dim)],
    #         axis=0)
    #
    #     de_word_emb = np.sum(
    #         [self.de_dictionary[w] * self.de_word2weight[w] for w in de_tokens if
    #          w in self.de_dictionary]
    #         or [np.zeros(self.dim)],
    #         axis=0)
    #
    #     if np.count_nonzero(en_word_emb) == 0 or np.count_nonzero(de_word_emb) == 0:
    #         return -1
    #
    #     score = self.cosine_similarity(en_word_emb, de_word_emb)
    #     return score

    # def get_sum_sim(self, s1, s2):
    #     if self.embedding != "bilbowa":
    #         en_tokens = word_tokenize(s1.lower())
    #         de_tokens = word_tokenize(s2.lower())
    #
    #     else:
    #         en_tokens = word_tokenize(s1)
    #         de_tokens = word_tokenize(s2)
    #
    #     if len(en_tokens) == 0 or len(de_tokens) == 0:
    #         return pd.np.NaN
    #
    #     en_word_emb = np.sum(
    #         [self.en_dictionary[w] for w in en_tokens if w in self.en_dictionary]
    #         or [np.zeros(self.dim)],
    #         axis=0)
    #     de_word_emb = np.sum(
    #         [self.de_dictionary[w] for w in de_tokens if w in self.de_dictionary]
    #         or [np.zeros(self.dim)],
    #         axis=0)
    #
    #     if np.count_nonzero(en_word_emb) == 0 or np.count_nonzero(de_word_emb) == 0:
    #         return -1
    #
    #     score = self.cosine_similarity(en_word_emb, de_word_emb)
    #     return score


    # def weighted_greedy_aligned_words_postags(self, s1, s2):
    #     text_en = Text(s1, hint_language_code='en')
    #     text_de = Text(s2, hint_language_code='de')
    #
    #     en_tokens = text_en.pos_tags
    #     de_tokens = text_de.pos_tags
    #
    #     if len(en_tokens) == 0 or len(de_tokens) == 0:
    #         return pd.np.NaN
    #
    #     sum_en = 0
    #     sum_weights_en = 0
    #     for en_token in en_tokens:
    #         maxscore = -2
    #         if en_token[0].lower() in self.en_dictionary:
    #             for de_token in de_tokens:
    #
    #                 if de_token[0].lower() in self.de_dictionary and en_token[1] == de_token[1]:
    #                     score = self.cosine_similarity(self.en_dictionary[en_token[0].lower()],
    #                                                    self.de_dictionary[de_token[0].lower()])
    #
    #                     if maxscore < score:
    #                         maxscore = score
    #
    #         if maxscore != -2:
    #             sum_en += maxscore
    #
    #         sum_weights_en += self.en_word2weight[en_token[0].lower()]
    #
    #     sum_de = 0
    #     sum_weights_de = 0
    #     for de_token in de_tokens:
    #         maxscore = -2
    #         if de_token[0].lower() in self.de_dictionary:
    #             for en_token in en_tokens:
    #
    #                 if en_token[0].lower() in self.en_dictionary and en_token[1] == de_token[1]:
    #                     score = self.cosine_similarity(self.en_dictionary[en_token[0].lower()],
    #                                                    self.de_dictionary[de_token[0].lower()])
    #
    #                     if maxscore < score:
    #                         maxscore = score
    #
    #         if maxscore != -2:
    #             sum_de += maxscore
    #
    #         sum_weights_de += self.de_word2weight[de_token[0].lower()]
    #
    #     sim_en = float(sum_en) / sum_weights_en
    #     sim_de = float(sum_de) / sum_weights_de
    #
    #     score = float(sim_en + sim_de) / 2
    #     return score
    #
    # def greedy_aligned_words_postags(self, s1, s2):
    #     text_en = Text(s1, hint_language_code='en')
    #     text_de = Text(s2, hint_language_code='de')
    #
    #     en_tokens = text_en.pos_tags
    #     de_tokens = text_de.pos_tags
    #
    #     # en_tokens = word_tokenize(s1.lower())
    #     # de_tokens = word_tokenize(s2.lower())
    #
    #     if len(en_tokens) == 0 or len(de_tokens) == 0:
    #         return pd.np.NaN
    #
    #     sum_en = 0
    #     sum_weights_en = 0
    #     for en_token in en_tokens:
    #         maxscore = -2
    #         if en_token[0].lower() in self.en_dictionary:
    #             for de_token in de_tokens:
    #
    #                 if de_token[0].lower() in self.de_dictionary and en_token[1] == de_token[1]:
    #                     score = self.cosine_similarity(self.en_dictionary[en_token[0].lower()],
    #                                                    self.de_dictionary[de_token[0].lower()])
    #
    #                     if maxscore < score:
    #                         maxscore = score
    #
    #         if maxscore != -2:
    #             sum_en += maxscore
    #
    #     sum_de = 0
    #     sum_weights_de = 0
    #     for de_token in de_tokens:
    #         maxscore = -2
    #         if de_token[0].lower() in self.de_dictionary:
    #             for en_token in en_tokens:
    #
    #                 if en_token[0].lower() in self.en_dictionary and en_token[1] == de_token[1]:
    #                     score = self.cosine_similarity(self.en_dictionary[en_token[0].lower()],
    #                                                    self.de_dictionary[de_token[0].lower()])
    #
    #                     if maxscore < score:
    #                         maxscore = score
    #
    #         if maxscore != -2:
    #             sum_de += maxscore
    #
    #     sim_en = float(sum_en) / len(en_tokens)
    #     sim_de = float(sum_de) / len(de_tokens)
    #
    #     score = float(sim_en + sim_de) / 2
    #     return score
    #
    # def get_vector_composition_tfidf(self, s1, s2):
    #     en_tokens = word_tokenize(s1.lower())
    #     de_tokens = word_tokenize(s2.lower())
    #
    #     if len(en_tokens) == 0 or len(de_tokens) == 0:
    #         return pd.np.NaN
    #
    #     union_list = collections.OrderedDict()
    #     for el in en_tokens:
    #         union_list[el] = 0
    #
    #     for el in de_tokens:
    #         if el not in union_list:
    #             union_list[el] = 1
    #
    #     en_vector = []
    #     de_vector = []
    #
    #     for el, lang in union_list.items():
    #         en_max = -1
    #         de_max = -1
    #
    #         if lang == 0:
    #             if el in self.en_dictionary:
    #                 for word in en_tokens:
    #                     if word in self.en_dictionary:
    #                         score = self.cosine_similarity(self.en_dictionary[el] * self.en_word2weight[el],
    #                                                        self.en_dictionary[word] * self.en_word2weight[word])
    #
    #                         if score > en_max:
    #                             en_max = score
    #                 en_vector.append(en_max)
    #
    #                 for word in de_tokens:
    #                     if word in self.de_dictionary:
    #                         score = self.cosine_similarity(self.en_dictionary[el] * self.en_word2weight[el],
    #                                                        self.de_dictionary[word] * self.de_word2weight[word])
    #
    #                         if score > de_max:
    #                             de_max = score
    #
    #                 de_vector.append(de_max)
    #
    #             # Putting Nan for OOV words
    #             else:
    #                 en_vector.append(pd.np.NaN)
    #                 de_vector.append(pd.np.NaN)
    #         else:
    #             if el in self.de_dictionary:
    #                 for word in en_tokens:
    #                     if word in self.en_dictionary:
    #                         score = self.cosine_similarity(self.de_dictionary[el] * self.de_word2weight[el],
    #                                                        self.en_dictionary[word] * self.en_word2weight[word])
    #
    #                         if score > en_max:
    #                             en_max = score
    #                 en_vector.append(en_max)
    #
    #                 for word in de_tokens:
    #                     if word in self.de_dictionary:
    #                         score = self.cosine_similarity(self.de_dictionary[el] * self.de_word2weight[el],
    #                                                        self.de_dictionary[word] * self.de_word2weight[word])
    #
    #                         if score > de_max:
    #                             de_max = score
    #
    #                 de_vector.append(de_max)
    #
    #             # Putting Nan for OOV words
    #             else:
    #                 en_vector.append(pd.np.NaN)
    #                 de_vector.append(pd.np.NaN)
    #
    #     # Replacing Nans with mean
    #     en = np.where(np.isnan(en_vector), np.nanmean(en_vector), en_vector)
    #     de = np.where(np.isnan(de_vector), np.nanmean(de_vector), de_vector)
    #
    #     return self.cosine_similarity(en, de)
    #
    # def get_min_sim(self, s1, s2):
    #     en_tokens = word_tokenize(s1.lower())
    #     de_tokens = word_tokenize(s2.lower())
    #
    #     en_emb_vecs = [self.en_dictionary[w] for w in en_tokens if w in self.en_dictionary]
    #     de_emb_vecs = [self.de_dictionary[w] for w in de_tokens if w in self.de_dictionary]
    #
    #     if np.count_nonzero(en_emb_vecs) == 0 or np.count_nonzero(de_emb_vecs) == 0:
    #         return -1
    #
    #     minscore = 100
    #
    #     for en_emb in en_emb_vecs:
    #         for de_emb in de_emb_vecs:
    #
    #             if np.count_nonzero(en_emb) != 0 and np.count_nonzero(de_emb) != 0:
    #
    #                 score = self.cosine_similarity(en_emb, de_emb)
    #
    #                 if score < minscore:
    #                     minscore = score
    #
    #     if minscore == 100:
    #         minscore = -1
    #
    #     return minscore
    #
    # # Get the word embedding as it is, dont compute any similarity score
    # def get_wordvec_mean(self, list1, list2):
    #     en_tokens = [item for cell in list1 for item in word_tokenize(cell.lower())]
    #     de_tokens = [item for cell in list2 for item in word_tokenize(cell.lower())]
    #
    #     en_word_emb = np.mean(
    #         [self.en_dictionary[w] for w in en_tokens if w in self.en_dictionary]
    #         or [np.zeros(self.dim)],
    #         axis=0)
    #     de_word_emb = np.mean(
    #         [self.de_dictionary[w] for w in de_tokens if w in self.de_dictionary]
    #         or [np.zeros(self.dim)],
    #         axis=0)
    #
    #     if np.count_nonzero(en_word_emb) == 0 or np.count_nonzero(de_word_emb) == 0:
    #         return -1
    #
    #     substracted = np.subtract(en_word_emb, de_word_emb)
    #     return substracted

    # def get_tfidf_mean_sim(self, s1, s2):
    #
    #     en_tokens = word_tokenize(s1.lower())
    #     de_tokens = word_tokenize(s2.lower())
    #
    #     en_word_emb = np.mean(
    #         [self.en_dictionary[w] * self.en_word2weight[w] for w in en_tokens if
    #          w in self.en_dictionary and w not in self.en_stop_words]
    #         or [np.zeros(self.dim)],
    #         axis=0)
    #
    #     de_word_emb = np.mean(
    #         [self.de_dictionary[w] * self.de_word2weight[w] for w in de_tokens if
    #          w in self.de_dictionary and w not in self.de_stop_words]
    #         or [np.zeros(self.dim)],
    #         axis=0)
    #
    #     if np.count_nonzero(en_word_emb) == 0 or np.count_nonzero(de_word_emb) == 0:
    #         return -1
    #
    #
    #     score = self.cosine_similarity(en_word_emb, de_word_emb)
    #     return score

    # def get_tfidf_max_weight(self, s1, s2):
    #     en_tokens = word_tokenize(s1.lower())
    #     de_tokens = word_tokenize(s2.lower())
    #
    #     en_weight = -100
    #     en_important_token = ""
    #
    #     for w in en_tokens:
    #         if w in self.en_dictionary and w not in self.en_stop_words:
    #             weight = self.en_word2weight[w]
    #
    #             if weight > en_weight:
    #                 en_weight = weight
    #                 en_important_token = w
    #
    #     de_weight = -100
    #     de_important_token = ""
    #
    #     for w in de_tokens:
    #         if w in self.de_dictionary and w not in self.de_stop_words:
    #             weight = self.de_word2weight[w]
    #
    #             if weight > de_weight:
    #                 de_weight = weight
    #                 de_important_token = w
    #
    #     if de_important_token == "" or en_important_token == "":
    #         return -1
    #
    #     en_word_emb = self.en_dictionary[en_important_token]
    #     de_word_emb = self.de_dictionary[de_important_token]
    #
    #     if np.count_nonzero(en_word_emb) == 0 or np.count_nonzero(de_word_emb) == 0:
    #         return -1
    #
    #     score = FastVector.cosine_similarity(en_word_emb, de_word_emb)
    #     # score = self.cosine_similarity(en_word_emb, de_word_emb)
    #     return score

    # def get_tfidf_sum_sim(self, s1, s2):
    #
    #     en_tokens = word_tokenize(s1.lower())
    #     de_tokens = word_tokenize(s2.lower())
    #
    #     en_word_emb = np.sum(
    #         [self.en_dictionary[w] * self.en_word2weight[w] for w in en_tokens if
    #          w in self.en_dictionary and w not in self.en_stop_words]
    #         or [np.zeros(self.dim)],
    #         axis=0)
    #
    #     de_word_emb = np.sum(
    #         [self.de_dictionary[w] * self.de_word2weight[w] for w in de_tokens if
    #          w in self.de_dictionary and w not in self.de_stop_words]
    #         or [np.zeros(self.dim)],
    #         axis=0)
    #
    #     if np.count_nonzero(en_word_emb) == 0 or np.count_nonzero(de_word_emb) == 0:
    #         return -1
    #
    #
    #     score = self.cosine_similarity(en_word_emb, de_word_emb)
    #     return score

    # def get_TfIdfVec_whole_sim(self, s1, s2):
    #
    #     en_tokens = word_tokenize(s1.lower())
    #     de_tokens = word_tokenize(s2.lower())
    #
    #     en_word_emb = np.mean(
    #         [self.en_dictionary[w] * self.en_whole_word2weight[w] for w in en_tokens if
    #          w in self.en_dictionary and w not in self.en_stop_words]
    #         or [np.zeros(self.dim)],
    #         axis=0)
    #
    #     de_word_emb = np.mean(
    #         [self.de_dictionary[w] * self.de_whole_word2weight[w] for w in de_tokens if
    #          w in self.de_dictionary and w not in self.de_stop_words]
    #         or [np.zeros(self.dim)],
    #         axis=0)
    #
    #     if np.count_nonzero(en_word_emb) == 0 or np.count_nonzero(de_word_emb) == 0:
    #         return -1
    #
    #     score = FastVector.cosine_similarity(en_word_emb, de_word_emb)
    #     # score = self.cosine_similarity(en_word_emb, de_word_emb)
    #     return score
