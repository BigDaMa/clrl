'''
-Uses Magellan's feature creation structure
-Finds corresponding attributes of dataset A and B
-Prepares data
-Manages feature extraction
'''


import pandas as pd
import six
from clrl import helper
import pyprind
import multiprocessing
import py_stringmatching as sm
from cloudpickle import cloudpickle
from joblib import Parallel
from joblib import delayed

import similarity_embedding as cl_emb
import similarity_embedding_oov as cl_emb_oov

sim_func = None
name_of_class = "crosslang_feature_extraction"
chosen_emb = None
is_oov = None

embeddings = ["fasttext", "babylon", "fbmuse", "multicca", "multiskip", "multicluster",
              "translationInvariance", "shuffle_5_300", "shuffle_3_300", "shuffle_7_300", "shuffle_10_300",
              "shuffle_15_300", "shuffle_5_40", "shuffle_5_100", "shuffle_5_200", "shuffle_5_512"]


def prepare(emb, lang, flag, dataset, oov_pickle_en, oov_pickle_de, sense_pickle_en, sense_pickle_de):
    global sim_func
    global chosen_emb
    global is_oov

    chosen_emb = emb
    is_oov = flag

    if chosen_emb in embeddings:
        if is_oov == "0":
            sim_func = cl_emb.Similarity_Embedding()

        elif is_oov == "1":
            sim_func = cl_emb_oov.Similarity_Embedding_OOV()

    # elif chosen_emb == "uwn":
    #     sim_func = cl_uwn.Similarity_Uwn()

    sim_func.prepare(emb, lang, dataset, oov_pickle_en, oov_pickle_de, sense_pickle_en, sense_pickle_de)


sim_function_names = ['affine',
                      'hamming_dist', 'hamming_sim',
                      'lev_dist', 'lev_sim',
                      'jaro',
                      'jaro_winkler',
                      'needleman_wunsch',
                      'smith_waterman',
                      'overlap_coeff', 'jaccard', 'dice',
                      'monge_elkan', 'cosine',
                      'exact_match', 'rel_diff', 'abs_norm',
                      'crosslang_mean_sim',
                      'crosslang_max_sim',
                      'crosslang_tfidf_mean_sim',
                      'crosslang_tfidf_max_weight',
                      'crosslang_sum_sim',
                      'crosslang_tfidf_sum_sim',
                      'crosslang_sif',
                      'crosslang_uwn_common_sense_weights',
                      'crosslang_uwn_sense_similarity_path',
                      'crosslang_uwn_sense_similarity_lch',
                      'crosslang_uwn_sense_similarity_wup',
                      'crosslang_sim_oov',
                      'crosslang_tfidf_max_sim',
                      'crosslang_number_difference',
                      'crosslang_vector_composition',
                      'crosslang_optimal_alignment',
                      'crosslang_greedy_aligned_words',
                      'crosslang_weighted_greedy_aligned_words',
                      'crosslang_aligned_words_senses_jaccard',
                      'crosslang_weighted_aligned_words_senses_jaccard',
                      'crosslang_aligned_words_senses_path_sim',
                      'crosslang_weighted_aligned_words_senses_path_sim',
                      'crosslang_uwn_sense_similarity_resnik',
                      'crosslang_uwn_sense_similarity_jcn',
                      'crosslang_uwn_sense_similarity_lin',
                      'crosslang_record_linkage_baseline']

"""  String based similarity measures   """


def crosslang_record_linkage_baseline(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.record_linkage_baseline(s1, s2)


def crosslang_uwn_sense_similarity_lin(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.sense_similarity_lin(s1, s2)


def crosslang_uwn_sense_similarity_jcn(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.sense_similarity_jcn(s1, s2)


def crosslang_uwn_sense_similarity_resnik(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.sense_similarity_resnik(s1, s2)


def crosslang_aligned_words_senses_path_sim(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.aligned_words_senses_path_sim(s1, s2)


def crosslang_weighted_aligned_words_senses_path_sim(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.weighted_aligned_words_senses_path_sim(s1, s2)


def crosslang_weighted_aligned_words_senses_jaccard(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.weighted_aligned_words_senses_jaccard(s1, s2)


def crosslang_aligned_words_senses_jaccard(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.aligned_words_senses_jaccard(s1, s2)


def crosslang_weighted_greedy_aligned_words(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.weighted_greedy_aligned_words(s1, s2)


def crosslang_greedy_aligned_words(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.greedy_aligned_words(s1, s2)


def crosslang_optimal_alignment(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.get_optimal_alignment(s1, s2)


def crosslang_vector_composition(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.get_vector_composition(s1, s2)


def crosslang_number_difference(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.get_number_difference(s1, s2)


def crosslang_sim_oov(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.get_oov_jaccard_sim(s1, s2)


def crosslang_uwn_sense_similarity_path(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.sense_similarity_path(s1, s2)


def crosslang_uwn_sense_similarity_lch(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.sense_similarity_lch(s1, s2)


def crosslang_uwn_sense_similarity_wup(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.sense_similarity_wup(s1, s2)


def crosslang_uwn_common_sense_weights(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.common_sense_weights(s1, s2)


def crosslang_sif(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.get_sif(s1, s2)


def crosslang_sum_sim(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.get_sum_sim(s1, s2)


def crosslang_tfidf_sum_sim(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.get_tfidf_sum_sim(s1, s2)


def crosslang_tfidf_max_sim(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.get_tfidf_max_sim(s1, s2)


def crosslang_tfidf_max_weight(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.get_tfidf_max_weight(s1, s2)


def crosslang_tfidf_mean_sim(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.get_tfidf_mean_sim(s1, s2)


def crosslang_mean_sim(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.get_mean_sim(s1, s2)


def crosslang_max_sim(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    return sim_func.get_max_sim(s1, s2)


''' Below part includes Magellan features and the architecture '''


def affine(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    measure = sm.Affine()
    return measure.get_raw_score(s1, s2)


def hamming_dist(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    measure = sm.HammingDistance()
    return measure.get_raw_score(s1, s2)


def hamming_sim(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    measure = sm.HammingDistance()
    return measure.get_sim_score(s1, s2)


def lev_dist(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    measure = sm.Levenshtein()
    return measure.get_raw_score(s1, s2)


def lev_sim(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    measure = sm.Levenshtein()
    return measure.get_sim_score(s1, s2)


def jaro(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    measure = sm.Jaro()
    return measure.get_raw_score(s1, s2)


def jaro_winkler(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    measure = sm.JaroWinkler()
    return measure.get_raw_score(s1, s2)


def needleman_wunsch(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    measure = sm.NeedlemanWunsch()
    return measure.get_raw_score(s1, s2)


def smith_waterman(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    s1 = helper.convert_to_str_unicode(s1)
    s2 = helper.convert_to_str_unicode(s2)

    measure = sm.SmithWaterman()
    return measure.get_raw_score(s1, s2)


"""  Token based measure """


def jaccard(arr1, arr2):
    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN
    # Create jaccard measure object
    measure = sm.Jaccard()
    # Call a function to compute a similarity score
    return measure.get_raw_score(arr1, arr2)


def cosine(arr1, arr2):
    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN
    # Create cosine measure object
    measure = sm.Cosine()
    # Call the function to compute the cosine measure.
    return measure.get_raw_score(arr1, arr2)


def overlap_coeff(arr1, arr2):
    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN
    # Create overlap coefficient measure object
    measure = sm.OverlapCoefficient()
    # Call the function to return the overlap coefficient
    return measure.get_raw_score(arr1, arr2)


def dice(arr1, arr2):
    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN

    # Create Dice object
    measure = sm.Dice()
    # Call the function to return the dice score
    return measure.get_raw_score(arr1, arr2)


"""  Hybrid based measure   """


def monge_elkan(arr1, arr2):
    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN
    # Create Monge-Elkan measure object
    measure = sm.MongeElkan()
    # Call the function to compute the Monge-Elkan measure
    return measure.get_raw_score(arr1, arr2)


"""  Exact match   """


def exact_match(s1, s2):
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    return bool(s1 == s2)


"""  Numeric similarity measures   """


# Relative difference
def rel_diff(d1, d2):
    if d1 is None or d2 is None:
        return pd.np.NaN
    if pd.isnull(d1) or pd.isnull(d2):
        return pd.np.NaN
    try:
        d1 = float(d1)
        d2 = float(d2)
    except ValueError:
        return pd.np.NaN
    if d1 == 0.0 and d2 == 0.0:
        return 0
    else:
        x = (2 * abs(d1 - d2)) / (d1 + d2)
        return x


# Absolute norm similarity
def abs_norm(d1, d2):
    if d1 is None or d2 is None:
        return pd.np.NaN
    if pd.isnull(d1) or pd.isnull(d2):
        return pd.np.NaN
    try:
        d1 = float(d1)
        d2 = float(d2)
    except ValueError:
        return pd.np.NaN
    if d1 == 0.0 and d2 == 0.0:
        return 0
    else:
        x = (abs(d1 - d2) / max(d1, d2))
        if x <= 10e-5:
            x = 0
        return 1.0 - x


# Tokenizers

# q-gram tokenizer
def tok_qgram(input_string, q):
    if pd.isnull(input_string):
        return pd.np.NaN

    measure = sm.QgramTokenizer(qval=q)
    return measure.tokenize(input_string)


def tok_delim(input_string, d):
    if pd.isnull(input_string):
        return pd.np.NaN

    measure = sm.DelimiterTokenizer(delim_set=[d])
    return measure.tokenize(input_string)


def tok_wspace(input_string):
    if pd.isnull(input_string):
        return pd.np.NaN

    measure = sm.WhitespaceTokenizer()
    return measure.tokenize(input_string)


# get look up table to generate readable feature descriptions
def _get_feature_name_lkp_tbl():
    # Initialize a lookup table
    lookup_table = dict()

    # Map features to more human readable descriptions
    lookup_table['lev_dist'] = 'Levenshtein Distance'
    lookup_table['lev_sim'] = 'Levenshtein Similarity'
    lookup_table['jaro'] = 'Jaro Distance'
    lookup_table['jaro_winkler'] = 'Jaro-Winkler Distance'
    lookup_table['exact_match'] = 'Exact Match'
    lookup_table['needleman_wunsch'] = 'Needleman-Wunsch Algorithm'
    lookup_table['smith_waterman'] = 'Smith-Waterman Algorithm'
    lookup_table['abs_norm'] = 'Absolute Norm'
    lookup_table['jaccard'] = 'Jaccard Similarity'
    lookup_table['monge_elkan'] = 'Monge-Elkan Algorithm'
    lookup_table['cosine'] = 'Cosine Similarity'
    lookup_table['qgm_1'] = "1-grams"
    lookup_table['qgm_2'] = "2-grams"
    lookup_table['qgm_3'] = "3-grams"
    lookup_table['qgm_4'] = "4-grams"
    lookup_table['dlm_dc0'] = 'Space Delimiter'
    lookup_table['dlm_wsp'] = 'Whitespace Delimiter'
    lookup_table['N/A'] = 'Not Applicable: Types do not match'
    lookup_table['crosslang_mean_sim'] = 'Cross-Language Embedding Mean Similarity'
    lookup_table['crosslang_max_sim'] = 'Cross-Language Embedding Max Similarity of words'
    lookup_table['crosslang_tfidf_mean_sim'] = 'Cross-Language Embedding tfidf Similarity'
    lookup_table['crosslang_tfidf_max_weight'] = 'Cross-Language Embedding tfidf Important Word Similarity on name'
    lookup_table['crosslang_sum_sim'] = 'Cross-Language Sum of Word Embeddings'
    lookup_table['crosslang_tfidf_sum_sim'] = 'Cross-Language Embedding tfidf Sum of Vectors on name'
    lookup_table['crosslang_sif'] = 'Cross-Language SIF smoothing'
    lookup_table['crosslang_uwn_common_sense_weights'] = 'Cross-language UWN Common Sense Weights'
    lookup_table['crosslang_uwn_sense_similarity_path'] = 'Cross-Language UWN Sense Path Similarity '
    lookup_table['crosslang_uwn_sense_similarity_lch'] = 'Cross-Language UWN Sense Lch Similarity'
    lookup_table['crosslang_uwn_sense_similarity_wup'] = 'Cross-Language UWN Sense Wup Similarity'
    lookup_table['crosslang_sim_oov'] = 'Cross-Language OOV Similarity'
    lookup_table['crosslang_tfidf_max_sim'] = 'Cross-Language Embedding tfidf Max Similarity'
    lookup_table['crosslang_number_difference'] = 'Number Difference in for the Numbers that are Strings'
    lookup_table['crosslang_vector_composition'] = 'Cross-Language Vector Composition Similarity'
    lookup_table['crosslang_optimal_alignment'] = 'Cross-Language Optimal Alignment Score'
    lookup_table['crosslang_greedy_aligned_words'] = 'Cross-Language Greedy Aligned Words'
    lookup_table['crosslang_weighted_greedy_aligned_words'] = 'Cross-Language Weighted Greedy Aligned Words'
    lookup_table[
        'crosslang_aligned_words_senses_jaccard'] = 'Cross-Language Aligned Words Senses Jaccard List Similarity'
    lookup_table[
        'crosslang_weighted_aligned_words_senses_jaccard'] = 'Cross-Language Weighted Aligned Words Senses Jaccard List Similarity'
    lookup_table['crosslang_aligned_words_senses_path_sim'] = 'Cross-Language Aligned Words Senses Path Similarity'
    lookup_table[
        'crosslang_weighted_aligned_words_senses_path_sim'] = 'Cross-Language Weighted Word Senses Path Similarity'
    lookup_table['crosslang_uwn_sense_similarity_resnik'] = 'Cross-Language Sense UWN Resnik Similarity'
    lookup_table['crosslang_uwn_sense_similarity_jcn'] = 'Cross-Language Sense UWN Jcn Similarity '
    lookup_table['crosslang_uwn_sense_similarity_lin'] = 'Cross-Language Sense UWN Lin Similarity'
    lookup_table['crosslang_record_linkage_baseline'] = 'Cross-Language Record Linkage Baseline'
    return lookup_table


# get look up table to generate features
def _get_feat_lkp_tbl():
    """
    This function embeds the knowledge of mapping what features to be
    generated for what kind of attr. types.

    """

    global chosen_emb
    global is_oov

    uwn_features = [('crosslang_uwn_common_sense_weights'),
                    ('crosslang_uwn_sense_similarity_path'),
                    ('crosslang_uwn_sense_similarity_lch'),
                    ('crosslang_uwn_sense_similarity_wup'),
                    ('crosslang_uwn_sense_similarity_resnik'),
                    ('crosslang_uwn_sense_similarity_jcn'),
                    ('crosslang_uwn_sense_similarity_lin')]

    oov_features = [('crosslang_sim_oov'),
                    ('crosslang_number_difference')]

    embedding_features = [('crosslang_mean_sim'),  # 0
                          ('crosslang_max_sim'),  # 1
                          ('crosslang_tfidf_mean_sim'),  # 3
                          ('crosslang_tfidf_max_sim'),  # 5
                          ('crosslang_tfidf_max_weight'),  # 6
                          ('crosslang_sif'),  # 7
                          ('crosslang_vector_composition'),  # 8
                          ('crosslang_optimal_alignment'),  # 9
                          ('crosslang_greedy_aligned_words'),  # 10
                          ('crosslang_weighted_greedy_aligned_words'),  # 11
                          ('crosslang_aligned_words_senses_jaccard'),  # 12
                          ('crosslang_weighted_aligned_words_senses_jaccard'),  # 13
                          ('crosslang_aligned_words_senses_path_sim'),  # 14
                          ('crosslang_weighted_aligned_words_senses_path_sim')  # 15
                          ]

    if chosen_emb in embeddings:
        if is_oov == "0":
            selected_features = embedding_features

        elif is_oov == "1":
            selected_features = embedding_features + oov_features

    elif chosen_emb == "uwn":
        selected_features = uwn_features

    #selected_features = [('crosslang_record_linkage_baseline')]

    # Initialize a lookup table
    lookup_table = dict()

    # Features for type str_eq_1w
    lookup_table['STR_EQ_1W'] = selected_features

    # ('lev_dist'), ('lev_sim'), ('jaro'),
    # ('jaro_winkler'),
    # ('exact_match'),
    # ('jaccard', 'qgm_3', 'qgm_3')

    # Features for type str_bt_1w_5w
    lookup_table['STR_BT_1W_5W'] = selected_features

    # [('jaccard', 'qgm_3', 'qgm_3'),
    #  ('cosine', 'dlm_dc0', 'dlm_dc0'),
    #  ('jaccard', 'dlm_dc0', 'dlm_dc0'),
    #  ('monge_elkan'), ('lev_dist'), ('lev_sim'),
    #  ('needleman_wunsch'),
    #  ('smith_waterman')]

    # Features for type str_bt_5w_10w
    lookup_table['STR_BT_5W_10W'] = selected_features

    # [('jaccard', 'qgm_3', 'qgm_3'),
    #  ('cosine', 'dlm_dc0', 'dlm_dc0'),
    #  ('monge_elkan'), ('lev_dist'), ('lev_sim')]

    # Features for type str_gt_10w
    lookup_table['STR_GT_10W'] = selected_features

    # [('jaccard', 'qgm_3', 'qgm_3'),
    #  ('cosine', 'dlm_dc0', 'dlm_dc0')]

    # Features for NUMERIC type
    lookup_table['NUM'] = [('exact_match'), ('abs_norm'), ('lev_dist'),
                           ('lev_sim')]

    # Features for BOOLEAN type
    lookup_table['BOOL'] = [('exact_match')]

    # Features for un determined type
    lookup_table['UN_DETERMINED'] = []

    # Finally, return the lookup table
    return lookup_table


def _get_type(column):
    """
     Given a pandas Series (i.e column in pandas DataFrame) obtain its type
    """
    # Validate input parameters
    # # We expect the input column to be of type pandas Series

    if not isinstance(column, pd.Series):
        raise AssertionError('Input (column) is not of type pandas series')

    # To get the type first drop all NaNa
    column = column.dropna()

    # Get type for each element and convert it into a set (and for
    # convenience convert the resulting set into a list)
    type_list = list(set(column.map(type).tolist()))

    # If the list is empty, then we cannot decide anything about the column.
    # We will raise a warning and return the type to be numeric.
    # Note: The reason numeric is returned instead of a special type because,
    #  we want to keep the types minimal. Further, explicitly recommend the
    # user to update the returned types later.
    if len(type_list) == 0:
        return 'un_determined'

    # If the column qualifies to be of more than one type (for instance,
    # in a numeric column, some values may be inferred as strings), then we
    # will raise an error for the user to fix this case.
    if len(type_list) > 1:
        raise AssertionError('Column %s qualifies to be more than one type. \n'
                             'Please explicitly set the column type like this:\n'
                             'A["address"] = A["address"].astype(str) \n'
                             'Similarly use int, float, boolean types.' % column.name)
    else:
        # the number of types is 1.
        returned_type = type_list[0]
        # Check if the type is boolean, if so return boolean
        if returned_type == bool or returned_type == pd.np.bool_:
            return 'boolean'

        # Check if the type is string, if so identify the subtype under it.
        # We use average token length to identify the subtypes

        # Consider string and unicode as same
        elif returned_type == str or returned_type == six.unichr or returned_type == six.text_type:
            # get average token length
            average_token_len = \
                pd.Series.mean(column.str.split().apply(_len_handle_nan))
            if average_token_len == 1:
                return "str_eq_1w"
            elif average_token_len <= 5:
                return "str_bt_1w_5w"
            elif average_token_len <= 10:
                return "str_bt_5w_10w"
            else:
                return "str_gt_10w"
        else:
            # Finally, return numeric if it does not qualify for any of the
            # types above.
            return "numeric"


def get_attr_types(data_frame):
    # We expect the input object (data_frame) to be of type pandas DataFrame.
    if not isinstance(data_frame, pd.DataFrame):
        raise AssertionError('Input table is not of type pandas dataframe')

    # Now get type for each column
    type_list = [_get_type(data_frame[col]) for col in data_frame.columns]

    # Create a dictionary containing attribute types
    attribute_type_dict = dict(zip(data_frame.columns, type_list))

    # Return the attribute type dictionary
    return attribute_type_dict


def _len_handle_nan(input_list):
    """
     Get the length of list, handling NaN
    """
    # Check if the input is of type list, if so return the len else return NaN
    if isinstance(input_list, list):
        return len(input_list)
    else:
        return pd.np.NaN


def get_sim_funs():
    """
    This function returns all the similarity functions supported by py_entitymatching.

    """
    # Get all the functions
    functions = [affine,
                 hamming_dist, hamming_sim,
                 lev_dist, lev_sim,
                 jaro,
                 jaro_winkler,
                 needleman_wunsch,
                 smith_waterman,
                 overlap_coeff, jaccard, dice,
                 monge_elkan, cosine,
                 exact_match, rel_diff, abs_norm,
                 crosslang_mean_sim,
                 crosslang_max_sim,
                 crosslang_tfidf_mean_sim,
                 crosslang_tfidf_max_weight,
                 crosslang_sum_sim,
                 crosslang_tfidf_sum_sim,
                 crosslang_sif,
                 crosslang_uwn_common_sense_weights,
                 crosslang_uwn_sense_similarity_path,
                 crosslang_uwn_sense_similarity_lch,
                 crosslang_uwn_sense_similarity_wup,
                 crosslang_sim_oov,
                 crosslang_tfidf_max_sim,
                 crosslang_number_difference,
                 crosslang_vector_composition,
                 crosslang_optimal_alignment,
                 crosslang_greedy_aligned_words,
                 crosslang_weighted_greedy_aligned_words,
                 crosslang_aligned_words_senses_jaccard,
                 crosslang_weighted_aligned_words_senses_jaccard,
                 crosslang_aligned_words_senses_path_sim,
                 crosslang_weighted_aligned_words_senses_path_sim,
                 crosslang_uwn_sense_similarity_resnik,
                 crosslang_uwn_sense_similarity_jcn,
                 crosslang_uwn_sense_similarity_lin,
                 crosslang_record_linkage_baseline]

    # Return a dictionary with the functions names as the key and the actual
    # functions as values.
    return dict(zip(sim_function_names, functions))


def _make_tok_qgram(q):
    """
    This function returns a qgran-based tokenizer with a fixed delimiter
    """

    def tok_qgram(s):
        # check if the input is of type base string
        if pd.isnull(s):
            return s

        s = helper.convert_to_str_unicode(s)

        measure = sm.QgramTokenizer(qval=q)
        return measure.tokenize(s)

    return tok_qgram


def _make_tok_delim(d):
    """
    This function returns a delimiter-based tokenizer with a fixed delimiter
    """

    def tok_delim(s):
        # check if the input is of type base string
        if pd.isnull(s):
            return s
        # Remove non ascii  characters. Note: This should be fixed in the
        # next version.
        # s = remove_non_ascii(s)

        s = helper.convert_to_str_unicode(s)

        # Initialize the tokenizer measure object
        measure = sm.DelimiterTokenizer(delim_set=[d])
        # Call the function that will tokenize the input string.
        return measure.tokenize(s)

    return tok_delim


def _get_single_arg_tokenizers(q=[2, 3], dlm_char=[' ']):
    """
    This function creates single argument tokenizers for the given input
    parameters.
    """
    # Validate the input parameters
    if q is None and dlm_char is None:
        print('Both q and dlm_char cannot be null')
        raise AssertionError('Both q and dlm_char cannot be null')
    # Initialize the key (function names) and value dictionaries (tokenizer
    # functions).
    names = []
    functions = []

    if q is not None:
        if not isinstance(q, list):
            q = [q]

        # Create a qgram function for the given list of q's
        qgm_fn_list = [_make_tok_qgram(k) for k in q]
        qgm_names = ['qgm_' + str(x) for x in q]
        # Update the tokenizer name, function lists
        names.extend(qgm_names)
        functions.extend(qgm_fn_list)

    names.append('wspace')
    functions.append(tok_wspace)

    if dlm_char is not None:
        if not isinstance(dlm_char, list) and isinstance(dlm_char,
                                                         six.string_types):
            dlm_char = [dlm_char]
        # Create a delimiter function for the given list of q's
        dlm_fn_list = [_make_tok_delim(k) for k in dlm_char]

        # Update the tokenizer name, function lists
        dlm_names = ['dlm_dc' + str(i) for i in range(len(dlm_char))]
        names.extend(dlm_names)
        functions.extend(dlm_fn_list)

    if len(names) > 0 and len(functions) > 0:
        return dict(zip(names, functions))
    else:
        print('Didnot create any tokenizers, returning empty dict.')
        return dict()


def _get_type_name_lkp_tbl():
    # Initialize a lookup table
    lookup_table = dict()

    # Map type names to more human readable names
    lookup_table['str_eq_1w'] = 'short string (1 word)'
    lookup_table['str_bt_1w_5w'] = 'short string (1 word to 5 words)'
    lookup_table['str_bt_5w_10w'] = 'medium string (5 words to 10 words)'
    lookup_table['str_gt_10w'] = 'short string (1 word)'
    lookup_table['numeric'] = 'numeric'
    lookup_table['boolean'] = 'boolean'
    lookup_table['un_determined'] = 'un-determined type'

    return lookup_table


def get_tokenizers_for_matching(q=[2, 3], dlm_char=[' ']):
    if q is None and dlm_char is None:
        raise AssertionError('Both q and dlm_char cannot be null')
    else:
        # Return single arg tokenizers for the given inputs.
        return _get_single_arg_tokenizers(q, dlm_char)


def _get_readable_type_name(column_type):
    # First get the look up table
    lookup_table = _get_type_name_lkp_tbl()

    # Check if the column type is in the dictionary
    if column_type in lookup_table:
        return lookup_table[column_type]
    else:
        raise TypeError('Unknown type')


def _get_features_for_type(column_type):
    """
    Get features to be generated for a type
    """
    # First get the look up table
    lookup_table = _get_feat_lkp_tbl()

    # Based on the column type, return the feature functions that should be
    # generated.
    if column_type is 'str_eq_1w':
        features = lookup_table['STR_EQ_1W']
    elif column_type is 'str_bt_1w_5w':
        features = lookup_table['STR_BT_1W_5W']
    elif column_type is 'str_bt_5w_10w':
        features = lookup_table['STR_BT_5W_10W']
    elif column_type is 'str_gt_10w':
        features = lookup_table['STR_GT_10W']
    elif column_type is 'numeric':
        features = lookup_table['NUM']
    elif column_type is 'boolean':
        features = lookup_table['BOOL']
    elif column_type is 'un_determined':
        features = lookup_table['UN_DETERMINED']
    else:
        raise TypeError('Unknown type')
    return features


def _get_readable_feature_name(feature):
    # First get the look up table
    lookup_table = _get_feature_name_lkp_tbl()

    readable_feature = []

    if isinstance(feature, six.string_types):
        # If feature is just a string, return the readable name
        if feature in lookup_table:
            return lookup_table[feature]
        else:
            raise AssertionError('Feature is not present in lookup table')
    elif len(feature) == 3:
        # If feature is a list, get the readable name of each part
        for name in feature:
            # Check if the feature is in the dictionary
            if name in lookup_table:
                readable_feature.append(lookup_table[name])
            else:
                raise AssertionError('Feature is not present in lookup table')
        return readable_feature[0] + ' [' + readable_feature[1] + ', ' + readable_feature[2] + "]"
    else:
        raise AssertionError('Features should have either 0 or 2 (one for each table) tokenizers')


def get_attr_types(data_frame):
    # Validate input paramaters

    # # We expect the input object (data_frame) to be of type pandas DataFrame.
    if not isinstance(data_frame, pd.DataFrame):
        raise AssertionError('Input table is not of type pandas dataframe')

    # Now get type for each column
    type_list = [_get_type(data_frame[col]) for col in data_frame.columns]

    # Create a dictionary containing attribute types
    attribute_type_dict = dict(zip(data_frame.columns, type_list))

    # Update the dictionary with the _table key and value set to the input
    # DataFrame
    attribute_type_dict['_table'] = data_frame

    # Return the attribute type dictionary
    return attribute_type_dict


def get_attr_corres(ltable, rtable):
    correspondence_list = []

    for column in ltable.columns:
        if column in rtable.columns:
            correspondence_list.append((column, column))
    # Initialize a correspondence dictionary.
    correspondence_dict = dict()
    # Fill the corres, ltable and rtable.
    correspondence_dict['corres'] = correspondence_list
    correspondence_dict['ltable'] = ltable
    correspondence_dict['rtable'] = rtable
    # Finally, return the correspondence dictionary
    return correspondence_dict


def flatten_list(inp_list):
    return [item for sublist in inp_list for item in sublist]


# fill function template
def fill_fn_template(attr1, attr2, sim_func, tok_func_1=None, tok_func_2=None):
    # construct function string
    s = 'from ' + name_of_class + ' import *\nfrom ' + name_of_class + ' import *\n'
    # get the function name
    fn_name = get_fn_name(attr1, attr2, sim_func, tok_func_1, tok_func_2)
    # proceed with function construction
    fn_st = 'def ' + fn_name + '(ltuple, rtuple):'
    s += fn_st
    s += '\n'

    # add 4 spaces
    s += '    '
    fn_body = 'return '
    if tok_func_1 is not None and tok_func_2 is not None:
        fn_body = fn_body + sim_func + '(' + tok_func_1 + '(' + 'ltuple["' + attr1 + '"]'
        fn_body += '), '
        fn_body = fn_body + tok_func_2 + '(' + 'rtuple["' + attr2 + '"]'
        fn_body = fn_body + ')) '
    else:
        fn_body = fn_body + sim_func + '(' + 'ltuple["' + attr1 + '"], rtuple["' + attr2 + '"])'
    s += fn_body

    return fn_name, attr1, attr2, tok_func_1, tok_func_2, sim_func, s


def get_fn_str(inp, attrs):
    if inp:
        args = []
        args.extend(attrs)
        if isinstance(inp, six.string_types) == True:
            inp = [inp]
        args.extend(inp)
        # fill function string from a template
        return fill_fn_template(*args)
    else:
        return None


def get_fn_name(attr1, attr2, sim_func, tok_func_1=None, tok_func_2=None):
    attr1 = '_'.join(attr1.split())
    attr2 = '_'.join(attr2.split())
    fp = '_'.join([attr1, attr2])
    name_lkp = dict()
    name_lkp["jaccard"] = "jac"
    name_lkp["lev_dist"] = "lev_dist"
    name_lkp["lev_sim"] = "lev_sim"
    name_lkp["cosine"] = "cos"
    name_lkp["monge_elkan"] = "mel"
    name_lkp["needleman_wunsch"] = "nmw"
    name_lkp["smith_waterman"] = "sw"
    name_lkp["jaro"] = "jar"
    name_lkp["jaro_winkler"] = "jwn"
    name_lkp["exact_match"] = "exm"
    name_lkp["abs_norm"] = "anm"
    name_lkp["rel_diff"] = "rdf"
    name_lkp["1"] = "1"
    name_lkp["2"] = "2"
    name_lkp["3"] = "3"
    name_lkp["4"] = "4"
    name_lkp["tok_whitespace"] = "wsp"
    name_lkp["tok_qgram"] = "qgm"
    name_lkp["tok_delim"] = "dlm"
    name_lkp["crosslang_mean_sim"] = "crosslang_mean_sim"
    name_lkp["crosslang_max_sim"] = "crosslang_max_sim"
    name_lkp["crosslang_tfidf_mean_sim"] = "crosslang_tfidf_mean_sim"
    name_lkp["crosslang_tfidf_max_weight"] = "crosslang_tfidf_max_weight"
    name_lkp["crosslang_sum_sim"] = "crosslang_sum_sim"
    name_lkp["crosslang_tfidf_sum_sim"] = "crosslang_tfidf_sum_sim"
    name_lkp["crosslang_sif"] = "crosslang_sif"
    name_lkp["crosslang_uwn_common_sense_weights"] = "crosslang_uwn_common_sense_weights"
    name_lkp["crosslang_uwn_sense_similarity_path"] = "crosslang_uwn_sense_similarity_path"
    name_lkp["crosslang_uwn_sense_similarity_lch"] = "crosslang_uwn_sense_similarity_lch"
    name_lkp["crosslang_uwn_sense_similarity_wup"] = "crosslang_uwn_sense_similarity_wup"
    name_lkp["crosslang_sim_oov"] = "crosslang_sim_oov"
    name_lkp["crosslang_tfidf_max_sim"] = "crosslang_tfidf_max_sim"
    name_lkp["crosslang_number_difference"] = "crosslang_number_difference"
    name_lkp["crosslang_vector_composition"] = "crosslang_vector_composition"
    name_lkp["crosslang_optimal_alignment"] = "crosslang_optimal_alignment"
    name_lkp["crosslang_greedy_aligned_words"] = "crosslang_greedy_aligned_words"
    name_lkp["crosslang_weighted_greedy_aligned_words"] = "crosslang_weighted_greedy_aligned_words"
    name_lkp["crosslang_aligned_words_senses_jaccard"] = "crosslang_aligned_words_senses_jaccard"
    name_lkp["crosslang_weighted_aligned_words_senses_jaccard"] = "crosslang_weighted_aligned_words_senses_jaccard"
    name_lkp["crosslang_aligned_words_senses_path_sim"] = "crosslang_aligned_words_senses_path_sim"
    name_lkp["crosslang_weighted_aligned_words_senses_path_sim"] = "crosslang_weighted_aligned_words_senses_path_sim"
    name_lkp["crosslang_uwn_sense_similarity_resnik"] = "crosslang_uwn_sense_similarity_resnik"
    name_lkp["crosslang_uwn_sense_similarity_jcn"] = "crosslang_uwn_sense_similarity_jcn"
    name_lkp["crosslang_uwn_sense_similarity_lin"] = "crosslang_uwn_sense_similarity_lin"
    name_lkp["crosslang_record_linkage_baseline"] = "crosslang_record_linkage_baseline"

    arg_list = [sim_func, tok_func_1, tok_func_2]
    nm_list = [name_lkp.get(tok, tok) for tok in arg_list if tok]
    sp = '_'.join(nm_list)
    return '_'.join([fp, sp])


# conv function string to function object and return with meta data
def conv_fn_str_to_obj(fn_tup, tok, sim_funcs):
    d_orig = {}
    d_orig.update(tok)
    d_orig.update(sim_funcs)
    d_ret_list = []
    for f in fn_tup:
        d_ret = {}
        name = f[0]
        attr1 = f[1]
        attr2 = f[2]
        tok_1 = f[3]
        tok_2 = f[4]
        simfunction = f[5]
        # exec(f[6] in d_orig)
        six.exec_(f[6], d_orig)
        d_ret['function'] = d_orig[name]
        d_ret['feature_name'] = name
        d_ret['left_attribute'] = attr1
        d_ret['right_attribute'] = attr2
        d_ret['left_attr_tokenizer'] = tok_1
        d_ret['right_attr_tokenizer'] = tok_2
        d_ret['simfunction'] = simfunction
        d_ret['function_source'] = f[6]

        d_ret_list.append(d_ret)

    return d_ret_list


# convert features from look up table to function objects
def _conv_func_objs(features, attributes,
                    tokenizer_functions, similarity_functions):
    """
    Convert features from look up table to function objects
    """

    # # First get the tokenizer and similarity functions list.
    tokenizer_list = tokenizer_functions.keys()
    similarity_functions_list = similarity_functions.keys()

    function_tuples = [get_fn_str(input, attributes) for input in features]

    # Convert the function string into a function object
    function_objects = conv_fn_str_to_obj(function_tuples, tokenizer_functions,
                                          similarity_functions)

    return function_objects


def get_features(ltable, rtable, l_attr_types, r_attr_types,
                 attr_corres, tok_funcs, sim_funcs):
    # Initialize output feature dictionary list
    feature_dict_list = []

    # Generate features for each attr. correspondence
    for ac in attr_corres['corres']:
        l_attr_type = l_attr_types[ac[0]]
        r_attr_type = r_attr_types[ac[1]]

        # Generate a feature only if the attribute types are same
        if l_attr_type != r_attr_type:
            print('py_entitymatching types: %s type (%s) and %s type (%s) '
                  'are different.'
                  'If you want to set them to be same and '
                  'generate features, '
                  'update output from get_attr_types and '
                  'use get_features command.\n.'
                  % (ac[0], l_attr_type, ac[1], r_attr_type))
            continue

        features = _get_features_for_type(l_attr_type)

        # Convert features to function objects
        fn_objs = _conv_func_objs(features, ac, tok_funcs, sim_funcs)
        # Add the function object to a feature list.
        feature_dict_list.append(fn_objs)

    # Create a feature table
    feature_table = pd.DataFrame(flatten_list(feature_dict_list))
    # Project out only the necessary columns.
    feature_table = feature_table[['feature_name', 'left_attribute',
                                   'right_attribute', 'left_attr_tokenizer',
                                   'right_attr_tokenizer',
                                   'simfunction', 'function',
                                   'function_source']]
    # Return the feature table.
    return feature_table


def get_features_for_matching(embedding, lang, flag, ltable, rtable, dataset, oov_pickle_en, oov_pickle_de,
                              sense_pickle_en,
                              sense_pickle_de):
    prepare(embedding, lang, flag, dataset, oov_pickle_en, oov_pickle_de, sense_pickle_en, sense_pickle_de)

    # Get similarity functions for generating the features for matching
    sim_funcs = get_sim_funs()
    # Get tokenizer functions for generating the features for matching
    tok_funcs = get_tokenizers_for_matching()

    # Get the attribute types of the input tables
    attr_types_ltable = get_attr_types(ltable)
    attr_types_rtable = get_attr_types(rtable)

    # Get the attribute correspondence between the input tables
    attr_corres = get_attr_corres(ltable, rtable)

    # Get the features
    feature_table = get_features(ltable, rtable, attr_types_ltable,
                                 attr_types_rtable, attr_corres,
                                 tok_funcs, sim_funcs)

    # Finally return the feature table
    return feature_table


########################################### Extract Features  #########################


def extract_feature_vecs(candset, ltable, rtable, feature_table,
                         show_progress=True, n_jobs=1):
    key = "_id"
    l_key = "id"
    r_key = "id"
    fk_rtable = "rtable_id"
    fk_ltable = "ltable_id"

    # Extract features
    # # Set index for convenience
    l_df = ltable.set_index(l_key, drop=False)
    r_df = rtable.set_index(r_key, drop=False)

    # # Apply feature functions
    col_names = list(candset.columns)

    fk_ltable_idx = col_names.index(fk_ltable)  # 1
    fk_rtable_idx = col_names.index(fk_rtable)  # 2

    n_procs = get_num_procs(n_jobs, len(candset))
    c_splits = pd.np.array_split(candset, n_procs)

    pickled_obj = cloudpickle.dumps(feature_table)

    feat_vals_by_splits = Parallel(n_jobs=n_procs)(delayed(get_feature_vals_by_cand_split)(pickled_obj,
                                                                                           fk_ltable_idx,
                                                                                           fk_rtable_idx,
                                                                                           l_df, r_df,
                                                                                           c_splits[i],
                                                                                           show_progress and i == len(
                                                                                               c_splits) - 1)
                                                   for i in range(len(c_splits)))

    feat_vals = sum(feat_vals_by_splits, [])

    # Construct output table
    feature_vectors = pd.DataFrame(feat_vals, index=candset.index.values)
    # # Rearrange the feature names in the input feature table order
    feature_names = list(feature_table['feature_name'])
    feature_vectors = feature_vectors[feature_names]

    # # Insert keys
    feature_vectors.insert(0, fk_rtable, candset[fk_rtable])
    feature_vectors.insert(0, fk_ltable, candset[fk_ltable])
    feature_vectors.insert(0, key, candset[key])

    return feature_vectors


def get_feature_vals_by_cand_split(pickled_obj, fk_ltable_idx, fk_rtable_idx, l_df, r_df, candsplit, show_progress):
    feature_table = cloudpickle.loads(pickled_obj)
    if show_progress:
        prog_bar = pyprind.ProgBar(len(candsplit))

    l_dict = {}
    r_dict = {}

    feat_vals = []
    for row in candsplit.itertuples(index=False):
        if show_progress:
            prog_bar.update()

        fk_ltable_val = row[fk_ltable_idx]
        fk_rtable_val = row[fk_rtable_idx]

        if fk_ltable_val not in l_dict:
            l_dict[fk_ltable_val] = l_df.ix[fk_ltable_val]
        l_tuple = l_dict[fk_ltable_val]

        if fk_rtable_val not in r_dict:
            r_dict[fk_rtable_val] = r_df.ix[fk_rtable_val]
        r_tuple = r_dict[fk_rtable_val]

        f = apply_feat_fns(l_tuple, r_tuple, feature_table)
        feat_vals.append(f)

    return feat_vals


def apply_feat_fns(tuple1, tuple2, feat_dict):
    """
    Apply feature functions to two tuples.
    """
    # Get the feature names
    feat_names = list(feat_dict['feature_name'])
    # Get the feature functions
    feat_funcs = list(feat_dict['function'])
    # Compute the feature value by applying the feature function to the input
    #  tuples.
    feat_vals = [f(tuple1, tuple2) for f in feat_funcs]
    # Return a dictionary where the keys are the feature names and the values
    #  are the feature values.
    return dict(zip(feat_names, feat_vals))


def get_num_procs(n_jobs, min_procs):
    # determine number of processes to launch parallely
    n_cpus = multiprocessing.cpu_count()
    n_procs = n_jobs
    if n_jobs < 0:
        n_procs = n_cpus + 1 + n_jobs
    # cannot launch less than min_procs to safeguard against small tables
    return min(n_procs, min_procs)
