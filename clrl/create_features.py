'''
Creates features for word embeddings

:param:
    dataset (uni, movie, title)
    lang    (de, es, fr)
    object  (embedding object) (see below)
    flag (0, 1)  - If 1, then features with OOV treatment is used
'''

import sys
import warnings
import pandas as pd
import crosslang_feature_extraction as ft

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def main(argv):
    topic = argv[0]
    lang = argv[1]
    embedding = argv[2]
    flag = argv[3]

    embeddings = ["fasttext", "babylon", "fbmuse", "multicca", "multiskip", "multicluster",
                  "translationInvariance", "shuffle_5_300", "shuffle_3_300", "shuffle_7_300", "shuffle_10_300",
                  "shuffle_15_300", "shuffle_5_40", "shuffle_5_100", "shuffle_5_200", "shuffle_5_512"]

    # --------------------------------------
    if topic == "uni":
        path = "/home/oyku/datasets/University/"

    elif topic == "movie":
        path = "/home/oyku/datasets/Movie/"

    elif topic == "title":
        path = "/home/oyku/datasets/Article/"

    else:
        print("Wrong dataset is given. It should be either uni, movie or title.")
        return

    # --------------------------------------

    if lang not in ["de", "es", "fr"]:
        print("Wrong language is given. It should be either de, es or fr.")
        return

    # --------------------------------------

    if flag == "0":
        method = embedding

    elif flag == "1":
        method = embedding + "_oov"

    else:
        print("Specify OOV flag as 0 or 1.")
        return

    # --------------------------------------

    if embedding not in embeddings and embedding != "uwn":
        print(
            "Wrong object is given. Either give uwn or one of those embeddings: fasttext, babylon, fbmuse, multicca, \
             multiskip, multicluster, translationInvariance, shuffle_5_300")
        return

    # --------------------------------------

    oov_pickle_path = "/home/oyku/myversion/oov_matches/" + lang + "/"
    sense_pickle_path = "/home/oyku/myversion/vocabulary/uwn/" + lang + "/"
    en_path = path + topic + "_en.csv"
    de_path = path + topic + "_" + lang + ".csv"
    labeled = path + topic + "_" + lang + "_blocked_original.csv"
    oov_pickle_en = oov_pickle_path + topic + "_en.p"
    oov_pickle_de = oov_pickle_path + topic + "_" + lang + ".p"
    sense_pickle_en = sense_pickle_path + topic + "_en.p"
    sense_pickle_de = sense_pickle_path + topic + "_" + lang + ".p"

    en = pd.read_csv(en_path)
    de = pd.read_csv(de_path)
    labeled = pd.read_csv(labeled)
    print(labeled.shape)

    print("Running " + embedding + " on " + topic + " dataset!")

    feature_path = path + "features/" + topic + "_" + lang + "_" + method + "_features.csv"

    feature_table = ft.get_features_for_matching(embedding, lang, flag, en, de, labeled, oov_pickle_en, oov_pickle_de,
                                                 sense_pickle_en,
                                                 sense_pickle_de)
    feature_table = feature_table[feature_table["left_attribute"] != "id"]
    train_features = ft.extract_feature_vecs(candset=labeled, ltable=en, rtable=de, feature_table=feature_table)
    train_features.to_csv(feature_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
