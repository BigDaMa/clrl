
'''
Blocks two datasets A and B and returns a candidate set C.
Blocking is performed on a given attribute, considering word overlap (uses Magellan's implementation).
We use blocking after translation of the given attribute.
It also labels the candidate set which is already obtained 
duplicate information from Wikipedia Interlanguage Links

'''

import py_entitymatching as em
import pandas as pd
import pyprind
import sys


def labelSample(sample_df, dup_file):
    dup_dict = {}

    dup_df = pd.read_csv(dup_file)

    for index, row in dup_df.iterrows():
        dup_dict[row['English ID']] = row['German ID']

    print("There are " + str(len(dup_dict)) + " duplicates originally")

    cols = list(sample_df) + ["Label"]
    labeled_df = pd.DataFrame(columns=cols)

    count = 0
    for index, row in sample_df.iterrows():
        if row["ltable_id"] in dup_dict and dup_dict[row["ltable_id"]] == row["rtable_id"]:
            count += 1
            labeled_df.loc[index] = row.tolist() + ["1"]
        else:
            labeled_df.loc[index] = row.tolist() + ["0"]

    print("Number of preserved duplicates: " + str(count))
    return labeled_df


def main(argv):
    topic = argv[0]
    lang = argv[1]

    flag = False
    excluded = ['_id', 'ltable_id', 'rtable_id', 'Label']

    path = "/home/oyku/datasets/"
    sample_path = path + "sample.csv"

    if topic == "uni":
        path = "/home/oyku/datasets/University/"
        ltable_blocker_field = "city"
        rtable_blocker_field = "city"

    elif topic == "movie":
        path = "/home/oyku/datasets/Movie/"
        ltable_blocker_field = "director"
        rtable_blocker_field = "director"

    elif topic == "title":
        path = "/home/oyku/datasets/Article/"
        ltable_blocker_field = "category"
        rtable_blocker_field = "category"

    en_csv = path + topic + "_en.csv"
    de_csv = path + topic + "_" + lang + ".csv"
    trans_de_csv = path + topic + "_" + lang + "_translated.csv"
    labeled_path = path + topic + "_" + lang + "_blocked_original.csv"
    tr_labeled_path = path + topic + "_" + lang + "_blocked_translated.csv"
    dup_file = path + topic + "_" + lang + "_duplicates.csv"

    labeled_df = pd.read_csv(labeled_path)
    tr_labeled_df = pd.read_csv(tr_labeled_path)

    features = path + "features/" + topic + "_" + lang + "_magellan_features.csv"
    tr_features = path + "features/" + topic + "_" + lang + "_translated_magellan_features.csv"

    tr_ft = pd.read_csv(tr_features)
    ft = pd.read_csv(features)

    A = em.read_csv_metadata(en_csv, key='id')
    B = em.read_csv_metadata(de_csv, key='id')
    T = em.read_csv_metadata(trans_de_csv, key='id')

    headerA = list(A)
    headerB = list(B)

    translated = True
    if not translated:
        rtable_blocker_field = 't_' + rtable_blocker_field
        T.rename(columns={ltable_blocker_field: rtable_blocker_field}, inplace=True)
        B = B.merge(T[['id', rtable_blocker_field]], on='id')
        em.set_key(B, 'id')
    else:
        B = T
        headerB = list(B)

    ob = em.OverlapBlocker()
    C = ob.block_tables(A, B, ltable_blocker_field, rtable_blocker_field, l_output_attrs=headerA,
                        r_output_attrs=headerB, word_level=True, overlap_size=2, rem_stop_words=False,
                        allow_missing=flag)
    print("Shape of sampled: {}".format(C.shape))

    labeled_df = labelSample(C, dup_file)

    if translated:
        labeled_df.to_csv(tr_labeled_path, index=False)
    else:
        labeled_df.to_csv(labeled_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
