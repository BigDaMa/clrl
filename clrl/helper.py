import six
import csv
import re
import pandas as pd




def convert_to_str_unicode(input_string):
    if not isinstance(input_string, six.string_types):
        input_string = six.u(str(input_string))

    if isinstance(input_string, bytes):
        input_string = input_string.decode('utf-8', 'ignore')

    input_string = re.sub(r'[^\w\s]',' ', input_string)
    return input_string


def cartesian(df1, df2):
    A = df1.rename(columns=lambda x: "l_" + x)
    B = df2.rename(columns=lambda x: "r_" + x)

    A['foo'] = 1
    B['foo'] = 1
    C = pd.merge(A, B, on='foo').drop('foo', axis=1)
    C['_id'] = range(1, len(C) + 1)
    start_columns = ["_id", "l_id", "r_id"]
    col = start_columns + [c for c in list(C) if c not in start_columns]
    C = C[col]

    return C


# def labelSample(sample_df, dup_file):
#     dup_dict = {}
#
#     dup_df = pd.read_csv(dup_file)
#
#     for index, row in dup_df.iterrows():
#         dup_dict[row['English ID']] = row['German ID']
#
#     print("There are " + str(len(dup_dict)) + " duplicates originally")
#
#     cols = list(sample_df) + ["Label"]
#     labeled_df = pd.DataFrame(columns=cols)
#
#     count = 0
#     for index, row in sample_df.iterrows():
#         if row["ltable_id"] in dup_dict and dup_dict[row["ltable_id"]] == row["rtable_id"]:
#             count += 1
#             labeled_df.loc[index] = row.tolist() + ["1"]
#         else:
#             labeled_df.loc[index] = row.tolist() + ["0"]
#
#     print("Number of preserved duplicates: " + str(count))
#     return labeled_df
#
#
# def extract_labels(df, dupfile, df_dup):
#     dup_dict = {}
#
#     with open(dupfile, 'r') as f:
#         freader = csv.reader(f)
#
#         headertemp = next(freader)
#         for row in freader:
#             dup_dict[int(row[0])] = int(row[1])
#
#     f.close()
#
#     df = df.assign(label='0')
#
#     for index, row in df.iterrows():
#         left = row['ltable_id']
#         right = row['rtable_id']
#         if left in dup_dict and dup_dict[left] == right:
#             df.at[index, 'Label'] = '1'
#         else:
#             df.at[index, 'Label'] = '0'
#
#     df.to_csv(df_dup, index=False)
