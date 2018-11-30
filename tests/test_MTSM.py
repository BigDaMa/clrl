'''
To test MT+SM baseline for all datasets in all languages

'''

import sys
import datetime
import warnings
import pandas as pd

from sklearn import model_selection
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def main(argv):

    topics = ["uni", "movie", "title"]
    langs = ["de", "es", "fr"]

    result_path = "/home/oyku/datasets/newexperiments/experiment_0/mtsm_baseline.csv"
    cols = ['Topic', 'Lang', 'F1', 'Recall', 'Precision']
    df = pd.DataFrame(columns=cols)

    for topic in topics:
        for lang in langs:

            # --------------------------------------
            if topic == "uni":
                path = "/home/oyku/datasets/University/"

            elif topic == "movie":
                path = "/home/oyku/datasets/Movie/"

            elif topic == "title":
                path = "/home/oyku/datasets/Article/"
            # --------------------------------------

            labeled = path + topic + "_" + lang + "_blocked_translated.csv"
            labeled = pd.read_csv(labeled)
            print(labeled.shape)

            print("Running translated test on " + topic + " dataset on language " + lang)


            fts_path = path + "features/" + topic + "_" + lang + "_baseline_features.csv"
            train_features = pd.read_csv(fts_path)
            print("Training features:  " + str(len(list(train_features))))

            exclude = ["_id", "ltable_id", "rtable_id"]
            gold = pd.DataFrame(labeled["Label"])

            cols = [col for col in list(train_features) if col not in exclude]
            train_features = train_features[cols]

            imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
            scale = StandardScaler()
            imp.fit(train_features)
            imp.statistics_[pd.np.isnan(imp.statistics_)] = 0
            features = scale.fit_transform(imp.transform(train_features))

            # Cross Validation
            model = XGBClassifier(random_state=7, n_estimators=350)
            kfold = model_selection.StratifiedKFold(n_splits=5, random_state=7)
            scoring = ['f1', 'recall', 'precision']
            scores = model_selection.cross_validate(model, features, gold.values.ravel(), cv=kfold, scoring=scoring)
            f1 = "%.3f (%.3f)" % (scores['test_f1'].mean() * 100, scores['test_f1'].std() * 100)
            recall = "%.3f (%.3f)" % (scores['test_recall'].mean() * 100, scores['test_recall'].std() * 100)
            precision = "%.3f (%.3f)" % (scores['test_precision'].mean() * 100, scores['test_precision'].std() * 100)

            print("Topic: %s ---  Lang: %s --- F1: %s     Recall: %s      Precision: %s" % (
            topic, lang, f1, recall, precision))
            version_results = [topic, lang, f1, recall, precision]
            df.loc[len(df)] = version_results

    df.to_csv(result_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
