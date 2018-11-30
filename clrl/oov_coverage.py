'''
-Receives OOV words of a dataset
-Returns INV words of OOVs if possible using
    - Morphology Check
    - Spell check
    - Subword check
'''


from babylon.fasttext import FastVector
from polyglot.text import Text, Word
import io
import sys
import stringdist
import numpy as np
from gensim.models import FastText as fText
import pickle


def union3(dict1, dict2, dict3):
    return dict(list(dict1.items()) + list(dict2.items()) + list(dict3.items()))


def load_vec(emb_path):
    fin = io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}

    for (i, line) in enumerate(fin):
        word, vect = line.rstrip().split(' ', 1)
        vect = np.fromstring(vect, sep=' ')
        data[word] = vect

    return data


def main(argv):
    topic = argv[0]
    filelang = argv[1]
    mainlang = argv[2]

    path = "/home/oyku/embeddings/fasttext/wiki." + filelang + ".align.vec"
    dictionary = load_vec(path)

    mono_path = "/home/oyku/monolingual_fasttext/cc." + filelang + ".300"
    mono_wv = fText.load_fasttext_format(mono_path)

    file = "/home/oyku/myversion/oov_words/" + mainlang + "/" + topic + "_" + filelang + ".txt"
    f = open(file, 'r', encoding='utf8')
    content = f.readlines()

    cont = set()

    for el in content:
        if not el.strip().isdigit():
            cont.add(el.strip())

    print("The number of OOVs: " + str(len(content)))
    print("The number of word OOVs: " + str(len(cont)))

    ## Morphologic
    morphs = {}
    for blob in cont:
        if not blob.isdigit():
            text = Text(blob)
            text.language = filelang
            morphemes = []
            for morp in text.morphemes:

                if len(morp) > 3 and morp in dictionary:
                    morphemes.append(morp)

            if len(morphemes) != 0:
                morphs[blob] = morphemes

    print("Morphologic check is over")

    left = cont.difference(morphs)

    ## Spelling
    spellex = {}
    for oov in left:
        if len(oov) > 2:
            possibles = []
            for inv in dictionary:
                if stringdist.rdlevenshtein(oov, inv) == 1:
                    possibles.append(inv)
            if len(possibles) == 1:
                spellex[oov] = possibles

    print("Spelling check is over")

    next_left = left.difference(spellex)

    fasttext_bin = {}
    for oov in next_left:
        try:
            similars = mono_wv.wv.most_similar(oov.strip())

            most_sim = ""
            for sim in similars:
                if sim[0] in dictionary and sim[1] > 0.5:
                    most_sim = sim[0]
                    break

            if most_sim != "":
                fasttext_bin[oov.strip()] = [most_sim]
        except:
            continue

    print("Fasttext check is over")

    print("-----------------------------------------------")

    print("Identified with morphologic analysis: " + str(len(morphs)))
    print("Identified with spell analysis: " + str(len(spellex)))
    print("Identified with Fasttext: " + str(len(fasttext_bin)))

    union = union3(morphs, spellex, fasttext_bin)
    print("Total: " + str(len(union)))

    saved_path = "/home/oyku/myversion/oov_matches/" + mainlang + "/" + topic + "_" + filelang + ".p"
    pickle.dump(union, open(saved_path, "wb"))


if __name__ == "__main__":
    main(sys.argv[1:])
