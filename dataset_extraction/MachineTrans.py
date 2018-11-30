'''
Uses Joshua Machine Translation to translate a csv file

'''

from subprocess import Popen
import csv
import pandas as pd

joshua_path = "/Users/oyku/Desktop/french-joshua/"


helperpath = joshua_path + "helper.txt"
helperoutpath = joshua_path + "helperout.txt"

path = "/Users/oyku/Desktop/French/Movie/"

originalpath = path + "movie_fr_withoutid.csv"
translatedpath = path + "movie_fr_translated.csv"

header = []


def prepareForTranslation():
    global header

    txtout = open(helperpath, "w+")
    with open(originalpath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:

            for col in row:
                txtout.write(col)
                txtout.write("\n")
            txtout.write("!!!!!\n")

    print("End of preparation")

def translate():
    p = Popen("./translate.sh", cwd=joshua_path, shell=True)
    stdout, stderr = p.communicate()
    print("End of translation")

def convertToCSV():
    with open(helperoutpath, "r") as f:
        with open(translatedpath, "w") as csvfile:
            content = f.readlines()
            trwriter = csv.writer(csvfile, delimiter=',')
            trwriter.writerow(["id"] + header)
            row = []
            id = 0
            for line in content:
                line = line.strip()
                if line != "!!!!!":
                    row.append(line)
                else:
                    id += 1
                    trwriter.writerow([str(id)] + row)
                    row.clear()

    print("End of CSV")






prepareForTranslation()
translate()
convertToCSV()

