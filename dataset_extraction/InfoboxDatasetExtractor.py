'''
- Extracts data from DBpedia Infobox files with a given language and infobox keyword 
- Loads datasets on CSV files
- Find duplicate records with using Wikipedia interlanguage links
'''

import csv
import sys
from importlib import reload


if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding('utf8')



path = "/Users/oyku/Desktop/French/"

otherlang = "fr"
topic = "uni"


# Change infobox keywords for each language and dataset
keyword_de = "Infobox_Université&"
keyword_en = "Infobox_University&"

infobox_de = path + "infobox_properties_mapped_" + otherlang + ".tql"
infobox_en = path + "infobox_properties_mapped_en.tql"
lang_link = path + "interlanguage_links_" + otherlang + ".tql"

de_csv = path + topic.title() + "/" + topic + "_" + otherlang + ".csv"
en_csv = path + topic.title() + "/" + topic + "_en.csv"
withoutid = path + topic.title() + "/" + topic + "_" + otherlang + "_withoutid.csv"
dupdet_de = path + topic.title() + "/" + topic + "_dupdet_" + otherlang + ".csv"
dupdet_en = path + topic.title() + "/" + topic + "_dupdet_en.csv"
dup_file = path + topic.title() + "/" + topic + "_duplicates.csv"

dbpedia_de = "http://" + otherlang + ".dbpedia.org/"
dbpedia_en = "http://dbpedia.org/"

globalPropNames = set()
all_entities = []
links_list = []
excluded_keywords = []

def initialize():
    global globalPropNames
    global all_entities
    global links_list

    links_list.clear()
    all_entities.clear()
    globalPropNames.clear()


def firstway(line):
    res = line.split('^^')
    if len(res) != 0:
        res = res[0]
    else:
        return ""

    res = res.split('"')
    if len(res) > 2:
        res = res[1]
    else:
        return ""

    res = res.replace("*", "")
    res = res.replace("\\n", ", ")

    return res


def secondway(line, link):
    tokens = line.split('>')
    res = tokens[2][1:]
    temp = link + "resource/"
    res = res[len(temp) + 1:].replace("_", " ")
    return res


# Extracts the property name from the lines of infobox dataset
def extractPropName(line, link):
    tokens = line.split(">")

    temp = tokens[1][1:]
    templink = link + "property/"
    propName = temp[len(templink) + 1:]

    return propName


# To extract the properties with their property name and property values.
def locateProperties(token, properties, nameKeyword, language):
    global globalPropNames
    global all_entities

    propdict = {}  # To store the each entities' properties
    link = ""  # Link describes the links in infobox according to language. Necessary to parse
    propName = ""

    if language == otherlang:
        link = dbpedia_de
    else:
        link = dbpedia_en

    for line in properties:
        propName = extractPropName(line, link)
        if propName in excluded_keywords:
            continue

        prop = firstway(line)

        if prop == "":
            prop = secondway(line, link)

        propdict[propName] = prop if propName not in propdict else propdict[propName] + ", " + prop

    for key in propdict:
        globalPropNames.add(key)

    propdict["link"] = token
    all_entities.append(propdict)


# This function is to have the same structure among all German entities, because German movie dataset is not consistent.
def extraCareforGermanMovie():
    global globalPropNames
    global all_entities

    to_be_removed = []
    word1 = "ot"
    word2 = "dt"

    if word1 in globalPropNames:
        globalPropNames.remove(word1)

    for a in all_entities:
        if word2 not in a and word1 not in a:
            to_be_removed.append(a)
        elif word2 not in a and word1 in a:
            a[word2] = a[word1]

    for a in to_be_removed:
        all_entities.remove(a)


def createCSV(outfilepath, dupfile, header, language):
    allrows = []
    with open(outfilepath, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for entity in all_entities:
            row = []
            for prop in header:
                if prop in entity:
                    row.append(entity[prop])
                else:
                    row.append("")

            allrows.append(row)
            writer.writerow(row)

    with open(dupfile, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Link", "ID"])
        for row in links_list:
            writer.writerow(row)

    # To be given to translator
    if language != "en":
        with open(withoutid, "w") as extr:
            extrwriter = csv.writer(extr)
            extrwriter.writerow(header[1:])

            for row in allrows:
                extrwriter.writerow(row[1:])


# Reads the infobox dataset, takes the lines with the keyword and send the locateProperties to extract properties
def createDataset(filepath, outfilepath, dupfile, keyword, nameKeyword, language):
    initialize()

    with open(filepath) as f:

        token = ""
        properties = []
        for line in f:


            if keyword in line:

                words = line.split(" ")
                link = words[0].strip()
                if token != link:
                    locateProperties(token, properties, nameKeyword, language)

                    properties.clear()
                    token = link

                properties.append(line)


    f.close()

    # if language == otherlang and keyword_de == "Infobox_Film&":
    #     extraCareforGermanMovie()

    id = 0
    to_be_removed = []
    for entity in all_entities:
        if nameKeyword in entity:
            id += 1
            entity["id"] = id
            links_list.append([entity["link"], entity["id"]])
        else:
            to_be_removed.append(entity)

    for entity in to_be_removed:
        all_entities.remove(entity)


    globalPropNames.remove(nameKeyword)
    header = ["id"] + [nameKeyword] + list(globalPropNames)

    print("Size of the dataset-" + language + ": " + str(len(all_entities)))
    print("Name of the entity: " + nameKeyword)
    print("Number of the properties:" + str(len(header)))
    print(header)

    createCSV(outfilepath, dupfile, header, language)


def findDuplicates():
    dict_de = {}
    with open(dupdet_de, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        header = next(reader)
        for row in reader:
            words = row[0].split(">,")
            link = words[0][1:]
            id = words[1]
            dict_de[link] = id
    csvfile.close()

    dict_en = {}
    with open(dupdet_en, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        header = next(reader)
        for row in reader:
            words = row[0].split(">,")
            link=words[0][1:]
            id = words[1]
            dict_en[link] = id
    csvfile.close()

    i = 0

    with open(dup_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["English ID", "German ID"])
        with open(lang_link) as f:
            for line in f:
                words = line.split(">")
                germanlink = words[0][1:]

                if germanlink in dict_de:
                    otherlink = words[2][2:]
                    if otherlink in dict_en:
                        i += 1
                        german_id = dict_de[germanlink]
                        eng_id = dict_en[otherlink]
                        writer.writerow([eng_id, german_id])

    print("There are {} duplicates".format(i))


#createDataset(infobox_de, de_csv, dupdet_de, keyword_de, "nom", otherlang)
#createDataset(infobox_en, en_csv, dupdet_en, keyword_en, "name", "en")

#findDuplicates()

