'''
- Extracts data from DBpedia Article Categories file.
- Find duplicates with the Wikipedia Interlanguage links

'''


import csv
import pandas as pd

path = "/Users/oyku/Desktop/French/Articles/"


otherlang = "fr"
path_de = path + "article_categories_" + otherlang + ".tql"
path_en = path + "article_categories_en.tql"
lang_link = path + "interlanguage_links_" + otherlang + ".tql"


dbpedia_de = "http://" + otherlang + ".dbpedia.org/resource/"
dbpedia_en = "http://dbpedia.org/resource/"


#German: Kategorie
#Spanish: Categoría
#French: Catégorie

if otherlang == "de":
    category = "Kategorie:"
elif otherlang == "es":
    category = "Categoría:"
elif otherlang == "fr":
    category = "Catégorie:"

cat_de = "http://" + otherlang + ".dbpedia.org/resource/" + category
cat_en = "http://dbpedia.org/resource/Category:"

cat_de_out = path + "titles_" + otherlang + ".csv"
cat_en_out = path + "titles_en.csv"
withoutid_de = path + otherlang + "_withoutid.csv"


dupdet_de = path + "titles_dupdet_" + otherlang + ".csv"
dupdet_en = path + "titles_dupdet_en.csv"
dup_file = path + "titles_duplicates.csv"
all_entities = []

id = 0

def initialize():
    global all_entities
    global id
    all_entities.clear()
    id = 0

def extractTitle(token, link):

    title = token[len(link) + 1:-1]
    title = title.replace('_', ' ')
    return title


def extractCategoryName(line, language):
    temp = ""
    if language == otherlang:
        temp = cat_de
    else:
        temp = cat_en


    words = line.split(" ")
    category = words[2][len(temp)+1:-1]
    category = category.replace("_", " ")


    return category



def createCSV(outfilepath, dupfile, language):

    allrows = []
    header = ["id", "link", "title", "category"]

    with open(outfilepath, 'w') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(header)

        for entity in all_entities:
            row = []
            row.append(entity["id"])
            row.append(entity["link"])
            row.append(entity["title"])
            row.append(entity["category"])
            writer.writerow(row)

    csvfile.close()

    with open(dupfile, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Link", "ID"])
        for entity in all_entities:
            row = []
            row.append(entity["link"])
            row.append(entity["id"])
            writer.writerow(row)

    csvfile.close()


def locateProperties(token, properties, lang):


    global all_entities
    global id

    propdict = {}
    all_categories = "" # To store the each entities' properties
    link = ""  # Link describes the links in infobox according to language. Necessary to parse

    if lang == otherlang:
        link = dbpedia_de
    else:
        link = dbpedia_en



    titlename = extractTitle(token,link)


    for line in properties:

        category = extractCategoryName(line, lang)

        if all_categories == "":
            all_categories = category
        else:
            all_categories = all_categories + ", " + category


    if titlename != "" and titlename != "#" and all_categories != "":
        id += 1
        propdict["id"] = str(id)
        propdict["title"] = titlename
        propdict["category"] = all_categories
        propdict["link"] = token
        all_entities.append(propdict)



def createDataset(inp, outfilepath, dupfile, language):

    initialize()

    with open(inp) as f:

        token = ""
        properties = []
        for line in f:
            words = line.split(" ")
            link = words[0].strip()

            if token != link:
                locateProperties(token, properties, language)
                properties.clear()
                token = link

            properties.append(line)

    f.close()

    createCSV(outfilepath, dupfile, language)



def findDuplicates():
    dict_de = {}
    with open(dupdet_de, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            words = row[0].split(">,")
            link = words[0][1:-1]
            id = row[1]
            dict_de[link] = id
    csvfile.close()

    dict_en = {}
    with open(dupdet_en, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            words = row[0].split(">,")
            link=words[0][1:-1]
            id = row[1]
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
                        row = []
                        row.append(eng_id)
                        row.append(german_id)
                        writer.writerow(row)

    print("English dataset size is {}".format(len(dict_en)))
    print("German dataset size is {}".format(len(dict_de)))
    print("There are {} duplicates".format(i))



# def sampleprocess():
#     # Read German link - id csv
#     german_list = pd.read_csv(dupdet_de)
#
#     # Put in a dictionary
#     german_dict = {}
#     for index, row in german_list.iterrows():
#         german_dict[row["Link"]] = row["ID"]
#
#     # Read English link - id csv
#     english_list = pd.read_csv(dupdet_en)
#
#     english_dict = {}
#     for index, row in english_list.iterrows():
#         english_dict[row["Link"]] = row["ID"]
#
#
#     # Finding the duplicate pairs
#     matches = {}
#     with open(lang_link) as f:
#         for line in f:
#             words = line.split(">")
#             germanlink = words[0] + ">"
#
#             if germanlink in german_dict:
#                 otherlink = words[2][1:] + ">"
#
#                 if otherlink in english_dict:
#                     german_id = german_dict[germanlink]
#                     eng_id = english_dict[otherlink]
#                     matches[germanlink] = (german_id, eng_id)
#
#     print("Number of duplicates: " + str(len(matches)))
#
#     #Creating German link and id pairs
#     matches_df = pd.DataFrame(columns=['Link', 'ID'])
#     i = 1
#     for key in matches:
#         i += 1
#         matches_df.loc[i] = [key, matches[key][0]]
#
#     matches_df.to_csv(dupdet_de, index=False)
#
#
#     # Creating English id and German id pairs
#     duplicates_df = pd.DataFrame(columns=["English ID", "German ID"])
#     i = 1
#     for key in matches:
#         i += 1
#         duplicates_df.loc[i] = [matches[key][1], matches[key][0]]
#
#     duplicates_df.to_csv(dup_file, index=False)
#
#     # Sampling the huge German file
#     sample_german = german_list.sample(n=1800)
#
#     # Merging the matches and sampled non-matches together
#     frames = [matches_df, sample_german]
#     result = pd.concat(frames)
#
#     result = result.drop_duplicates(subset=['ID'], keep=False)
#
#     print("Number of other language dataset: " + str(len(result)))
#
#     german_df = pd.read_csv(cat_de_out)
#     result['ID'] = result['ID'].apply(int)
#     german_df['id'] = german_df['id'].apply(int)
#
#     # Joining the final list with real data
#     final = pd.merge(result, german_df, how='inner', left_on="ID", right_on="id")
#
#     final = final.drop(columns=['ID', 'link', "Link"])
#     final.to_csv(path + "titles_" + otherlang + "1.csv", index=False)
#
#
#     # Writing it without ids for future translation
#     cols = ["title", "category"]
#     without_id = final[cols]
#     without_id.to_csv(withoutid_de, index=False)

# createDataset(path_de, cat_de_out, dupdet_de, otherlang)
#createDataset(path_en, cat_en_out, dupdet_en, "en")

#sampleprocess()

#findDuplicates()

