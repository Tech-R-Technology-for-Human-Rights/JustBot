import os
import re
import json
import pandas as pd
import numpy as np




def get_json_data(file):
    with open(file, 'r') as f:
        return json.load(f)

def json_to_text_(doc):
    res = []
    if not len(doc['elements']):  # Remove this condition to add subsection titles 
        res.append(doc['content'])
    for e in doc['elements']:
        res.extend(json_to_text_(e))
    return res


def text_cleaner(text):
    ct = re.sub(elim, '', text)
    return ct

def json_to_text(doc):
    clean_lines = []
    for line in json_to_text_(doc):
        if re.search("relevant domestic|relevant international documents|the law|constitution of|the code of criminal procedure", line.lower() ):
            return " ".join(clean_lines)
        elif line == '...':
            pass
        else:
            clean_lines.append(text_cleaner(line))
    return " ".join(clean_lines)

def json_to_text_with_law(doc):
    clean_lines = []
    hint = 0
    for line in json_to_text_(doc):
        if re.search("relevant domestic|relevant international documents|the law|constitution of|the code of criminal procedure", line.lower() ):
            hint = 1
        elif line == '...':
            pass
        else:
            clean_lines.append(text_cleaner(line))
    return " ".join(clean_lines), hint


def extract_facts(dict, law = False):
    judgment = list(dict['content'].values())[0]
    if law: 
        try:
            facts = next((e for e in judgment if e.get('section_name') == 'facts'))
            text, hint = json_to_text_with_law(facts)
            return text, hint
        except:
            return []
    else:
        try:
            facts = next((e for e in judgment if e.get('section_name') == 'facts'))
            text = json_to_text(facts)
            return text
        except:
            return []


def extract_violation(doc):
    C = doc['conclusion']
    v = [e['base_article'] for e in C if (e.get('type') == 'violation')]# and not re.findall(r'p\d', e['base_article'
    nv = [m['base_article'] for m in C if (m.get('type') == 'no-violation')]# and not re.findall(r'p\d', e['base_article'
    return v , nv


def make_case(doc, law = False):
    C = doc['conclusion']
    if C:
        v, nv = extract_violation(doc)
        if law :
            facts_law_list = [extract_facts(doc, law = True)]
            if v and len(facts_hint_list[0]) == 2:
                art = int(v)
                text = facts_law_list[0][0]
                hint = facts_law_list[0][1]
                
                return text,hint, v, nv
            else:
                pass
        else:
            facts = extract_facts(doc)
            return facts , list(set(v)), list(set(nv))

def get_arts (dict):
    art_list = dict["article"]
    art_list = [art for art in art_list if not re.search(r'p\d', art)]
    return art_list

#TF-IDF Functions #########################################################################

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import spacy
from spacy import displacy
from collections import Counter
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Doc, Token, Span
from collections import Counter
from string import punctuation
import en_core_web_lg
to_remove =  list(set(list(STOP_WORDS) + list(['\n', 'u00a0',"applicants"]) + list(punctuation)+['applicant','proceeding', 'judgment', 'case', 'request',
'decision', 'bear', 'appeal','sodišče',"višje", "celju",'vienna',
"january","february","march","april","may","june","july","august","september","october","november","december",
'court', 'appendix', 'baku', 'zmir']) )

# turns a given text into lemas and removes entities
def spacy_ner_cleaner (txt):
    doc = spacy_nlp(txt)
    rm_ents = []
    ner_lbls = ['PERSON', 'LAW', 'GPE', 'ORG', 'DATE', 'ORDINAL', 'CARDINAL', 'NORP', 'MONEY', 'WORK_OF_ART', 'LOC']
    for ent in doc.ents:
        if ent.label_ in ner_lbls or ent.text.lower() in to_remove :
            rm_ents.append(ent.text.lower())
    rm_ents = iter([n.split() for n in rm_ents])
    rm_ents_split = []
    while True:
        n = next(rm_ents, "end")
        if n == "end":
            break
        else:
            rm_ents_split += n
    print(rm_ents_split)
    
    words = []
    for ent in doc:
        print(ent.text)
        c1 = not re.search(r'\d',ent.lemma_)
        c2 =  not ent.text.lower() in  rm_ents_split+to_remove
        c3 = len(ent.lemma_) > 3
        if (c1 and c2 and c3):
            words.append(ent.lemma_)
            print("Aded lemma >>>>>>>>>>>>", ent.lemma_)
        else:
            pass
    return (" ".join(words))

##### Functions to summarize a text
def check_w_to_remove (w):
    cond = w.lower() in to_remove
    return cond

def add_to_voc(dict, word):
    word = word.lower()
    if word in dict.keys():
        dict[word] += 1
    else:
        dict[word] = 1

# Builds a dictionary of word freequencies
def evaluate(txt):
    doc = spacy_nlp(txt)
    words = [w.text for w in doc]
    vocab_dict = {}
    for w in words:
        if not check_w_to_remove(w):
            add_to_voc(vocab_dict,w)
    max_w = max(vocab_dict.values())
    for w in vocab_dict.keys():
        vocab_dict[w] = vocab_dict[w]/max_w

    sent_dict = {}
    for sent in doc.sents:
        for word in sent:
            lw = word.text.lower()
            if lw in vocab_dict.keys():
                if sent in sent_dict.keys():
                    sent_dict[sent] += vocab_dict[lw]
                else:
                    sent_dict[sent] = vocab_dict[lw]
            else:
                continue
    return sent_dict

def summarizer (txt):
    l = len(txt)
    sent_dict = evaluate(txt)
    top = sorted(sent_dict.values(), reverse = True)
    cutoff = 0.25
    if l < 500:
        cutoff = 1
    elif l < 1000:
        cutoff = 0.5
    elif l > 10000:
        cutoff = 0.1
    else:
        pass
    top_25_p = int(cutoff *len(top))
    top = top[: top_25_p]
    summary = []
    for sent , value in sent_dict.items():
        if value in top:
            summary.append(sent.text)
    return " ".join(summary)

##############################################################################################
spacy_nlp = en_core_web_lg.load()
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

elim = '\d+\. ?|\d+\.\\xa0|\\xa0|§ \d+'
obj = get_json_data('echr_2_0_0_unstructured_cases.json')
total_cases = len(obj)
indice = next((i for i in range(total_cases)))

#articles = list(set( sum([get_arts(case) for case in  obj], []) ))[1:-1]

print(f"We have {total_cases} cases in the list")

data = [make_case(dict) for dict in obj]
cases  = [case for case in data if (case and case[0])]

# Extracting base articles that will serve as labels
base_articles = [list(extract_violation(doc)) for doc in obj]
base_articles = [item for subl in base_articles for item in subl]
base_articles = [item for subl in base_articles for item in subl]
base_articles = list(set(base_articles))
print(f'Unique base_article labels are:\n{base_articles}')


#>>>>> Generating summary of facts
# Getting summaries of cases facts
print("Stage 1 : generating summaries ")
summs = [summarizer(txt) for txt , v,nv in cases]
summs = pd.DataFrame({'facts_summary': summs } )

#>>>>> TFIDF features
# getting the TF-IDF vectorizer for top ngrams 
import statistics as stat
print("Stage 2 : generating TFIDF N-gram features ")
fact_lemas = [spacy_ner_cleaner(txt) for txt , v,nv in cases ]

nas = 0
for f in fact_lemas:
    if not f or f==[] :
        nas +=1
print("NAS : ", nas)
print('total: ', total_cases)



# define vectorizer 
tfidf = TfidfVectorizer(sublinear_tf = True, 
min_df = 15, 
norm='l2', 
encoding='latin-1',
ngram_range=(1,4),
stop_words='english')

#vectorize n-grames
features = np.array(tfidf.fit_transform(fact_lemas).toarray())
# Calculate the mean tf_idf of each column and return the sorting indexes for descending order of the mean value
inds = np.argsort([stat.mean(features[:, col]) for col in range(features.shape[1])],  axis = 0 )[::-1]


# Rearrange features and keep only the top n N-grams
top_n = 6000
features = features[:,inds[0:top_n]]
feature_names = np.array(tfidf.get_feature_names())[inds[0:top_n]]
#Building pandas df with top ngtams based on tf-idf values
features_df = pd.DataFrame(features, columns=feature_names)
#-------------------------------------------------------------------------------------

print("Stage 3 : generating labels")
# >>> LABELS
#Adding the labels for the cases extracted
#articles = ["1","2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "18", "p1-1","p1-2","p1-3","p4-4", "p7-2", "p7-3", "p7-1", "p7-4", "p6-1","p12-1"]+
#['41', '3', '36', '34', '39', '37', '8', '30', '12', '19', '2', '57', '13', '43', '7', '5', '35', '4', '9', '38', '29', '53', '14',
# '33', '56', '10', '32', '16', '17', '11', '15', '18', '6', '25', '1', '52', '28', '27', '26']
#articles = list(set(articles))


case_labels_v = ["Article " + str(art) for art in base_articles]
case_input = {}

for art in base_articles:
    temp = []
    for d in cases:

        if d is not None:
            if str(art) in d[1]:
                temp.append(1)
            elif str(art) in d[2]:
                temp.append(-1)
            else:
                temp.append(0)
        # commented as case ignores  data as well
#         else:
#             temp.append(0)
    case_input[case_labels_v[base_articles.index(art)]] = temp

labels = pd.DataFrame.from_dict(case_input)
##------------------------------------------------------


# Exporting data as pandas df
print("MERGING ... ")
BIG_data = pd.concat([summs,features_df,labels], axis=1)
print("FINAL DATAFRAME SHAPE : ", BIG_data.shape)
print("EXPORTING AS ./open_data_summary_tfidf_multilabel_exp_4_extra_lbls.csv")
BIG_data.to_csv("./open_data_summary_tfidf_multilabel_exp_4_extra_lbls.csv")
print("DONE")



