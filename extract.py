import json
import spacy
import math
import pandas as pd
import tqdm
from keybert import KeyBERT
import warnings
import nltk
from nltk import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
import numpy as np

warnings.filterwarnings("ignore")

def extract_tfidf(tag_num, json_file):
    file = open(json_file, 'r')
    json_list = json.load(file)
    
    corpus = {}

    res_title = []
    res_tag = []
    res_link = []
    res_title_tag = []
    res_category = []
    res_author = []
    res_image = []
    
    for json_raw in tqdm.tqdm(json_list):
        # for every product
        review_list = []
        for rev in json_raw["reviews"]:
            if 'body' in rev.keys():
                review_list.append(rev['body'])
            else:
                continue
        if len(review_list) == 0:
            continue
        corpus[json_raw["title"]] = review_list
        D = len(review_list)
        
        nlp = spacy.load("en_core_web_sm")
        dict_list = []
        tokens_pos = {}
        for review in review_list:
            dict_all = {}
            word_list = nlp(review)
            for token in word_list:
                if token.is_stop:
                    continue
                if token.is_punct:
                    continue
                tokens_pos[str(token.lemma_)] = token.pos_
                if str(token.lemma_) in dict_all.keys():
                    dict_all[str(token.lemma_)] += 1
                else:
                    dict_all[str(token.lemma_)] = 1
                if str(token.lemma_) not in tokens_pos.keys():
                    tokens_pos[str(token.lemma_)] = tokens_pos
            dict_list.append(dict_all)
 
        tf_idf = {}
        for doc in dict_list:
            for key,value in doc.items():
                tf = math.log(value + 1, 2)
                k = 0
                for i in dict_list:
                    if key in i.keys():
                        k += 1
                idf = 1 + math.log(D / k)
                if key not in tf_idf:
                    tf_idf[key] = tf * idf
                else:
                    tf_idf[key] = max(tf * idf, tf_idf[key])
        tf_idf_sorted = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
        i = 0
        j = 0
        title_str = json_raw["title"] + ' '
        tag_str = ''
        while i < len(tf_idf_sorted) and j < tag_num:
            if tf_idf_sorted[i][0] in tokens_pos and tokens_pos[tf_idf_sorted[i][0]] == 'ADJ':
                tag_str += tf_idf_sorted[i][0] + ', '
                j += 1
            i += 1
            
        # for output
        if 'title' in json_raw.keys():
            res_title.append(json_raw["title"]) 
        else:
            res_title.append('')
        res_title_tag.append((title_str+tag_str).strip())
        res_tag.append(tag_str.strip())
        if 'link' in json_raw.keys():
            res_link.append(json_raw["link"])
        else:
            res_link.append('')
        res_category_str = ''
        if 'categories' in json_raw.keys():
            if len(json_raw["categories"]) != 0:
                if 'name' in json_raw["categories"][0].keys():
                    res_category_str = json_raw["categories"][0]['name']
        res_category.append(res_category_str)
        res_author_tmp = []
        if 'authors' in json_raw.keys():
            if len(json_raw["authors"]) != 0:
                for a in json_raw["authors"]:
                    if 'name' in a.keys():
                        res_author_tmp.append(a['name'])
        res_author_tmp = set(res_author_tmp)
        res_author_str = ''
        for a in res_author_tmp:
            res_author_str += a
            res_author_str += ', '
        res_author.append(res_author_str) 
        if 'image' in json_raw.keys():
            res_image.append(json_raw["image"])
        else:
            res_image.append('')

    return res_title, res_tag, res_title_tag, res_link, res_category, res_author, res_image

def extract_bert(tag_num, json_file):
    file = open(json_file, 'r')
    json_list = json.load(file)
    
    corpus = {}
    
    res_title = []
    res_tag = []
    res_link = []
    res_title_tag = []
    res_category = []
    res_author = []
    res_image = []
    
    for json_raw in tqdm.tqdm(json_list):
        # for every product
        review_list = []
        for rev in json_raw["reviews"]:
            if 'body' in rev.keys():
                review_list.append(rev['body'])
            else:
                continue
        if len(review_list) == 0:
            continue
        corpus[json_raw["title"]] = review_list
        review_all = ''
        for review in review_list:
            review_all += review + ' '
        
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(review_all, keyphrase_ngram_range=(1, 1), stop_words='english', use_mmr=True, diversity=0.7, top_n=int(tag_num*1.5))
        tag_str = ''
        ind = 0
        for k in keywords:
            if not k[0].isdigit():
                ind += 1
                tag_str += k[0] + ' '
            if ind == tag_num:
                break
        title_str = json_raw["title"] + ', '
    
        # for output
        if 'title' in json_raw.keys():
            res_title.append(json_raw["title"]) 
        else:
            res_title.append('')
        res_title_tag.append((title_str+tag_str).strip())
        res_tag.append(tag_str.strip())
        if 'link' in json_raw.keys():
            res_link.append(json_raw["link"])
        else:
            res_link.append('')
        res_category_str = ''
        if 'categories' in json_raw.keys():
            if len(json_raw["categories"]) != 0:
                if 'name' in json_raw["categories"][0].keys():
                    res_category_str = json_raw["categories"][0]['name']
        res_category.append(res_category_str)
        res_author_tmp = []
        if 'authors' in json_raw.keys():
            if len(json_raw["authors"]) != 0:
                for a in json_raw["authors"]:
                    if 'name' in a.keys():
                        res_author_tmp.append(a['name'])
        res_author_tmp = set(res_author_tmp)
        res_author_str = ''
        for a in res_author_tmp:
            res_author_str += a
            res_author_str += ', '
        res_author.append(res_author_str)
        if 'image' in json_raw.keys():
            res_image.append(json_raw["image"])
        else:
            res_image.append('')
            

    return res_title, res_tag, res_title_tag, res_link, res_category, res_author, res_image
    
def clean(text):
    text = text.lower()
    printable = set(string.printable)
    text = filter(lambda x: x in printable, text)
    text = "".join(list(text))
    return text

def extract_textrank(tag_num, json_file):
    file = open(json_file, 'r')
    json_list = json.load(file)

    corpus = {}
    
    res_title = []
    res_tag = []
    res_link = []
    res_title_tag = []
    res_category = []
    res_author = []
    res_image = []
    
    for json_raw in tqdm.tqdm(json_list):
        # for every product
        review_list = []
        for rev in json_raw["reviews"]:
            if 'body' in rev.keys():
                review_list.append(rev['body'])
            else:
                continue
        if len(review_list) == 0:
            continue
        corpus[json_raw["title"]] = review_list
        review_all = ''
        for review in review_list:
            review_all += review + ' '

        Cleaned_text = clean(review_all)
        text = word_tokenize(Cleaned_text)
        # nltk.download('averaged_perceptron_tagger')
        
        POS_tag = nltk.pos_tag(text)
        # nltk.download('wordnet')
        
        wordnet_lemmatizer = WordNetLemmatizer()
        adjective_tags = ['JJ','JJR','JJS']
        lemmatized_text = []
        for word in POS_tag:
            if word[1] in adjective_tags:
                lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a")))
            else:
                lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun
                    
        POS_tag = nltk.pos_tag(lemmatized_text)
        
        stopwords = []
        wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW'] 
        for word in POS_tag:
            if word[1] not in wanted_POS:
                stopwords.append(word[0])
        punctuations = list(str(string.punctuation))
        stopwords = stopwords + punctuations
        
        stopword_file = open("long_stopwords.txt", "r")
        #Source = https://www.ranks.nl/stopwords
        lots_of_stopwords = []
        for line in stopword_file.readlines():
            lots_of_stopwords.append(str(line.strip()))
        stopwords_plus = []
        stopwords_plus = stopwords + lots_of_stopwords
        stopwords_plus = set(stopwords_plus)
        
        processed_text = []
        for word in lemmatized_text:
            if word not in stopwords_plus:
                processed_text.append(word)
        vocabulary = list(set(processed_text))
        
        vocab_len = len(vocabulary)

        weighted_edge = np.zeros((vocab_len,vocab_len),dtype=np.float32)

        score = np.zeros((vocab_len),dtype=np.float32)
        window_size = 3
        covered_coocurrences = []

        for i in range(0,vocab_len):
            score[i]=1
            for j in range(0,vocab_len):
                if j==i:
                    weighted_edge[i][j]=0
                else:
                    for window_start in range(0,(len(processed_text)-window_size)):
                        window_end = window_start+window_size
                        window = processed_text[window_start:window_end]
                        if (vocabulary[i] in window) and (vocabulary[j] in window):
                            index_of_i = window_start + window.index(vocabulary[i])
                            index_of_j = window_start + window.index(vocabulary[j])
                            # index_of_x is the absolute position of the xth term in the window 
                            # (counting from 0) 
                            # in the processed_text
                            if [index_of_i,index_of_j] not in covered_coocurrences:
                                weighted_edge[i][j]+=1/math.fabs(index_of_i-index_of_j)
                                covered_coocurrences.append([index_of_i,index_of_j])
                            
        inout = np.zeros((vocab_len),dtype=np.float32)
        for i in range(0,vocab_len):
            for j in range(0,vocab_len):
                inout[i]+=weighted_edge[i][j]
        
        MAX_ITERATIONS = 50
        d = 0.85
        threshold = 0.0001 #convergence threshold
        for iter in range(0,MAX_ITERATIONS):
            prev_score = np.copy(score)
            for i in range(0,vocab_len):
                summation = 0
                for j in range(0,vocab_len):
                    if weighted_edge[i][j] != 0:
                        summation += (weighted_edge[i][j]/inout[j])*score[j] 
                score[i] = (1-d) + d*(summation)
            if np.sum(np.fabs(prev_score-score)) <= threshold: #convergence condition
                print("Converging at iteration "+str(iter)+"....")
                break 

        phrases = []
        phrase = " "
        for word in lemmatized_text:
            if word in stopwords_plus:
                if phrase!= " ":
                    phrases.append(str(phrase).strip().split())
                phrase = " "
            elif word not in stopwords_plus:
                phrase+=str(word)
                phrase+=" "
            
        unique_phrases = []
        for phrase in phrases:
            if phrase not in unique_phrases:
                unique_phrases.append(phrase)
                
        for word in vocabulary:
            #print word
            for phrase in unique_phrases:
                if (word in phrase) and ([word] in unique_phrases) and (len(phrase)>1):
                    #if len(phrase)>1 then the current phrase is multi-worded.
                    #if the word in vocabulary is present in unique_phrases as a single-word-phrase
                    # and at the same time present as a word within a multi-worded phrase,
                    # then I will remove the single-word-phrase from the list.
                    unique_phrases.remove([word])
                    
        phrase_scores = []
        keywords = []
        for phrase in unique_phrases:
            phrase_score=0
            keyword = ''
            for word in phrase:
                keyword += str(word)
                keyword += " "
                phrase_score+=score[vocabulary.index(word)]
            phrase_scores.append(phrase_score)
            keywords.append(keyword.strip())
            
        sorted_index = np.flip(np.argsort(phrase_scores),0)
        
        
        tag_str = ''
        for i in range(0, min(tag_num, len(sorted_index))):
            # print(str(keywords[sorted_index[i]])+", ", end=' ')
            tag_str += str(keywords[sorted_index[i]]) + ', '
        title_str = json_raw["title"] + ' '

        # for output
        if 'title' in json_raw.keys():
            res_title.append(json_raw["title"]) 
        else:
            res_title.append('')
        res_title_tag.append((title_str+tag_str).strip())
        res_tag.append(tag_str.strip())
        if 'link' in json_raw.keys():
            res_link.append(json_raw["link"])
        else:
            res_link.append('')
        res_category_str = ''
        if 'categories' in json_raw.keys():
            if len(json_raw["categories"]) != 0:
                if 'name' in json_raw["categories"][0].keys():
                    res_category_str = json_raw["categories"][0]['name']
        res_category.append(res_category_str)
        res_author_tmp = []
        if 'authors' in json_raw.keys():
            if len(json_raw["authors"]) != 0:
                for a in json_raw["authors"]:
                    if 'name' in a.keys():
                        res_author_tmp.append(a['name'])
        res_author_tmp = set(res_author_tmp)
        res_author_str = ''
        for a in res_author_tmp:
            res_author_str += a
            res_author_str += ', '
        res_author.append(res_author_str) 
        if 'image' in json_raw.keys():
            res_image.append(json_raw["image"])
        else:
            res_image.append('')
    
    return res_title, res_tag, res_title_tag, res_link, res_category, res_author, res_image
    

if __name__ == '__main__':
    # tf-idf
    # res_title, res_tag, res_title_tag, res_link, res_category, res_author, res_image = extract_tfidf(10, 'books_result_page_2000.json')
    # df = pd.DataFrame({'Title': res_title, 'Title_and_tag': res_title_tag, 'Tag': res_tag, 'Link': res_link, 'Category': res_category, 'Author': res_author, 'Image': res_image})
    # df.to_csv('tag_10_tfidf_2000.csv', encoding='utf-8', index=False)
    
    # bert
    # res_title, res_tag, res_title_tag, res_link, res_category, res_author, res_image = extract_bert(10, 'books_result_page_2000.json')
    # df = pd.DataFrame({'Title': res_title, 'Title_and_tag': res_title_tag, 'Tag': res_tag, 'Link': res_link, 'Category': res_category, 'Author': res_author, 'Image': res_image})
    # df.to_csv('tag_10_bert_2000.csv', encoding='utf-8', index=False)
    
    # textrank
    res_title, res_tag, res_title_tag, res_link, res_category, res_author, res_image = extract_textrank(10, 'books_result_page_2000.json')
    df = pd.DataFrame({'Title': res_title, 'Title_and_tag': res_title_tag, 'Tag': res_tag, 'Link': res_link, 'Category': res_category, 'Author': res_author, 'Image': res_image})
    df.to_csv('tag_10_textrank_2000.csv', encoding='utf-8', index=False)