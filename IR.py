# python3.8 -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 1 -input json -index index -storePositions -storeDocvectors -storeRaw
import json
from pyserini.index import IndexReader
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from subprocess import call
import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import warnings
import time
import os

warnings.filterwarnings("ignore")


def csv2json(tag_file, json_folder, json_file):
    # csvfile = pd.read_csv("tag_10_bert_1000.csv")
    csvfile = pd.read_csv(tag_file)
    # jsonfile = open('json/tag_10_bert_1000.json', 'w')
    jsonfile = open(os.path.join(json_folder, json_file), 'w')
    for i in tqdm.tqdm(range(csvfile.shape[0])):
        dic = {"id": csvfile.loc[i,'Title'], "contents": csvfile.loc[i,'Title_and_tag']}
        json.dump(dic, jsonfile)
        jsonfile.write('\n')
        

class Ranker(object):
    def __init__(self, index_reader):
        self.index_reader = index_reader

    def score(query):        
        rank_score = 0
        return rank_score


class BM25Ranker(Ranker):
    def __init__(self, index_reader, doc_id_list, query_list):
        super(BM25Ranker, self).__init__(index_reader)
        
        self.avg_d_len = index_reader.stats()['total_terms'] / index_reader.stats()['documents'] # avg_dl
        self.N = index_reader.stats()['documents']
        self.doc_vector_dic = {}
        for doc_id in tqdm.tqdm(doc_id_list):
            self.doc_vector_dic[doc_id] = index_reader.get_document_vector(doc_id)
        
        self.analyze_query_list = []
        for query in tqdm.tqdm(query_list): 
            self.analyze_query_list.append([index_reader.analyze(term)[0] if len(index_reader.analyze(term)) > 0 else "" for term in query.split(" ")])

    def score(self, query, doc_id, k1=1.2, b=0.75, k3=1.2):
        rank_score = 0
        
        index_reader = self.index_reader
        avg_d_len = self.avg_d_len
        N = self.N
        analyzed_query = self.analyze_query_list[query]
        doc_vector = self.doc_vector_dic[str(doc_id)]
        d_len = sum(doc_vector.values()) # |d|
        
        k1 = 0.5
        b = 1.5
        k3 = 1.2
        
        for analyzed_term in analyzed_query:
            if analyzed_term in doc_vector.keys():
                tf_q = analyzed_query.count(analyzed_term) # c(w,q) 
                df, _ = index_reader.get_term_counts(analyzed_term, analyzer=None) # df(w)    
                tf = doc_vector[analyzed_term] # c(w,d)
                term1 = np.log((N - df + 0.5) / (df + 0.5))
                term2 = (k1 + 1) * tf / (k1 * ((1 - b) + b * d_len / avg_d_len) + tf)
                term3 = (k3 + 1) * tf_q / (k3 + tf_q)
                rank_score += term1 * term2 * term3

        return rank_score

def bm25(input_corpus, input_query, output_file, res_doc_num = 10):
    json_folder = "json"
    index_folder = "index"
    csv2json(input_corpus, json_folder, input_corpus.split('.')[0]+'.json')
    call(["python3.8", "-m", "pyserini.index", "-collection", "JsonCollection", "-generator", "DefaultLuceneDocumentGenerator", "-threads", "1", "-input", json_folder, "-index", index_folder, "-storePositions", "-storeDocvectors", "-storeRaw"])
    
    index_reader = IndexReader(index_folder)
    f_doc = pd.read_csv(input_corpus)
    f_query = pd.read_csv(input_query)
    doc_id_list = list(f_doc['Title'])
    query_list = list(f_query['Query Description'])
    ranker = BM25Ranker(index_reader, doc_id_list, query_list)
    f_query_id = []
    f_query_content = []
    f_tag = []
    f_title = []
    f_link = []
    f_category = []
    f_author = []
    f_image = []
    
    tag_file = pd.read_csv(input_corpus)
    f_tag_dic = {}
    for i in range(tag_file.shape[0]):
        tmptmp = []
        tmptmp.append(tag_file.loc[i,'Tag'])
        tmptmp.append(tag_file.loc[i,'Link'])
        tmptmp.append(tag_file.loc[i,'Category'])
        tmptmp.append(tag_file.loc[i,'Author'])
        tmptmp.append(tag_file.loc[i,'Image'])
        f_tag_dic[tag_file.loc[i,'Title']] = tmptmp
        
    for q in range(len(query_list)):
        score = {}
        for doc_id in tqdm.tqdm(doc_id_list):
            s = ranker.score(q, doc_id)
            score[doc_id] = s
        score_s = sorted(score.items(), key=lambda x: x[1], reverse=True)
        j = 0
        j_uniq = 0
        while j <= res_doc_num:
            if doc_id not in f_title:
                doc_id = score_s[j][0]
                f_query_id.append(str(q))
                f_query_content.append(query_list[q])
                f_title.append(doc_id)
                f_tag.append(f_tag_dic[doc_id][0])
                f_link.append(f_tag_dic[doc_id][1])
                f_category.append(f_tag_dic[doc_id][2])
                f_author.append(f_tag_dic[doc_id][3])
                f_image.append(f_tag_dic[doc_id][4])
                j_uniq += 1
            j += 1
    
    f_df = pd.DataFrame({'QueryId': f_query_id, 'QueryCtonten': f_query_content, 'Title': f_title, 'Tag': f_tag,
                         'Link': f_link, 'Category': f_category, 'Author': f_author, 'Image': f_image})
    f_df.to_csv(output_file, index=False)
    
    return 


def bert(input_corpus, input_query, output_file, min_doc_num = 15, max_doc_num = 30, threshold = 0.25):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # embedder = SentenceTransformer('bert-base-nli-mean-tokens') #BERT BASE
    # embedder = SentenceTransformer('bert-large-nli-stsb-mean-tokens') # LARGE BERT
    f_corpus = pd.read_csv(input_corpus)
    corpus = list(f_corpus['Title_and_tag'])
    corpus_dic = {}
    for i in tqdm.tqdm(range(len(corpus))):
        corpus_dic[f_corpus.loc[i,'Title_and_tag']] = f_corpus.loc[i,'Title']
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    f_query = pd.read_csv(input_query)
    queries = list(f_query['Query Description'])

    f_query_id = []
    f_query_content = []
    f_tag = []
    f_title = []
    f_link = []
    f_category = []
    f_author = []
    f_image = []
    
    top_k = min(150, len(corpus))
    for q in tqdm.tqdm(range(len(queries))):
        query_embedding = embedder.encode(queries[q], convert_to_tensor=True)
        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        min_doc_num_idx = 0
        max_doc_num_idx = 0
        uniq_j = 0
        f_title_tmp = []
        for score, idx in zip(top_results[0], top_results[1]):
            if (score >= threshold or min_doc_num_idx <= min_doc_num) and max_doc_num_idx < max_doc_num:
                if corpus_dic[corpus[idx]] not in f_title_tmp:
                    # print(corpus[idx], "(Score: {:.4f})".format(score))
                    f_query_id.append(str(q))
                    f_query_content.append(queries[q])
                    f_title.append(corpus_dic[corpus[idx]])
                    idx = idx.item()
                    f_tag.append(f_corpus.loc[idx, 'Tag'])
                    f_link.append(f_corpus.loc[idx, 'Link'])
                    f_category.append(f_corpus.loc[idx, 'Category'])
                    f_author.append(f_corpus.loc[idx, 'Author'])
                    f_image.append(f_corpus.loc[idx, 'Image'])
                    uniq_j+=1
                    f_title_tmp.append(corpus_dic[corpus[idx]])
                    min_doc_num_idx += 1
                    max_doc_num_idx += 1
            else:
                break
        print("===",uniq_j)

    f_df = pd.DataFrame({'QueryId': f_query_id, 'QueryCtonten': f_query_content, 'Title': f_title, 'Tag': f_tag,
                         'Link': f_link, 'Category': f_category, 'Author': f_author, 'Image': f_image})
    f_df.to_csv(output_file, index=False)
    return 
    
    
    
if __name__ == '__main__':
    res_doc_num = 10
    ticks1 = time.time()
    # bm25('tag_10_bert_1000.csv', 'query_20.csv', 'a.csv', res_doc_num)
    # bert('tag_10_textrank_1000.csv', 'query_20.csv', 'out_10_textrank_bert_1000.csv', res_doc_num)
    bert('tag_10_textrank_2000.csv', 'query_20.csv', 'out_10_textrank_bert_2000.csv', 15, 30, 0.25)
    ticks2 = time.time()
    print(ticks2-ticks1)
    
    
    
