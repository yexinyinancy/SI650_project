import json
import spacy
import math
import pandas as pd
import tqdm
import warnings
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import warnings
import time

warnings.filterwarnings("ignore")

def IR_bert(input_corpus, input_query, min_doc_num = 15, max_doc_num = 30, threshold = 0.25):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # embedder = SentenceTransformer('bert-base-nli-mean-tokens') #BERT BASE
    # embedder = SentenceTransformer('bert-large-nli-stsb-mean-tokens') # LARGE BERT
    f_corpus = pd.read_csv(input_corpus)
    corpus = list(f_corpus['Title_and_tag'])
    corpus_dic = {}
    for i in tqdm.tqdm(range(len(corpus))):
        corpus_dic[f_corpus.loc[i,'Title_and_tag']] = f_corpus.loc[i,'Title']
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    f_query_content = []
    f_tag = []
    f_title = []
    f_link = []
    f_category = []
    f_author = []
    f_image = []
    
    top_k = min(100, len(corpus))
    
    query_embedding = embedder.encode(input_query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    min_doc_num_idx = 0
    max_doc_num_idx = 0
    # uniq_j = 0
    f_title_tmp = []
    for score, idx in zip(top_results[0], top_results[1]):
        if (score >= threshold or min_doc_num_idx <= min_doc_num) and max_doc_num_idx < max_doc_num:
            if corpus_dic[corpus[idx]] not in f_title_tmp:
                f_query_content.append(input_query)
                f_title.append(corpus_dic[corpus[idx]])
                idx = idx.item()
                f_tag.append(f_corpus.loc[idx, 'Tag'])
                f_link.append(f_corpus.loc[idx, 'Link'])
                f_category.append(f_corpus.loc[idx, 'Category'])
                f_author.append(f_corpus.loc[idx, 'Author'])
                f_image.append(f_corpus.loc[idx, 'Image'])
                # uniq_j += 1
                f_title_tmp.append(corpus_dic[corpus[idx]])
                min_doc_num_idx += 1
                max_doc_num_idx += 1
        else:
            break
        # print("===",uniq_j)

    f_df = pd.DataFrame({'QueryCtonten': f_query_content, 'Title': f_title, 'Tag': f_tag,
                         'Link': f_link, 'Category': f_category, 'Author': f_author, 'Image': f_image})
    return f_df
    

if __name__ == '__main__':
    tag_file = 'tag_10_tfidf_2000.csv'
    output_file = 'out.csv'
    input_query = 'historical'
    ticks1 = time.time()
    f_df = IR_bert(tag_file, input_query, 15, 30, 0.25)
    f_df.to_csv(output_file, index=False)
    ticks2 = time.time()
    print(ticks2-ticks1)
    