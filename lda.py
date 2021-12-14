# # Importing modules
# import pandas as pd

# # Load the regular expression library
# import re

# import gensim
# from gensim.utils import simple_preprocess
# import nltk
# # nltk.download('stopwords')
# from nltk.corpus import stopwords

# import gensim.corpora as corpora
# from pprint import pprint

# def sent_to_words(sentences):
#       for sentence in sentences:
#             # deacc=True removes punctuations
#             yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
      
# def remove_stopwords(texts):
#       return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
      
# if __name__ == '__main__':
#       papers = pd.read_csv('tag_10_tfidf.csv')    
#       # Remove punctuation
#       papers['Title_processed'] = papers['Tag'].map(lambda x: re.sub('[,\.!?]', '', x))
#       # Convert the titles to lowercase
#       papers['Title_processed'] = papers['Title_processed'].map(lambda x: x.lower())
      
#       stop_words = stopwords.words('english')
#       stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


#       data = papers.Title_processed.values.tolist()
#       data_words = list(sent_to_words(data))
#       # remove stop words
#       data_words = remove_stopwords(data_words)
#       # print(len(data_words))
#       # print(len(data_words[0]))
#       # print(len(data_words[0][0]))
#       # print(len(data_words[0][0][0]))
      
#       # Create Dictionary
#       id2word = corpora.Dictionary(data_words)
#       # Create Corpus
#       texts = data_words
#       # Term Document Frequency
#       corpus = [id2word.doc2bow(text) for text in texts]

#       # number of topics
#       num_topics = 10
#       # Build LDA model
#       lda_model = gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topics)
#       # Print the Keyword in the 10 topics
#       pprint(lda_model.print_topics())
#       doc_lda = lda_model[corpus]
      
def tokenization_with_gen_stop(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)

    return result

import nltk
# nltk.download('stopwords')
import gensim, spacy
import gensim.corpora as corpora
from nltk.corpus import stopwords

import pandas as pd
import re
from tqdm import tqdm
import time
import json

import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
# import matplotlib.pyplot as plt
# %matplotlib inline


if __name__ == '__main__':
      file = open('books_result.json', 'r')
      json_list = json.load(file)
      data = []
      for json_raw in tqdm(json_list):
            # for every product
            review_list = ''
            for rev in json_raw["reviews"]:
                  if 'body' in rev.keys():
                        review_list += rev['body']
                  else:
                        continue
            if review_list == '':
                  continue
            
            data.append(review_list)
      
      
      
      ## Setup nlp for spacy
      nlp = spacy.load("en_core_web_sm")

      # Load NLTK stopwords
      stop_words = stopwords.words('english')
      # Add some extra words in it if required
      stop_words.extend(['from', 'subject', 'use','pron'])

      # df = pd.read_csv('tag_10_tfidf.csv')    
      
      # Convert into list
      # data = df.Title.values.tolist()
      ### Cleaning data
      # Remove Emails
      data = [re.sub('S*@S*s?', '', sent) for sent in data]
      # Remove new line characters and extra space
      data = [re.sub('s+', ' ', sent) for sent in data]
      # Remove single quotes
      data = [re.sub("'", "", sent) for sent in data]
      ### Lemmatization
      data_lemma = []
      for txt in tqdm(data):
            lis = []
            doc = nlp(txt)
            for token in doc:
                  lis.append(token.lemma_)
            data_lemma.append(' '.join(lis))

      ## Apply tokenization function
      data_words = []
      for txt in tqdm(data_lemma):
            data_words.append(tokenization_with_gen_stop(txt))

      ### NLTK Stopword removal (extra stopwords)

      data_words_clean = []
      for word in tqdm(data_words):
            wrd = []
            for w in word:
                  if w not in stop_words:
                        wrd.append(w)
            data_words_clean.append(wrd)
            
      # Create Dictionary
      dictionary = corpora.Dictionary(data_words_clean)
      # Print dictionary
      print(dictionary.token2id)

      ## Create Term document frequency (corpus)
      # Term Document Frequency
      corpus = [dictionary.doc2bow(text) for text in data_words_clean]
      # Print corpus for first document
      print(corpus[0])
      
      # Easy to observe format of corpus
      [[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]]
      
      start_time = time.time()
      ##
      NUM_TOPICS = 15
      ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary,random_state=100,passes=10)
      # Saving trained model
      ldamodel.save('LDA_NYT')
      # Loading trained model
      ldamodel = gensim.models.ldamodel.LdaModel.load('LDA_NYT')
      ## Print time taken to train the model
      print("--- %s seconds ---" % (time.time() - start_time))
      
      print(ldamodel.print_topics(-1))
      
      # Compute Perplexity Score
      print('nPerplexity Score: ', ldamodel.log_perplexity(corpus))
      # Compute Coherence Score
      coherence_model_lda = gensim.models.CoherenceModel(model=ldamodel, texts=data_words_clean, dictionary=dictionary, coherence='c_v')
      coherence_lda = coherence_model_lda.get_coherence()
      print('nCoherence Score: ', coherence_lda)
      
      
      
      # predict
      ## Keeping first content of dataframe as our new document
      new_doc = 'history'
      ### Cleaning data
      # Remove Emails
      data = re.sub('S*@S*s?', '', new_doc)
      # Remove new line characters and extra space
      data = re.sub('s+', ' ', data)
      # Remove single quotes
      data = re.sub("'", "", data)
      ### Lemmatization
      data_lemma = []
      lis = []
      doc = nlp(data)
      for token in doc:
            lis.append(token.lemma_)
            data_lemma.append(' '.join(lis))
      ## Apply tokenization function
      data_words = []
      for txt in tqdm(data_lemma):
            data_words.append(tokenization_with_gen_stop(txt))
      ### NLTK Stopword removal (extra stopwords)
      data_words_clean_new = []
      for word in tqdm(data_words):
            for w in word:
                  if w not in stop_words:
                        data_words_clean_new.append(w)
      # Create corpus for new document
      corpus_new = dictionary.doc2bow(data_words_clean_new)
      print(corpus_new)
      print(ldamodel.get_document_topics(corpus_new))


      
      


      