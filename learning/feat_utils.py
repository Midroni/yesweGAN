import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim #for model loading
from functools import reduce
from nltk.sentiment.vader import SentimentIntensityAnalyzer


#-----------------------------------HELPER-FUNCTIONS-------------------------
def generate_ngrams(string, n):
    '''
    Break text into tokens and return a list of the desired ngrams from the input string.
    '''
    #break text in tokens, not counting empty tokens
    tokens = [token for token in string.split(" ") if token != ""]
    
    #use the zip function to generate the desired n_gram list
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def cos_sim(A,B):
    '''
    Return the cosine similarity of two matrices.
    '''
    return cosine_similarity(A,B)[0][0]

def is_string(string):
    if not isinstance(string, str):
        raise ValueError('input to "col" must of type str') 

def try_divide(a,b):
    return b if b == 0 else a/b


def create_term_dict(corpus, threshold=False):
    '''
    Returns the term dictionary of a corpus.
    
    Parameters
    ----------
    corpus:
        A text corpus of shape [num_documents,1]. The 2nd dimension contains the text sample for that document.
    threshold:
        The minimum number of documents a term must appear in for it to be added to the term dictionary.

    Returns
    -------
    term_dict:
        A dictionary for which the key stores all unique terms, and the value for each key is a list of 
        indices for the documents in the corpus which contain the term. Duplicate indices indicate 
        multiple appearances within the same document.
    '''
    term_dict = {}
    for idx, document in enumerate(corpus):
        for term in document:
            if term in set(term_dict.keys()): #if the term is already in the keys, append
                term_dict[term].append(idx)
            else: #otherwise, add new key
                term_dict[term] = [idx]
    
    #automatically remove terms which don't appear in 'threshold' documents
    if threshold:
        for term in list(term_dict): #list allows us to change dictionary in iteration
            if len(term_dict[term]) <= threshold:
                term_dict.pop(term) #delete that term
    return term_dict

def create_term_doc_matrix(term_dict,corpus):
    '''
    Returns the term-document matrix of a corpus of shape [num_documents,].
    
    Parameters
    ----------
    term_dict:
        The term dictionary of corpus.
    corpus:
         The corpus of text from which the term_dictionary has been made.

    Returns
    -------
    term_doc:
        A term-document matrix of shape [num_documents, num_terms] with 
        (rows, columns) corresponding to (documents, terms). Values at the [i,j]th
        index indicate the number of times term j appears in document i.   
    '''
    A = np.zeros([len(corpus),len(term_dict)]) #rows x col = doc x terms
    for idx, term in enumerate(term_dict):
        for d in term_dict[term]:
            A[d,idx] += 1 
    return np.asarray(A)

def tfidf_matrix(term_doc):
    '''
    Returns the term-frequency-inverse-document-frequency matrix of a term-document matrix.
    
    Parameters
    ----------
    term_doc:
        The term-document matrix of a corpus of words. 

    Returns
    -------
    tfidf_matrix:
        A matrix of tf-idf values for each term-document relationship with
        (rows, columns) corresponding to (documents, terms). 
    '''
    col_sums = np.sum(term_doc, axis=0)
    A = np.zeros(term_doc.shape)
    B = np.copy(term_doc)
    for i in range(term_doc.shape[0]):
        for j in range(term_doc.shape[1]):
            term_count = col_sums[j] #number of terms in the document j
            tf = B[i,j] / term_count #divide all rows by the frequency per term
            
            row_i = list(B[i]) 
            row_i = [d for d in row_i if d>0] #filter out docs that don't have the term
            nt = len(row_i) #nt is the number of documents which have the term
            
            idf = log(float(term_doc.shape[1])) / nt
            A[i,j] = tf*idf
    return A

#-----------------------------------COUNT-FEATURES--------------------------
def add_feature_ngram_count(df,col,n):
    '''
    Adds a count of the number of ngrams for a desired column in a dataframe.
    '''
    is_string(col) #value_test     
    df['count_%sgram_%s' %(n,col)] = df[col].apply(lambda x: len(generate_ngrams(x,n)))

def add_feature_unique_ngram_count(df,col,n):
    '''
    Adds a count of the unique number of ngrams for a desired column in a dataframe.
    '''
    is_string(col) #value_test
    df['count_unique_%sgram_%s' %(n,col)] = df.apply(lambda x: len(set(generate_ngrams(x[col],n))),axis=1)

def add_feature_unique_ngram_ratio(df,col,n):
    '''
    Adds a feature column of ratios which indicate the proportion of the string's text which is composed of unique ngrams.
    '''
    is_string(col) #value_test
    df['ratio_unique_%sgram_%s' %(n,col)] = \
    list(map(try_divide, df['count_unique_%sgram_%s' %(n,col)], df['count_%sgram_%s' %(n,col)]))

def add_feature_overlap_ngrams(df,col1,col2,n):
    '''
    Adds the number overlapping ngrams between two desired columns in a dataframe.
    '''
    is_string(col1) #value_test
    is_string(col2) #value_test
    df['overlap_%sgram_%s_%s' %(n,col1,col2)] = \
    list(df.apply(lambda x: sum([1.0 for s in generate_ngrams(x[col1],n) if s in set(generate_ngrams(x[col2],n))]),axis=1))

def add_feature_sentence_count(df,col):
    '''
    Adds the sentence count feature for a desired column in a pandas dataframe.
    '''
    is_string(col) #value_test
    df['num_sents_%s' %(col)] = df[col].apply(lambda x: len(sent_tokenize(x)))

def add_feature_word_count(df,col):
    '''
    Adds the word_count feature for a desired column in a pandas dataframe.
    '''
    is_string(col) #value_test
    df['num_word_%s' %(col)] = df[col].apply(lambda x: len(str(x).split(" ")))

def add_feature_character_count(df,col):
    '''
    Adds the char_count feature for a desired column in a pandas dataframe.
    '''
    is_string(col) #value_test
    df['num_chars_%s' %(col)] = df[col].apply(lambda x: len(str(x)))
    
    
#-----------------------------------TFIDF-FEATURES--------------------------

def add_feature_tfidf_svd_similarity(df,col1,col2,vec):
    '''
    Adds the cosine similarity feature for the tfidf matrices of each column
    
    Parameters
    ----------
    df:
        The dataframe which the row is added to.
    col1:
        The first column of text for the similarity measurements. 
    col2:
        The second column of text for the similarity measurements.
    n:
        The ngram number to include in the similarity comparison.
        
    Returns
    -------
    N/A 
    '''
    vec.fit(np.add(df[col1].values,df[col2].values)) #train on all text contained in the two columns
    vocabulary = vec.vocabulary_ 
    
    new_vec= TfidfVectorizer(ngram_range=vec.ngram_range,max_df=vec.max_df,
                            min_df=vec.min_df,vocabulary=vec.vocabulary_)
    tfidf1 = new_vec.fit_transform(df[col1])
    new_vec= TfidfVectorizer(ngram_range=vec.ngram_range,max_df=vec.max_df,
                            min_df=vec.min_df,vocabulary=vec.vocabulary_)
    tfidf2 = new_vec.fit_transform(df[col2])
    
    df['sim_tfidf_%s_to_%sgram_%s_%s' %(vec.ngram_range[0],vec.ngram_range[1],col1,col2)] = list(map(cos_sim, tfidf1,tfidf2))
    
#-----------------------------------WORD2VEC-FEATURES------------------------

def add_feature_w2v_similarity(df,col1,col2,model,n):
    '''
    Adds the Word2Vec cosine similarity feature to the dataframe between two specific columns.
    
    Parameters
    ----------
    df:
        The dataframe which the row is added to.
    col1:
        The first column of text for the similarity measurements. 
    col2:
        The second column of text for the similarity measurements.
    model:
        A gensim model loaded with Word2Vec embeddings.
    n:
        The ngram number to include in the similarity comparison.
        
    Returns
    -------
    N/A 
    '''
    if not model:
        raise ValueError('You must pass in a model with a loaded Word2Vec embedding')
        
    gram_list1 = df[col1].map(lambda x: generate_ngrams(x,n)).values
    gram_list2 = df[col2].map(lambda x: generate_ngrams(x,n)).values
    
    vecs1 = map(lambda x: reduce(np.add, [model[gram] for gram in x if gram in model], [0.]*300), gram_list1)
    vecs2 = map(lambda x: reduce(np.add, [model[gram] for gram in x if gram in model], [0.]*300), gram_list2)
    #get values from map by calling the generator in a list
    vecs1 = list(vecs1)
    vecs2 = list(vecs2)
    #shape into necessary form
    vecs1 = [np.reshape(x,newshape=[1,-1]) for x in vecs1]
    vecs2 =[np.reshape(x,newshape=[1,-1]) for x in vecs2]
    df['sim_w2v_%s_%s' %(col1,col2)] = w2v_sims = np.squeeze(list(map(cos_sim, vecs1,vecs2)))

#-----------------------------------SENTIMENT-FEATURES-----------------------
    
def add_feature_sentiment_sid(df,col,avg=False):
    '''
    Adds features from the sentiment of the sentences as processed by nltk SentimentIntensityAnalyzer().
    
    Parameters
    ----------
    df:
        The dataframe which the row is added to.
    col:
        The column of text to be analyzed. 
    avg:
        A binary True/False value of whether or not to average the sentiment.
        Sentiment should be averaged if your sentences parameter is a list of lists.
        
    Returns
    -------
    N/A
    '''
    sid = SentimentIntensityAnalyzer()
    sentences = df[col].values
    result = list()
    for sentence in sentences:
        vs = sid.polarity_scores(sentence)
        result.append(vs)
    if avg:
        df2 = pd.DataFrame(result).mean()
        columns = ['sent_'+ x + '_' + col for x in df2.columns]
        df2.columns = columns
        for key in df2:
            df[key] = df2[key]
    else:
        df2 = pd.DataFrame(result)
        columns = ['sent_'+ x + '_' + col for x in df2.columns]
        df2.columns = columns
        for key in df2:
            df[key] = df2[key]