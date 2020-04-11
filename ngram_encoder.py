import nltk
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from tqdm import tqdm
from collections import Counter
import re
import heapq

df=pd.read_csv('metadata.csv')
abstracts=df['abstract']

ngram_count = 2

def string_cleaner(text):
    # Clean the documents
    stop = set(stopwords.words('english') + stopwords.words('spanish') + stopwords.words('french'))
    exclude = string.punctuation
    wordnet_lemmatizer = WordNetLemmatizer()
    start_strip_word = ['abstract', 'background', 'summary', 'objective']
    text = str(text).lower() # downcase
    for word in start_strip_word:
        if text.startswith(word):
            text = text[len(word):]
    tokens = nltk.tokenize.word_tokenize(text) # split string into words (tokens)
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stop] # remove stopwords
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    return tokens

def generate_ngram_mappings(text_tokens,ngram_count = 2):
    if len(text_tokens) == 1:
        text_tokens = [text_tokens]
    ngrams = {}
    for corpus in tqdm(text_tokens):
        for i in range(len(corpus) - ngram_count):
            seq = ' '.join(corpus[i:i + ngram_count])
            if seq not in ngrams.keys():
                ngrams[seq] = []
            ngrams[seq].append(corpus[i + ngram_count])
    return ngrams

abstract_tokens = [string_cleaner(abstract) for abstract in abstracts]
ngram_mapping = generate_ngram_mappings(abstract_tokens)

ngram_counts = {}
for (key,values) in ngram_mapping.items():
    ngram_counts[key] = dict(Counter(values))

all_counts = []
for count_dict in ngram_counts.values():
    # print(count_dict)
    all_counts.extend(list(count_dict.values()))

num_ngrams_to_keep = 5000
keep_count = heapq.nlargest(num_ngrams_to_keep, all_counts)[-1]

current_count_words = 0
for (key,values) in tqdm(ngram_counts.items()):
    trimmed_values = values.copy()
    for (k, v) in values.items():
        if v<keep_count or current_count_words>=5000:
            trimmed_values.pop(k)
        else:
            current_count_words+=1
    # if there is still a values in the dictionary re-assign the ngram with the trimmed dictionary
    # else, pop the current ngram, as there are no words that meet the current minimum count
    if trimmed_values:
        ngram_mapping[key]=list(trimmed_values.keys())
    else:
        ngram_mapping.pop(key)

ngram_columns = []
for (key,values) in tqdm(ngram_mapping.items()):
    for value in values:
        ngram_columns.append(" ".join([key,value]))

ngram_count_matrix = pd.DataFrame(0, index=np.arange(len(abstracts)),columns=ngram_columns)

for i in tqdm(range(len(abstracts))):
    tokens = abstract_tokens[i]
    for j in range(len(tokens) - ngram_count):
        seq = ' '.join(tokens[j:j + ngram_count + 1])
        if seq in ngram_columns:
            col_ind = ngram_columns.index(seq)
            ngram_count_matrix.iloc[i,col_ind]+=1



from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(ngram_count_matrix)
pca = PCA(n_components=100)
pca_df = pca.fit_transform(scaled_df)
print(sum(pca.explained_variance_ratio_))
kmeans = KMeans(n_clusters=10, random_state=0).fit(pca_df)
clusters = kmeans.predict(pca_df)
df["pca_clusters"] = clusters
df['pca_clusters'].value_counts()

#always more types of topic modeling Latent Discriminate Analysis
from sklearn.decomposition import NMF, LatentDirichletAllocation
nmf = NMF(n_components=100)
nnmf_df = nmf.fit_transform(ngram_count_matrix)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0).fit(nnmf_df)
clusters = kmeans.predict(nnmf_df)

df["nnmf_clusters"] = clusters
df['nnmf_clusters'].value_counts()



# hierarchal clustering
from sklearn.cluster import AgglomerativeClustering
agg=AgglomerativeClustering(n_clusters=10).fit(nnmf_df)
agg_df = agg.transform(nnmf_df)


# # Run LDA
# lda = LatentDirichletAllocation( max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(scaled_df)
#
# lda_df = lda.transform(scaled_df)