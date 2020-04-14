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
import tensorflow as tf
import pickle as p

pd.set_option('display.expand_frame_repr', False)

df=pd.read_csv('metadata.csv')
df["publish_year"] = list(map(lambda date: int(date[:4]) if type(date)==str else 0,df['publish_time']))
df = df[df["publish_year"]>=2000].reset_index(drop=True)
abstracts=df['abstract']

ngram_count = 1
num_words_to_keep = 20000
num_ngrams_to_keep = 15000

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

# text_tokens = [string_cleaner(abstract) for abstract in abstracts]
# with open("abstract_tokens.p","wb") as handle:
#     p.dump(text_tokens,handle)

with open("abstract_tokens.p","rb") as handle:
    text_tokens = p.load(handle)


#
# from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# scaler = MinMaxScaler()
# scaled_df = scaler.fit_transform(ngram_count_matrix)
# pca = PCA(n_components=100)
# pca_df = pca.fit_transform(scaled_df)
# print(sum(pca.explained_variance_ratio_))
# kmeans = KMeans(n_clusters=10, random_state=0).fit(pca_df)
# clusters = kmeans.predict(pca_df)
# df["pca_clusters"] = clusters
# df['pca_clusters'].value_counts()

# #always more types of topic modeling Latent Discriminate Analysis
from sklearn.decomposition import NMF, LatentDirichletAllocation
# nmf = NMF(n_components=100)
# nnmf_df = nmf.fit_transform(ngram_count_matrix)
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=10, random_state=0).fit(nnmf_df)
# clusters = kmeans.predict(nnmf_df)
#
# df["nnmf_clusters"] = clusters
# df['nnmf_clusters'].value_counts()
from sklearn.feature_extraction.text import CountVectorizer

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english',max_features=num_words_to_keep,ngram_range=(2,2))
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform([" ".join(tokens) for tokens in text_tokens])

# Helper function (pulled from https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0)
import seaborn as sns
import matplotlib.pyplot as plt
def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(", ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)


number_topics = 12
number_words = 10



# Run LDA
lda = LatentDirichletAllocation(n_components=number_topics, max_iter=100,random_state=0,n_jobs=-1)
lda.fit(count_data)
lda_data = pd.DataFrame(lda.transform(count_data), columns = ["Topic_"+str(i) for i in range(number_topics)])
agg_df = pd.concat([df,lda_data], axis=1)

aggregations = {"Topic_"+str(i):'mean' for i in range(number_topics)}
aggregations["url"] = "count"
agg_df = agg_df.groupby("publish_year").agg(aggregations)

# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)


plt.figure()
plt.plot(agg_df["url"])
plt.xlabel('Years')
plt.ylabel('Document Count')


plt.figure()
plt.plot(agg_df["Topic_3"], label="COVID-19 Topic")
plt.plot(agg_df["Topic_5"], label="MERS Topic")
plt.plot(agg_df["Topic_6"], label="SARS Topic")
plt.xlabel('Years')
plt.ylabel('Topic Percent Relevance')
plt.legend(loc='upper left')

plt.figure()
plt.plot(agg_df["Topic_2"], label="Virus Transmissions Topic")
plt.plot(agg_df["Topic_4"], label="Vaccine Topic")
plt.plot(agg_df["Topic_0"], label="Modeling/Immune Response Topic")
plt.xlabel('Years')
plt.ylabel('Topic Percent Relevance')
plt.legend(loc='upper left')

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# # scaler = MinMaxScaler()
# # scaled_df = scaler.fit_transform(ngram_count_matrix)
# pca = PCA(n_components=25)
# pca_df = pca.fit_transform(count_data.todense())
# print(sum(pca.explained_variance_ratio_))
# kmeans = KMeans(n_clusters=10, random_state=0).fit(pca_df)
# clusters = kmeans.predict(pca_df)
# df["pca_clusters"] = clusters
# df['pca_clusters'].value_counts()


# lda_df = lda.transform(scaled_df)

#
# # hierarchal clustering
# from sklearn.cluster import AgglomerativeClustering
# agg=AgglomerativeClustering(n_clusters=10).fit(nnmf_df)
# agg_df = agg.transform(nnmf_df)

from pyLDAvis import sklearn as sklearn_lda
import pickle
import pyLDAvis
import os

LDAvis_data_filepath = os.path.join('./ldavis_prepared_' + str(number_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
with open(LDAvis_data_filepath, 'wb') as f:
    pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath) as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_' + str(number_topics) + '.html')