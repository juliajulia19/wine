import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, wordpunct_tokenize
import spacy
# spacy.cli.download('en_core_web_sm') #один раз указываем чтоб скачать модель на мак, потом закомментить, чтоб не скачивалась каждый раз
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

stop_words = stopwords.words('english')
Tfidf = TfidfVectorizer()

wine_df = pd.read_csv('winemag-data-130k-v2.csv')
#удаляю сразу столбцы, которые нам вряд ли пригодятся
wine_df = wine_df.drop(['designation', 'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'winery'], axis=1)
wine_df = wine_df[:5000] #взяла только первые 5000 строк иначе очень долго ждать
print(wine_df.shape)
print(wine_df.isnull().sum())

#предобработка столбца с описание
nlp_eng = spacy.load('en_core_web_sm')
def preprocess(text):
    tokenized = word_tokenize(text.lower())
    text_clean = []
    for word in tokenized:
        if word[0].isalnum() and word not in stop_words:
            text_clean.append(word)
    doc = nlp_eng(' '.join(text_clean)) #передаем в spacy и лемматизируем
    lemmas = []
    for token in doc:
        lemmas.append(token.lemma_)
    return ' '.join(lemmas)

#попробуем поделить на кластеры по столбцу с описанием
wine_df['description_preprocessed'] = wine_df['description'].apply(preprocess)
wine_df.to_csv('output.csv')
X = wine_df['description_preprocessed']

Tfidf.fit(X)
X_vec = Tfidf.transform(X)

count = CountVectorizer()
count.fit(X)
X_count = count.transform(X)

#попробуем разделить на кластеры с помощью k средних
k_means_clusters_5 = KMeans(n_clusters=5, random_state=0).fit(X_vec)
k_means_clusters_10 = KMeans(n_clusters=10, random_state=0).fit(X_vec)
k_means_clusters_15 = KMeans(n_clusters=15, random_state=0).fit(X_vec)
k_means_clusters_20 = KMeans(n_clusters=20, random_state=0).fit(X_vec)
k_means_clusters_3 = KMeans(n_clusters=3, random_state=0).fit(X_vec)
k_means_clusters_4 = KMeans(n_clusters=4, random_state=0).fit(X_vec)
k_means_clusters_7 = KMeans(n_clusters=7, random_state=0).fit(X_vec)
k_means_clusters_8 = KMeans(n_clusters=8, random_state=0).fit(X_vec)
k_means_clusters_12 = KMeans(n_clusters=12, random_state=0).fit(X_vec)
k_means_clusters_13 = KMeans(n_clusters=13, random_state=0).fit(X_vec)
k_means_clusters_17 = KMeans(n_clusters=17, random_state=0).fit(X_vec)
k_means_clusters_25 = KMeans(n_clusters=25, random_state=0).fit(X_vec)
k_means_clusters_30 = KMeans(n_clusters=30, random_state=0).fit(X_vec)
k_means_clusters_35 = KMeans(n_clusters=35, random_state=0).fit(X_vec)
k_means_clusters_40 = KMeans(n_clusters=40, random_state=0).fit(X_vec)
k_means_clusters_45 = KMeans(n_clusters=45, random_state=0).fit(X_vec)
k_means_clusters_50 = KMeans(n_clusters=50, random_state=0).fit(X_vec)
k_means_clusters_70 = KMeans(n_clusters=70, random_state=0).fit(X_vec)
k_means_clusters_100 = KMeans(n_clusters=100, random_state=0).fit(X_vec)

print(silhouette_score(X_vec, k_means_clusters_5.labels_))
print(silhouette_score(X_vec, k_means_clusters_10.labels_))
print(silhouette_score(X_vec, k_means_clusters_15.labels_))
print(silhouette_score(X_vec, k_means_clusters_20.labels_))

print(silhouette_score(X_vec, k_means_clusters_3.labels_))
print(silhouette_score(X_vec, k_means_clusters_4.labels_))
print(silhouette_score(X_vec, k_means_clusters_7.labels_))
print(silhouette_score(X_vec, k_means_clusters_8.labels_))
print(silhouette_score(X_vec, k_means_clusters_12.labels_))
print(silhouette_score(X_vec, k_means_clusters_13.labels_))
print(silhouette_score(X_vec, k_means_clusters_17.labels_))
print(silhouette_score(X_vec, k_means_clusters_25.labels_))
print(silhouette_score(X_vec, k_means_clusters_30.labels_))
print(silhouette_score(X_vec, k_means_clusters_35.labels_))
print(silhouette_score(X_vec, k_means_clusters_40.labels_))
print(silhouette_score(X_vec, k_means_clusters_45.labels_))
print(silhouette_score(X_vec, k_means_clusters_50.labels_))
print(silhouette_score(X_vec, k_means_clusters_70.labels_))
print(silhouette_score(X_vec, k_means_clusters_100.labels_))


k_means_clusters_5_c = KMeans(n_clusters=5, random_state=0).fit(X_count)
k_means_clusters_10_c = KMeans(n_clusters=10, random_state=0).fit(X_count)
k_means_clusters_15_c = KMeans(n_clusters=15, random_state=0).fit(X_count)
k_means_clusters_20_c = KMeans(n_clusters=20, random_state=0).fit(X_count)
print(silhouette_score(X_count, k_means_clusters_5_c.labels_))
print(silhouette_score(X_count, k_means_clusters_10_c.labels_))
print(silhouette_score(X_count, k_means_clusters_15_c.labels_))
print(silhouette_score(X_count, k_means_clusters_20_c.labels_))

#все score силуэтов меньше 0.01 очень плохо

#попробуем разделить на группы с помощью LDA
#пока не делим на тренировочное и тестовое множество, чтобы вывести perplexity и score, тк не знаю что в y определить как зависимую переменную

lda50 = LatentDirichletAllocation(n_components=50, random_state=0)
doc_topic50 = lda50.fit_transform(X_vec)


lda30 = LatentDirichletAllocation(n_components=30, random_state=0)
doc_topic30 = lda30.fit_transform(X_vec)


lda100 = LatentDirichletAllocation(n_components=100, random_state=0)
doc_topic100 = lda100.fit_transform(X_vec)

#посмотрим какие темы получились для первой модели

tf_feature_names = Tfidf.get_feature_names_out()
n_components = 10 #графики для первый 10 тем
def plot_top_words(model, feature_names, n_top_words, title, n_components, max_plots=10):
    fig, axes = plt.subplots(1, max_plots, figsize=(25, 10))  # параметры отображения
    axes = axes.flatten()
    all_features = {}  # словарь для сохранения ключевых слов для тем

    for topic_idx, topic in enumerate(model.components_):
        if topic_idx < max_plots:
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]

            # строка для сохранения темы и слов в словарь
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f'Topic {topic_idx + 1}',
                         fontdict={'fontsize': 13})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=10)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=14)

    plt.show()

plot_top_words(lda50, tf_feature_names, 20, 'Распределение слов по темам, LDA-модель', n_components) #20 самых весомых слов

#самые частотные слова первых 10 топиков
def topwords(model, features, ntopwords):
    result = {}
    for topic_idx, topic in enumerate(model.components_):
        result[topic_idx] = [features[i] for i in topic.argsort()[:-ntopwords - 1:-1]]
    return result

ntopwords = 10
top_words = topwords(lda50, tf_feature_names, ntopwords)
print(top_words)