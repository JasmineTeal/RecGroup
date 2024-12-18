# lda_recommender.py

import os
import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
from gensim.models import LdaModel
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import webbrowser
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import log2


# =========================
# 1. 数据加载与清理
# =========================

def load_and_clean_data(data_path):
    # 加载数据
    movies = pd.read_csv(os.path.join(data_path, 'movies.csv'))
    tags = pd.read_csv(os.path.join(data_path, 'tags.csv'))
    ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))

    # 清理tags数据
    tags = tags.dropna(subset=['tag'])
    tags['tag'] = tags['tag'].astype(str)
    tags = tags[tags['tag'].str.strip() != '']

    return movies, tags, ratings


# =========================
# 2. LDA模型训练
# =========================

def preprocess_tags(tags, movies):
    # 合并标签
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    movie_tags = pd.merge(movies[['movieId', 'title', 'genres']], movie_tags, on='movieId', how='left')

    # 对于没有标签的电影，填充为空字符串
    movie_tags['tag'] = movie_tags['tag'].fillna('')

    # 文本预处理（仅分词，不移除停用词）
    movie_tags['processed_tags'] = movie_tags['tag'].apply(lambda x: simple_preprocess(x, deacc=True))

    return movie_tags


def train_lda(movie_tags, num_topics=10):
    # 创建词典和语料库
    dictionary = corpora.Dictionary(movie_tags['processed_tags'])
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in movie_tags['processed_tags']]

    # 训练LDA模型
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)

    # 获取每个电影的主题分布
    def get_topic_distribution(text):
        bow = dictionary.doc2bow(text)
        topics = lda_model.get_document_topics(bow, minimum_probability=0)
        # 确保topics是一个列表
        if not isinstance(topics, list):
            return [(i, 0.0) for i in range(num_topics)]
        return topics

    movie_tags['topic_distribution'] = movie_tags['processed_tags'].apply(get_topic_distribution)

    # 转换为向量
    movie_tags['topic_vector'] = movie_tags['topic_distribution'].apply(
        lambda x: topics_to_vector(x, num_topics)
    )

    # 创建项目的主题矩阵
    movie_topic_matrix = np.vstack(movie_tags['topic_vector'].values)

    return lda_model, dictionary, corpus, movie_topic_matrix, movie_tags


def topics_to_vector(topics, num_topics):
    vector = np.zeros(num_topics)
    for topic, prob in topics:
        vector[topic] = prob
    return vector


# =========================
# 3. 构建用户画像
# =========================

def build_user_profiles(ratings, movie_tags, num_topics):
    # 合并评分和主题分布
    user_ratings = pd.merge(ratings, movie_tags[['movieId', 'topic_distribution']], on='movieId', how='left')

    # 填充缺失的topic_distribution为空列表
    user_ratings['topic_distribution'] = user_ratings['topic_distribution'].apply(
        lambda x: x if isinstance(x, list) else [(i, 0.0) for i in range(num_topics)]
    )

    # 构建用户画像
    user_profiles = user_ratings.groupby('userId').apply(build_user_profile, num_topics=num_topics).reset_index()
    user_profiles.columns = ['userId', 'profile']

    return user_profiles


def build_user_profile(group, num_topics):
    user_topics = np.zeros(num_topics)
    total_weight = 0
    for _, row in group.iterrows():
        topics = row['topic_distribution']
        rating = row['rating']
        weight = rating  # 可以根据需要调整权重，例如使用评分值
        total_weight += weight
        for topic, prob in topics:
            user_topics[topic] += prob * weight
    if total_weight > 0:
        user_topics /= total_weight
    return user_topics.tolist()


# =========================
# 4. 生成推荐
# =========================

def recommend_movies(user_id, user_profiles, movie_tags, movie_topic_matrix, top_n=10):
    # 获取用户的主题偏好
    user_profile = user_profiles[user_profiles['userId'] == user_id]['profile'].values
    if len(user_profile) == 0:
        return pd.DataFrame()
    user_vector = np.array(user_profile[0]).reshape(1, -1)

    # 计算相似度
    similarities = cosine_similarity(user_vector, movie_topic_matrix)[0]

    # 将相似度添加到movie_tags
    recommendations = movie_tags.copy()
    recommendations['similarity'] = similarities

    return recommendations.sort_values(by='similarity', ascending=False).head(top_n)


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2.T)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2, axis=1)
    # 防止除以零
    norm_vec2 = np.where(norm_vec2 == 0, 1e-10, norm_vec2)
    return (dot_product / (norm_vec1 * norm_vec2)).flatten()


# =========================
# 5. 评估指标计算
# =========================

def evaluate_precision_recall_f1(user_profiles, ratings, movie_tags, movie_topic_matrix, top_n=10):
    precision_list = []
    recall_list = []
    f1_list = []

    unique_users = user_profiles['userId'].unique()

    for user_id in unique_users:
        # 获取用户实际喜欢的电影（评分 >= 4）
        user_true = ratings[(ratings['userId'] == user_id) & (ratings['rating'] >= 4)]['movieId'].tolist()
        if len(user_true) == 0:
            continue  # 如果用户没有喜欢的电影，跳过

        # 获取推荐的电影
        recommended = recommend_movies(user_id, user_profiles, movie_tags, movie_topic_matrix, top_n=top_n)
        user_recommended = recommended['movieId'].tolist()

        # 计算交集
        hit = len(set(user_recommended) & set(user_true))

        # 计算指标
        precision = hit / top_n
        recall = hit / len(user_true) if len(user_true) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # 计算平均指标
    avg_precision = np.mean(precision_list) if precision_list else 0
    avg_recall = np.mean(recall_list) if recall_list else 0
    avg_f1 = np.mean(f1_list) if f1_list else 0

    return avg_precision, avg_recall, avg_f1


def calculate_dcg(relevance, k):
    dcg = 0.0
    for i in range(1, k + 1):
        if i - 1 < len(relevance) and relevance[i - 1]:
            dcg += 1 / log2(i + 1)
    return dcg


def calculate_ndcg(recommended, relevant, k):
    dcg = calculate_dcg(relevant[:k], k)
    ideal = calculate_dcg(sorted(relevant, reverse=True)[:k], k)
    if ideal == 0:
        return 0
    return dcg / ideal


def evaluate_ndcg(user_profiles, ratings, movie_tags, movie_topic_matrix, top_n=10):
    ndcg_list = []

    unique_users = user_profiles['userId'].unique()

    for user_id in unique_users:
        # 获取用户实际喜欢的电影（评分 >= 4）
        user_true = ratings[(ratings['userId'] == user_id) & (ratings['rating'] >= 4)]['movieId'].tolist()
        if len(user_true) == 0:
            continue

        # 获取推荐的电影
        recommended = recommend_movies(user_id, user_profiles, movie_tags, movie_topic_matrix, top_n=top_n)
        user_recommended = recommended['movieId'].tolist()

        # 创建相关性列表
        relevance = [1 if movie in user_true else 0 for movie in user_recommended]

        # 计算NDCG@K
        ndcg = calculate_ndcg(user_recommended, relevance, top_n)
        ndcg_list.append(ndcg)

    # 计算平均NDCG
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0
    return avg_ndcg


def evaluate_rmse_mae(user_profiles, ratings, movie_tags):
    predictions = []
    truths = []

    for _, row in ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        true_rating = row['rating']

        # 获取用户的主题偏好
        user_profile = user_profiles[user_profiles['userId'] == user_id]['profile'].values
        if len(user_profile) == 0:
            continue
        user_profile = np.array(user_profile[0])

        # 获取电影的主题分布
        movie_topic = movie_tags[movie_tags['movieId'] == movie_id]['topic_vector'].values
        if len(movie_topic) == 0:
            continue
        movie_topic = np.array(movie_topic[0])

        # 预测评分（点积）
        pred_rating = np.dot(user_profile, movie_topic)

        predictions.append(pred_rating)
        truths.append(true_rating)

    if not predictions or not truths:
        return None, None

    rmse = np.sqrt(mean_squared_error(truths, predictions))
    mae = mean_absolute_error(truths, predictions)

    return rmse, mae


# =========================
# 6. 可视化
# =========================

def visualize_lda(lda_model, corpus, dictionary, output_file='lda_visualization.html'):
    lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_vis, output_file)
    print(f"LDA可视化结果已保存为 {output_file}")

    # 自动在默认浏览器中打开HTML文件
    file_path = os.path.abspath(output_file)
    webbrowser.open(f'file://{file_path}')


# =========================
# 7. 主函数
# =========================

def main():
    # 设置数据路径
    data_path = '../dataset/ml-latest-small/'  # 请确保数据文件位于 'data/' 目录下

    # 1. 加载和清理数据
    movies, tags, ratings = load_and_clean_data(data_path)
    print("数据加载和清理完成。")

    # 2. 预处理标签并训练LDA模型
    movie_tags = preprocess_tags(tags, movies)
    num_topics = 4  # 主题数，可以根据需要调整
    lda_model, dictionary, corpus, movie_topic_matrix, movie_tags = train_lda(movie_tags, num_topics=num_topics)
    print("LDA模型训练完成。")

    # 打印一些示例的 topic_distribution
    print("\n示例的 topic_distribution:")
    print(movie_tags['topic_distribution'].head())

    # 3. 构建用户画像
    user_profiles = build_user_profiles(ratings, movie_tags, num_topics)
    print("用户画像构建完成。")

    # 4. 可视化LDA模型
    visualize_lda(lda_model, corpus, dictionary)

    # 5. 评估推荐系统
    avg_precision, avg_recall, avg_f1 = evaluate_precision_recall_f1(user_profiles, ratings, movie_tags,
                                                                     movie_topic_matrix, top_n=10)
    avg_ndcg = evaluate_ndcg(user_profiles, ratings, movie_tags, movie_topic_matrix, top_n=10)
    rmse, mae = evaluate_rmse_mae(user_profiles, ratings, movie_tags)

    print(f"\n=== 推荐系统评估结果 ===")
    print(f"平均 Precision@10: {avg_precision:.4f}")
    print(f"平均 Recall@10: {avg_recall:.4f}")
    print(f"平均 F1-Score@10: {avg_f1:.4f}")
    print(f"平均 NDCG@10: {avg_ndcg:.4f}")
    if rmse is not None and mae is not None:
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    else:
        print("RMSE 和 MAE 无法计算，因为没有足够的数据。")


if __name__ == '__main__':
    main()
