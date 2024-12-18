import pandas as pd


def load_data():
    # 文件路径（根据实际情况修改路径）
    ratings_path = '../dataset/ml-latest-small/ratings.csv'
    movies_path = '../dataset/ml-latest-small/movies.csv'
    tags_path = '../dataset/ml-latest-small/tags.csv'

    # 加载评分数据
    ratings = pd.read_csv(
        ratings_path,
        names=['userId', 'movieId', 'rating', 'timestamp'],
        sep=',',
        skiprows=1,  # 跳过标题行
        encoding='utf-8'
    )

    # 加载电影数据
    movies = pd.read_csv(
        movies_path,
        names=['movieId', 'title', 'genres'],
        sep=',',
        skiprows=1,
        encoding='utf-8'
    )

    # 加载标签数据
    tags = pd.read_csv(
        tags_path,
        names=['userId', 'movieId', 'tag', 'timestamp'],
        sep=',',
        skiprows=1,
        encoding='utf-8'
    )



    return ratings, movies, tags


# 加载数据
ratings, movies, tags = load_data()

print("Ratings Data Shape:", ratings.shape)
print("Movies Data Shape:", movies.shape)
print("Tags Data Shape:", tags.shape)

from collections import Counter


def clean_tags(tags, min_freq=5):
    # 转换为小写并去除空格
    tags['tag'] = tags['tag'].str.lower().str.strip()

    # 统计标签频率
    tag_counts = Counter(tags['tag'])

    # 过滤低频标签
    tags = tags[tags['tag'].isin([tag for tag, count in tag_counts.items() if count >= min_freq])]

    return tags


# 清理标签
tags_cleaned = clean_tags(tags, min_freq=5)
print("Cleaned Tags Data Shape:", tags_cleaned.shape)

import networkx as nx


def build_heterogeneous_graph(ratings, movies, tags):
    G = nx.Graph()

    # 添加用户节点
    users = ratings['userId'].unique()
    G.add_nodes_from(users, bipartite='users')

    # 添加电影节点
    movies_ids = movies['movieId'].unique()
    G.add_nodes_from(movies_ids, bipartite='movies')

    # 添加标签节点
    unique_tags = tags['tag'].unique()
    G.add_nodes_from(unique_tags, bipartite='tags')

    # 添加用户-电影边，权重为评分
    for _, row in ratings.iterrows():
        G.add_edge(row['userId'], row['movieId'], weight=row['rating'])

    # 添加用户-标签边，权重为标签使用次数
    user_tag_counts = tags.groupby(['userId', 'tag']).size().reset_index(name='count')
    for _, row in user_tag_counts.iterrows():
        G.add_edge(row['userId'], row['tag'], weight=row['count'])

    # 添加标签-电影边，基于标签在电影中的出现频率
    tag_movie_counts = tags.groupby(['tag', 'movieId']).size().reset_index(name='count')
    for _, row in tag_movie_counts.iterrows():
        G.add_edge(row['tag'], row['movieId'], weight=row['count'])

    return G


# 构建图
G = build_heterogeneous_graph(ratings, movies, tags_cleaned)
print("Graph has {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))


def personalized_pagerank_recommendation(G, user_id, top_n=10, alpha=0.85):
    # Personalized PageRank 以用户为种子节点
    personalization = {node: 0 for node in G.nodes()}
    personalization[user_id] = 1

    # 计算 PageRank
    pagerank_scores = nx.pagerank(G, alpha=alpha, personalization=personalization)

    # 筛选电影节点并排序
    movies = [node for node, attr in G.nodes(data=True) if attr.get('bipartite') == 'movies']
    scored_movies = {movie: score for movie, score in pagerank_scores.items() if movie in movies}

    # 排序并返回 top N
    recommended_movies = sorted(scored_movies.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommended_movies


# 示例推荐
user_id_example = ratings['userId'].iloc[0]
pagerank_recommended = personalized_pagerank_recommendation(G, user_id_example, top_n=10)
print("Personalized PageRank 推荐结果：")
for movie_id, score in pagerank_recommended:
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    print(f"电影ID: {movie_id}, 评分: {score:.6f}, 标题: {title}")

from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def node2vec_embedding(G, dimensions=64, walk_length=30, num_walks=200, workers=4):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers,
                        quiet=True)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model


def node2vec_recommendation(model, user_id, movies, top_n=10):
    # 获取用户向量
    user_str = str(user_id)
    if user_str not in model.wv:
        print(f"用户 {user_id} 不在嵌入模型中。")
        return []
    user_vector = model.wv[user_str]

    # 获取所有电影向量
    movie_ids = [str(movie) for movie in movies]
    available_movies = [movie for movie in movie_ids if movie in model.wv]
    if not available_movies:
        return []
    movie_vectors = np.array([model.wv[movie] for movie in available_movies])

    # 计算余弦相似度
    similarities = cosine_similarity([user_vector], movie_vectors)[0]

    # 获取 top N 电影
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommended_movies = [int(movies[i]) for i in top_indices[:top_n]]

    return recommended_movies


# 训练 Node2Vec 模型（此过程可能需要一些时间）
node2vec_model = node2vec_embedding(G)

# 示例推荐
node2vec_recommended_ids = node2vec_recommendation(node2vec_model, user_id_example, movies['movieId'].tolist(),
                                                   top_n=10)
print("\nNode2Vec 推荐结果：")
for movie_id in node2vec_recommended_ids:
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    print(f"电影ID: {movie_id}, 标题: {title}")

from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation


def label_propagation_recommendation(G, user_id, top_n=10):
    # 获取用户相关标签
    user_tags = [n for n in G.neighbors(user_id) if G.nodes[n]['bipartite'] == 'tags']

    # 获取这些标签关联的电影
    recommended_movies = set()
    for tag in user_tags:
        movies = [n for n in G.neighbors(tag) if G.nodes[n]['bipartite'] == 'movies']
        recommended_movies.update(movies)

    # 排序推荐的电影（基于标签权重）
    movie_scores = {}
    for movie in recommended_movies:
        neighbors = G.neighbors(movie)
        score = sum(
            [G[user_id][tag]['weight'] for tag in user_tags if G.has_edge(user_id, tag) and G.has_edge(tag, movie)])
        movie_scores[movie] = score

    # 返回 top N 电影
    recommended_movies_sorted = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommended_movies_sorted


# 示例推荐
label_prop_recommended = label_propagation_recommendation(G, user_id_example, top_n=10)
print("\n标签传播推荐结果：")
for movie_id, score in label_prop_recommended:
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    print(f"电影ID: {movie_id}, 分数: {score}, 标题: {title}")

from sklearn.model_selection import train_test_split

def split_data(ratings, test_size=0.2):
    train, test = train_test_split(ratings, test_size=test_size, random_state=42)
    return train, test

# 划分数据
train_ratings, test_ratings = split_data(ratings, test_size=0.2)
print("训练集大小:", train_ratings.shape)
print("测试集大小:", test_ratings.shape)

# 清理测试集中的标签
tags_train = tags_cleaned[tags_cleaned['userId'].isin(train_ratings['userId'].unique()) & tags_cleaned['movieId'].isin(train_ratings['movieId'].unique())]

# 构建图
G_train = build_heterogeneous_graph(train_ratings, movies, tags_train)
print("训练图有 {} 个节点和 {} 条边".format(G_train.number_of_nodes(), G_train.number_of_edges()))

# 训练 Node2Vec 模型
node2vec_model_train = node2vec_embedding(G_train)

from sklearn.metrics import mean_squared_error


def evaluate_recommendation(recommended, test_user_movies, k=10):
    recommended_set = set([movie for movie in recommended[:k]])
    relevant_set = set(test_user_movies)

    true_positives = recommended_set & relevant_set
    precision = len(true_positives) / len(recommended_set) if recommended_set else 0
    recall = len(true_positives) / len(relevant_set) if relevant_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # 计算 RMSE（假设推荐评分为预测评分）
    # 这里简化为计算推荐电影的实际评分与平均评分的 RMSE
    if len(test_user_movies) == 0:
        rmse = None
    else:
        # 获取推荐电影在测试集中的实际评分
        recommended_ratings = test_ratings[
            (test_ratings['userId'] == user_id_example) & (test_ratings['movieId'].isin(recommended_set))]
        if not recommended_ratings.empty:
            predicted_ratings = [train_ratings['rating'].mean()] * len(recommended_ratings)
            actual_ratings = recommended_ratings['rating'].values
            rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        else:
            rmse = None

    return precision, recall, f1, rmse


# 获取测试集用户的实际评分
test_user = test_ratings[test_ratings['userId'] == user_id_example]
test_user_movies = test_user['movieId'].tolist()

# Personalized PageRank 推荐
pagerank_recommended = personalized_pagerank_recommendation(G_train, user_id_example, top_n=10)

# Node2Vec 推荐
node2vec_recommended_ids = node2vec_recommendation(node2vec_model_train, user_id_example, movies['movieId'].tolist(), top_n=10)

# 标签传播推荐
label_prop_recommended = label_propagation_recommendation(G_train, user_id_example, top_n=10)

# 评估推荐结果
# Personalized PageRank
pagerank_movie_ids = [movie_id for movie_id, _ in pagerank_recommended]
pagerank_metrics = evaluate_recommendation(pagerank_movie_ids, test_user_movies, k=10)

# Node2Vec
node2vec_metrics = evaluate_recommendation(node2vec_recommended_ids, test_user_movies, k=10)

# 标签传播
label_prop_movie_ids = [movie_id for movie_id, _ in label_prop_recommended]
label_prop_metrics = evaluate_recommendation(label_prop_movie_ids, test_user_movies, k=10)

# 展示结果
import pandas as pd

results = pd.DataFrame({
    '算法': ['Personalized PageRank', 'Node2Vec', '标签传播'],
    '准确率': [pagerank_metrics[0], node2vec_metrics[0], len(label_prop_recommended) > 0 and label_prop_metrics[0] or 0],
    '召回率': [pagerank_metrics[1], node2vec_metrics[1], len(label_prop_recommended) > 0 and label_prop_metrics[1] or 0],
    'F1 分数': [pagerank_metrics[2], node2vec_metrics[2], len(label_prop_recommended) > 0 and label_prop_metrics[2] or 0],
    'RMSE': [pagerank_metrics[3] if pagerank_metrics[3] else np.nan,
             node2vec_metrics[3] if node2vec_metrics[3] else np.nan,
             label_prop_metrics[3] if label_prop_metrics[3] else np.nan]
})

print("\n推荐算法评估结果：")
print(results)
