import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score, mean_squared_error, mean_absolute_error

# 1. 数据加载与预处理
ratings = pd.read_csv('../dataset/ml-latest-small/ratings.csv')
tags = pd.read_csv('../dataset/ml-latest-small/tags.csv')
movies = pd.read_csv('../dataset/ml-latest-small/movies.csv')

# 标签清理
tag_counts = tags['tag'].value_counts()
filtered_tags = tags[tags['tag'].isin(tag_counts[tag_counts >= 5].index)]
filtered_tags['tag'] = filtered_tags['tag'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()
tag_counts = filtered_tags['tag'].value_counts()
filtered_tags = filtered_tags[filtered_tags['tag'].isin(tag_counts[tag_counts >= 5].index)]
print(f"清理后标签数量: {filtered_tags['tag'].nunique()}")

# 2. 构建图
G = nx.Graph()

# 添加用户节点，统一为字符串并添加前缀
users = filtered_tags['userId'].unique()
users_str = [f"user_{u}" for u in users]
G.add_nodes_from(users_str, bipartite='users')

# 添加标签节点，添加前缀
tags_unique = filtered_tags['tag'].unique()
tags_str = [f"tag_{t}" for t in tags_unique]
G.add_nodes_from(tags_str, bipartite='tags')

# 添加用户-标签边，使用带前缀的节点标签
edges = list(zip([f"user_{u}" for u in filtered_tags['userId']], [f"tag_{t}" for t in filtered_tags['tag']]))
G.add_edges_from(edges)

print(f"图中节点数: {G.number_of_nodes()}")
print(f"图中边数: {G.number_of_edges()}")

# 3. 节点嵌入（Node2Vec）
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4, p=1, q=1, seed=42)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# 提取用户和标签的嵌入向量
user_embeddings = {node: model.wv[node] for node in users_str if node in model.wv}
tag_embeddings = {node: model.wv[node] for node in tags_str if node in model.wv}

print(f"用户嵌入向量维度: {len(next(iter(user_embeddings.values())))}")
print(f"标签嵌入向量维度: {len(next(iter(tag_embeddings.values())))}")

# 4. 推荐生成
user_ids = list(user_embeddings.keys())
tag_ids = list(tag_embeddings.keys())
user_matrix = np.array([user_embeddings[user] for user in user_ids])
tag_matrix = np.array([tag_embeddings[tag] for tag in tag_ids])
similarity_matrix = cosine_similarity(user_matrix, tag_matrix)
user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
tag_id_to_index = {tag_id: idx for idx, tag_id in enumerate(tag_ids)}


def recommend_tags(user_id, top_m=10):
    user_node = f"user_{user_id}"
    if user_node not in user_id_to_index:
        return []
    user_idx = user_id_to_index[user_node]
    scores = similarity_matrix[user_idx]
    top_indices = scores.argsort()[-top_m:][::-1]
    recommended_tags = [tag_ids[idx].replace('tag_', '') for idx in top_indices]
    return recommended_tags


# 5. 评估与验证
train_tags, test_tags = train_test_split(filtered_tags, test_size=0.2, random_state=42)

# 构建训练图
train_G = nx.Graph()
train_users_str = [f"user_{u}" for u in train_tags['userId'].unique()]
train_tags_str = [f"tag_{t}" for t in train_tags['tag'].unique()]
train_G.add_nodes_from(train_users_str, bipartite='users')
train_G.add_nodes_from(train_tags_str, bipartite='tags')
train_edges = list(zip([f"user_{u}" for u in train_tags['userId']], [f"tag_{t}" for t in train_tags['tag']]))
train_G.add_edges_from(train_edges)

print(f"训练图中节点数: {train_G.number_of_nodes()}")
print(f"训练图中边数: {train_G.number_of_edges()}")

# 重新训练Node2Vec模型
train_node2vec = Node2Vec(train_G, dimensions=64, walk_length=30, num_walks=200, workers=4, p=1, q=1, seed=42)
train_model = train_node2vec.fit(window=10, min_count=1, batch_words=4)

# 生成嵌入向量
train_user_embeddings = {node: train_model.wv[node] for node in train_users_str if node in train_model.wv}
train_tag_embeddings = {node: train_model.wv[node] for node in train_tags_str if node in train_model.wv}

print(f"训练用户嵌入向量维度: {len(next(iter(train_user_embeddings.values())))}")
print(f"训练标签嵌入向量维度: {len(next(iter(train_tag_embeddings.values())))}")

# 构建相似度矩阵
train_user_ids = list(train_user_embeddings.keys())
train_tag_ids = list(train_tag_embeddings.keys())
train_user_matrix = np.array([train_user_embeddings[user] for user in train_user_ids])
train_tag_matrix = np.array([train_tag_embeddings[tag] for tag in train_tag_ids])
train_similarity_matrix = cosine_similarity(train_user_matrix, train_tag_matrix)
train_user_id_to_index = {user_id: idx for idx, user_id in enumerate(train_user_ids)}
train_tag_id_to_index = {tag_id: idx for idx, tag_id in enumerate(train_tag_ids)}


# 推荐函数调整为基于训练模型，并返回推荐标签及其分数
def recommend_tags_train(user_id, top_m=10):
    user_node = f"user_{user_id}"
    if user_node not in train_user_id_to_index:
        return [], []
    user_idx = train_user_id_to_index[user_node]
    scores = train_similarity_matrix[user_idx]
    top_indices = scores.argsort()[-top_m:][::-1]
    recommended_tags = [train_tag_ids[idx].replace('tag_', '') for idx in top_indices]
    recommended_scores = [scores[idx] for idx in top_indices]
    return recommended_tags, recommended_scores


# 构建测试用户-标签字典
test_user_tags = defaultdict(set)
for _, row in test_tags.iterrows():
    test_user_tags[row['userId']].add(row['tag'])

# 推荐并评估
precision_at_10_list = []
recall_at_10_list = []
f1_at_10_list = []
ndcg_at_10_list = []

for user_id in test_user_tags:
    recommended, scores = recommend_tags_train(user_id, top_m=10)
    relevant = test_user_tags[user_id]
    recommended_set = set(recommended)
    relevant_set = set(relevant)

    if not recommended:
        # 如果没有推荐结果，跳过评估
        continue

    # 计算 Precision@10
    intersection = recommended_set & relevant_set
    precision_at_10 = len(intersection) / len(recommended_set) if recommended_set else 0
    precision_at_10_list.append(precision_at_10)

    # 计算 Recall@10
    recall_at_10 = len(intersection) / len(relevant_set) if relevant_set else 0
    recall_at_10_list.append(recall_at_10)

    # 计算 F1-Score@10
    if precision_at_10 + recall_at_10 > 0:
        f1_at_10 = 2 * precision_at_10 * recall_at_10 / (precision_at_10 + recall_at_10)
    else:
        f1_at_10 = 0
    f1_at_10_list.append(f1_at_10)

    # 计算 NDCG@10
    # y_true 是相关性，y_score 是相似度分数
    relevance = [1 if tag in relevant_set else 0 for tag in recommended]
    ndcg = ndcg_score([relevance], [scores], k=10)
    ndcg_at_10_list.append(ndcg)

# 计算平均指标
avg_precision_at_10 = np.mean(precision_at_10_list) if precision_at_10_list else 0
avg_recall_at_10 = np.mean(recall_at_10_list) if recall_at_10_list else 0
avg_f1_at_10 = np.mean(f1_at_10_list) if f1_at_10_list else 0
avg_ndcg_at_10 = np.mean(ndcg_at_10_list) if ndcg_at_10_list else 0

print(f"Precision@10: {avg_precision_at_10:.4f}")
print(f"Recall@10: {avg_recall_at_10:.4f}")
print(f"F1-Score@10: {avg_f1_at_10:.4f}")
print(f"NDCG@10: {avg_ndcg_at_10:.4f}")

# 7. 评分预测（可选）
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 数据划分：训练集和测试集（80%训练，20%测试）
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

# 构建用户-电影评分矩阵
train_user_movie_matrix = train_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
test_user_movie_matrix = test_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# 获取共同用户和电影
common_users = train_user_movie_matrix.index.intersection(test_user_movie_matrix.index)
common_movies = train_user_movie_matrix.columns.intersection(test_user_movie_matrix.columns)

print(f"共同用户数量: {len(common_users)}")
print(f"共同电影数量: {len(common_movies)}")

train_matrix = train_user_movie_matrix.loc[common_users, common_movies]
test_matrix = test_user_movie_matrix.loc[common_users, common_movies]

# 构建电影-标签矩阵
movie_tags = filtered_tags[['movieId', 'tag']].drop_duplicates()
movie_tags['tag'] = movie_tags['tag'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()
movie_tags['tag'] = movie_tags['tag'].apply(lambda x: f"tag_{x}")

# 生成电影嵌入向量（平均标签嵌入）
movie_embeddings = {}
for movie_id in common_movies:
    tags_of_movie = movie_tags[movie_tags['movieId'] == movie_id]['tag']
    tags_of_movie = [tag for tag in tags_of_movie if f"tag_{tag}" in train_tag_embeddings]
    if tags_of_movie:
        movie_embeddings[movie_id] = np.mean([train_tag_embeddings[tag] for tag in tags_of_movie], axis=0)
    else:
        movie_embeddings[movie_id] = np.zeros(64)

# 检查电影嵌入是否正确
missing_movies = [movie_id for movie_id in common_movies if movie_id not in movie_embeddings]
if missing_movies:
    print(f"缺少的电影ID: {missing_movies}")

movie_matrix = np.array([movie_embeddings[movie_id] for movie_id in common_movies])

# 计算用户与电影的预测评分（点积）
predicted_ratings = cosine_similarity(train_user_matrix, movie_matrix) * 5  # 归一化到5分制

# 获取测试评分
test_ratings_list = []
predicted_ratings_list = []

for user_id in common_users:
    if user_id not in common_users:
        continue
    user_idx = train_user_movie_matrix.index.get_loc(user_id)
    for movie_id in common_movies:
        test_rating = test_matrix.loc[user_id, movie_id]
        if test_rating > 0:
            movie_idx = common_movies.get_loc(movie_id)
            if user_idx >= predicted_ratings.shape[0] or movie_idx >= predicted_ratings.shape[1]:
                # print(f"用户索引 {user_idx} 或电影索引 {movie_idx} 越界")
                continue
            predicted_rating = predicted_ratings[user_idx, movie_idx]
            # 限制预测评分在1-5之间
            predicted_rating = min(max(predicted_rating, 1), 5)
            test_ratings_list.append(test_rating)
            predicted_ratings_list.append(predicted_rating)

# 计算 RMSE 和 MAE
if test_ratings_list and predicted_ratings_list:
    rmse = np.sqrt(mean_squared_error(test_ratings_list, predicted_ratings_list))
    mae = mean_absolute_error(test_ratings_list, predicted_ratings_list)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
else:
    print("测试集评分列表为空，无法计算 RMSE 和 MAE。")
