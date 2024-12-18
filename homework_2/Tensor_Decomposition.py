import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, ndcg_score, mean_squared_error, mean_absolute_error
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.decomposition import NMF

# 文件路径（请根据实际路径调整）
movies_path = '../dataset/ml-latest-small/ml-latest-small/movies.csv'
ratings_path = '../dataset/ml-latest-small/ml-latest-small/ratings.csv'
tags_path = '../dataset/ml-latest-small/ml-latest-small/tags.csv'

# 加载数据
try:
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    tags = pd.read_csv(tags_path)
    print("数据加载成功。")
except FileNotFoundError as e:
    print(f"错误: {e}")
    print("请检查文件路径是否正确，并确保文件存在。")
    exit(1)

# 映射用户ID和电影ID到连续索引（基于评分和标签的并集）
all_user_ids = pd.concat([ratings['userId'], tags['userId']]).unique()
all_movie_ids = pd.concat([ratings['movieId'], tags['movieId']]).unique()
tag_list = tags['tag'].unique()

user_id_map = {id: idx for idx, id in enumerate(all_user_ids)}
movie_id_map = {id: idx for idx, id in enumerate(all_movie_ids)}
tag_id_map = {tag: idx for idx, tag in enumerate(tag_list)}

num_users = len(all_user_ids)  # 应为610
num_movies = len(all_movie_ids)  # 依据数据集大小调整
num_tags = len(tag_list)  # 1589

print(f"用户数量: {num_users}, 电影数量: {num_movies}, 标签数量: {num_tags}")

# 过滤低频标签（可选）
min_tag_count = 5
tag_counts = tags['tag'].value_counts()
filtered_tags = tag_counts[tag_counts >= min_tag_count].index
tags_filtered = tags[tags['tag'].isin(filtered_tags)]

# 更新标签列表和映射
tag_list = filtered_tags.unique()
tag_id_map = {tag: idx for idx, tag in enumerate(tag_list)}
num_tags = len(tag_list)

print(f"过滤后标签数量: {num_tags}")

# 构建标签张量
print("开始构建标签张量...")
train_tensor = np.zeros((num_users, num_movies, num_tags), dtype=np.float32)

for _, row in tags_filtered.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    tag = row['tag']

    if user_id in user_id_map and movie_id in movie_id_map and tag in tag_id_map:
        user_idx = user_id_map[user_id]
        movie_idx = movie_id_map[movie_id]
        tag_idx = tag_id_map[tag]
        train_tensor[user_idx, movie_idx, tag_idx] = 1

print("标签张量构建完成。")
print("标签张量形状:", train_tensor.shape)  # 应为(610, 9742, 168)

# 将 NumPy 张量转换为 Tensorly 张量
tensor = tl.tensor(train_tensor, dtype=tl.float32)

# 设置分解秩
rank = 20  # 可根据需要调整

# 进行 CP 分解
print("开始进行 CP 分解...")
factors = parafac(tensor, rank=rank, n_iter_max=100, tol=1e-6, init='random', random_state=0)
print("CP 分解完成。")

# 确认 factors.factors 包含3个因子矩阵
print(f"Number of factors returned: {len(factors.factors)}")
for idx, factor in enumerate(factors.factors):
    print(f"Factor {idx} shape: {factor.shape}")

# 解包因子矩阵
if len(factors.factors) == 3:
    user_factors, movie_factors, tag_factors = factors.factors
else:
    raise ValueError(f"Expected 3 factor matrices, but got {len(factors.factors)}")

# 预测标签张量
print("开始生成预测张量...")
predicted_tensor = tl.cp_to_tensor(factors)
print(f"predicted_tensor 类型: {type(predicted_tensor)}")  # 确认类型
predicted_tensor_np = predicted_tensor  # 移除了 .numpy()
predicted_binary = (predicted_tensor_np > 0.5).astype(int)
print("预测张量生成完成。")

# 构建测试集标签矩阵（与训练集相同）
print("开始构建测试集标签矩阵...")
test_tensor_np = train_tensor.copy()
print("测试集标签矩阵构建完成。")

# 初始化指标
precision_list = []
recall_list = []
f1_list = []
ndcg_list = []

print("开始计算评价指标...")

# 遍历每个用户-电影对进行评价
for user_idx in range(num_users):
    for movie_idx in range(num_movies):
        true_tags = test_tensor_np[user_idx, movie_idx, :]
        pred_tags = predicted_binary[user_idx, movie_idx, :]

        if np.sum(true_tags) == 0:
            continue  # 无标签，跳过

        # 计算 Precision, Recall, F1
        # 使用多标签分类的微平均指标
        precision, recall, f1, _ = precision_recall_fscore_support(true_tags, pred_tags, average='binary',
                                                                   zero_division=0)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # 计算 NDCG
        true_relevance = true_tags.reshape(1, -1)
        scores = predicted_tensor_np[user_idx, movie_idx].reshape(1, -1)
        ndcg = ndcg_score(true_relevance, scores)
        ndcg_list.append(ndcg)

# 计算平均指标
avg_precision = np.mean(precision_list) if precision_list else 0
avg_recall = np.mean(recall_list) if recall_list else 0
avg_f1 = np.mean(f1_list) if f1_list else 0
avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0

print(f"平均 Precision: {avg_precision:.4f}")
print(f"平均 Recall: {avg_recall:.4f}")
print(f"平均 F1-Score: {avg_f1:.4f}")
print(f"平均 NDCG: {avg_ndcg:.4f}")

# 评分预测

# 构建评分矩阵
print("开始构建评分矩阵...")
# 过滤仅包含已映射用户和电影的评分
filtered_ratings = ratings[ratings['userId'].isin(all_user_ids) & ratings['movieId'].isin(all_movie_ids)]
rating_matrix = np.zeros((num_users, num_movies), dtype=np.float32)

for _, row in filtered_ratings.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    rating = row['rating']

    if user_id in user_id_map and movie_id in movie_id_map:
        user_idx = user_id_map[user_id]
        movie_idx = movie_id_map[movie_id]
        rating_matrix[user_idx, movie_idx] = rating

print("评分矩阵构建完成。")

# 使用非负矩阵分解 (NMF) 作为评分预测模型
rank_nmf = 20  # 可根据需要调整
nmf_model = NMF(n_components=rank_nmf, init='random', random_state=0, max_iter=200)
print("开始进行 NMF 分解...")
user_features = nmf_model.fit_transform(rating_matrix)
movie_features = nmf_model.components_
print("NMF 分解完成。")

# 预测评分
print("开始预测评分...")
predicted_ratings = np.dot(user_features, movie_features)
print("评分预测完成。")

# 构建测试评分数据（与训练集相同）
print("开始构建测试评分数据...")
# 使用相同的filtered_ratings
y_true = filtered_ratings['rating'].values
y_pred = []

for _, row in filtered_ratings.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    if user_id in user_id_map and movie_id in movie_id_map:
        user_idx = user_id_map[user_id]
        movie_idx = movie_id_map[movie_id]
        y_pred.append(predicted_ratings[user_idx, movie_idx])
    else:
        y_pred.append(0)  # 或其他默认值

# 计算 RMSE 和 MAE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
