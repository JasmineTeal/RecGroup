import csv

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt


# 加载MovieLens 1M数据集
def load_data():
    ratings_path = 'dataset/ml-1m/ratings.dat'
    movies_path = 'dataset/ml-1m/movies.dat'

    ratings = pd.read_csv(
        ratings_path,
        sep='::',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        engine='python',
        encoding='ISO-8859-1'
    )

    movies = pd.read_csv(
        movies_path,
        sep='::',
        names=['MovieID', 'Title', 'Genres'],
        engine='python',
        encoding='ISO-8859-1'
    )

    return ratings, movies


# 划分训练集和测试集
def train_test_split(ratings, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(ratings))
    test_set_size = int(len(ratings) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train = ratings.iloc[train_indices].copy()
    test = ratings.iloc[test_indices].copy()
    return train, test


# 创建电影-用户评分矩阵（训练集）
def create_movie_user_matrix(ratings):
    movie_user_matrix = ratings.pivot(index='MovieID', columns='UserID', values='Rating')
    return movie_user_matrix


# 计算电影相似度（基于训练集）
def calculate_item_similarity(movie_user_matrix):
    # 缺失值填0
    matrix_filled = movie_user_matrix.fillna(0)
    similarity = cosine_similarity(matrix_filled)
    similarity_matrix = pd.DataFrame(similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)
    return similarity_matrix


# 基于物品的协同过滤评分预测函数
def predict_rating(user_id, movie_id, movie_user_matrix, similarity_matrix, k=5):
    # 如果目标电影或用户不在矩阵中，则返回 NaN
    if movie_id not in movie_user_matrix.index or user_id not in movie_user_matrix.columns:
        return np.nan

    # 获取与目标电影相似的其他电影
    item_similarities = similarity_matrix[movie_id].drop(movie_id, errors='ignore')
    item_similarities = item_similarities.sort_values(ascending=False)

    # 选择前k个最相似的电影
    top_k_items = item_similarities.iloc[:k]

    # 加权平均
    numer = 0.0
    denom = 0.0
    for similar_movie_id, sim in top_k_items.items():
        similar_movie_rating = movie_user_matrix.at[similar_movie_id, user_id] if (
                    similar_movie_id in movie_user_matrix.index and user_id in movie_user_matrix.columns) else np.nan
        if not np.isnan(similar_movie_rating):
            numer += sim * similar_movie_rating
            denom += sim

    if denom == 0:
        return np.nan
    return numer / denom


# 评估模型
def evaluate_model(train, test, k=5):
    print('Start create matrix')
    movie_user_matrix = create_movie_user_matrix(train)
    print('Start calculating similarity')
    similarity_matrix = calculate_item_similarity(movie_user_matrix)

    preds = []
    trues = []
    print('Start predicting')
    # 遍历测试集中的用户-电影对，进行预测
    for idx, row in test.iterrows():
        user_id = row['UserID']
        movie_id = row['MovieID']
        true_rating = row['Rating']

        pred_rating = predict_rating(user_id, movie_id, movie_user_matrix, similarity_matrix, k=k)

        if not np.isnan(pred_rating):
            preds.append(pred_rating)
            trues.append(true_rating)

    # 计算RMSE和MAE
    preds = np.array(preds)
    trues = np.array(trues)
    rmse = sqrt(np.mean((preds - trues) ** 2))
    mae = np.mean(np.abs(preds - trues))
    return rmse, mae


def recommend_movies(user_id, movie_user_matrix, similarity_matrix, movies, k=5, top_n=10):
    """
    为指定用户推荐电影
    """
    predicted_ratings = {}

    # 遍历所有电影
    for movie_id in movie_user_matrix.index:
        # 如果用户未评分该电影，则预测评分
        if np.isnan(movie_user_matrix.at[movie_id, user_id]):
            predicted_rating = predict_rating(user_id, movie_id, movie_user_matrix, similarity_matrix, k=k)
            if not np.isnan(predicted_rating):
                predicted_ratings[movie_id] = predicted_rating

    # 按评分排序，取前 top_n 个电影
    top_movies = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    recommended_movie_ids = [movie[0] for movie in top_movies]

    # 查找电影的标题和类型
    recommended_movies = movies[movies['MovieID'].isin(recommended_movie_ids)]

    return recommended_movies[['MovieID', 'Title', 'Genres']]


def main():
    # 加载数据
    ratings, movies = load_data()

    # 划分数据集: 80%训练，20%测试
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    # 评估
    rmse, mae = evaluate_model(train, test, k=5)

    print("评估结果：")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print('Start sample recommendation')
    # 构建电影-用户评分矩阵和相似度矩阵
    movie_user_matrix = create_movie_user_matrix(train)
    similarity_matrix = calculate_item_similarity(movie_user_matrix)

    # 输入目标用户ID
    user_id = 1  # 假设目标用户是ID为1的用户

    # 推荐电影
    recommendations = recommend_movies(user_id, movie_user_matrix, similarity_matrix, movies, k=5, top_n=10)

    print("推荐的电影列表：")
    print(recommendations)

    output_file = 'item-based results.csv'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('Evaluation Metrics\n')
        f.write(f'RMSE: {rmse}\n')
        f.write(f'MAE: {mae}\n')
        f.write('Sample Recomendation\n')
        f.write(f'Sample UserID: {user_id}\n')
        f.write(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
