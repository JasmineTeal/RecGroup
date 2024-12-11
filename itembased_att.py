import time
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
from sklearn.feature_extraction.text import TfidfVectorizer


# 加载MovieLens 1M数据集
def load_data():
    ratings_path = 'dataset/ml-1m/ratings.dat'
    movies_path = 'dataset/ml-1m/movies.dat'
    users_path = 'dataset/ml-1m/users.dat'

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

    users = pd.read_csv(
        users_path,
        sep='::',
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
        engine='python',
        encoding='ISO-8859-1'
    )

    return ratings, movies, users


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


# 计算电影属性相似度
def calculate_movie_attribute_similarity(movies):
    # 提取类型
    genres = movies['Genres']

    # 使用 TF-IDF 计算类型相似度
    vectorizer = TfidfVectorizer()
    genre_vectors = vectorizer.fit_transform(genres)

    # 计算余弦相似度
    similarity = cosine_similarity(genre_vectors)
    similarity_matrix = pd.DataFrame(similarity, index=movies['MovieID'], columns=movies['MovieID'])
    return similarity_matrix


# 综合评分和属性相似度
def combine_similarity(rating_similarity, attribute_similarity, alpha=0.7):
    return alpha * rating_similarity + (1 - alpha) * attribute_similarity


# 计算综合的电影相似度
def calculate_item_similarity_with_attributes(movie_user_matrix, movies, alpha=0.7):
    # 基于评分的相似度
    print("Calculating rating-based similarity...")
    matrix_filled = movie_user_matrix.fillna(0)
    rating_similarity = cosine_similarity(matrix_filled)
    rating_similarity_matrix = pd.DataFrame(rating_similarity, index=movie_user_matrix.index,
                                            columns=movie_user_matrix.index)

    # 基于属性的相似度
    print("Calculating attribute-based similarity...")
    attribute_similarity_matrix = calculate_movie_attribute_similarity(movies)

    # 结合两种相似度
    print("Combining similarities...")
    combined_similarity_matrix = combine_similarity(rating_similarity_matrix, attribute_similarity_matrix, alpha=alpha)

    return combined_similarity_matrix

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


# 修改 evaluate_model 函数
def evaluate_model_with_attributes(train, test, movies, k=5, alpha=0.7):
    print('Start create matrix')
    movie_user_matrix = create_movie_user_matrix(train)
    print('Start calculating similarity')
    time1 = time.time()
    similarity_matrix = calculate_item_similarity_with_attributes(movie_user_matrix, movies, alpha=alpha)
    time2 = time.time()
    print('Finish calculating similarity')
    print(f'Cost time to calculate similarity: {time2 - time1}s')

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


# 修改 main 函数
def main():
    # 加载数据
    ratings, movies, users = load_data()

    # 划分数据集: 80%训练，20%测试
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    # 评估
    alpha = 0.9  # 评分相似度和属性相似度的权重
    rmse, mae = evaluate_model_with_attributes(train, test, movies, k=5, alpha=alpha)

    print("评估结果：")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print('Start sample recommendation')
    # 构建电影-用户评分矩阵和相似度矩阵
    movie_user_matrix = create_movie_user_matrix(train)
    similarity_matrix = calculate_item_similarity_with_attributes(movie_user_matrix, movies, alpha=alpha)

    # 输入目标用户ID
    user_id = 1  # 假设目标用户是ID为1的用户

    # 推荐电影
    recommendations = recommend_movies(user_id, movie_user_matrix, similarity_matrix, movies, k=5, top_n=10)

    print("推荐的电影列表：")
    print(recommendations)

    output_file = 'item-based results with attributes.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('Evaluation Metrics\n')
        f.write(f'RMSE: {rmse}\n')
        f.write(f'MAE: {mae}\n')
        f.write('Sample Recommendation\n')
        f.write(f'Sample UserID: {user_id}\n')
        f.write(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()