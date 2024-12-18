import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from math import sqrt


# 加载MovieLens 1M数据集
def load_data():
    # 假设文件已使用合适的编码加载
    ratings_path = '../dataset/ml-1m/ratings.dat'
    movies_path = '../dataset/ml-1m/movies.dat'
    users_path = '../dataset/ml-1m/users.dat'

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
    # 固定随机种子
    np.random.seed(random_state)
    # 打乱顺序
    shuffled_indices = np.random.permutation(len(ratings))
    test_set_size = int(len(ratings) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train = ratings.iloc[train_indices].copy()
    test = ratings.iloc[test_indices].copy()
    return train, test

def create_user_movie_matrix(ratings):
    user_movie_matrix = ratings.pivot(index='UserID', columns='MovieID', values='Rating')
    return user_movie_matrix

# 计算用户属性相似度
def calculate_user_attribute_similarity(users):
    # 提取用户属性
    attributes = users[['Gender', 'Age', 'Occupation']]

    # One-Hot 编码离散属性
    encoder = OneHotEncoder()
    encoded_attributes = encoder.fit_transform(attributes).toarray()

    # 计算余弦相似度
    similarity = cosine_similarity(encoded_attributes)
    similarity_matrix = pd.DataFrame(similarity, index=users['UserID'], columns=users['UserID'])
    return similarity_matrix


# 综合评分和属性相似度
def combine_similarity(rating_similarity, attribute_similarity, alpha=0.7):
    return alpha * rating_similarity + (1 - alpha) * attribute_similarity


# 计算用户相似度（结合属性）
def calculate_user_similarity_with_attributes(user_movie_matrix, users, alpha=0.7):
    # 基于评分的用户相似度
    print("Calculating rating-based similarity...")
    matrix_filled = user_movie_matrix.fillna(0)
    rating_similarity = cosine_similarity(matrix_filled)
    rating_similarity_matrix = pd.DataFrame(rating_similarity, index=user_movie_matrix.index,
                                            columns=user_movie_matrix.index)

    # 基于用户属性的相似度
    print("Calculating attribute-based similarity...")
    attribute_similarity_matrix = calculate_user_attribute_similarity(users)

    # 结合两种相似度
    print("Combining similarities...")
    combined_similarity_matrix = combine_similarity(rating_similarity_matrix, attribute_similarity_matrix, alpha=alpha)

    return combined_similarity_matrix

def predict_rating(user_id, movie_id, user_movie_matrix, similarity_matrix, k=5):
    # 如果目标用户或电影不在训练集的矩阵中，则返回平均分或0（简单fallback）
    if user_id not in user_movie_matrix.index or movie_id not in user_movie_matrix.columns:
        return np.nan

    # 获取用户的相似用户
    if user_id not in similarity_matrix.index:
        return np.nan
    user_similarities = similarity_matrix[user_id].drop(user_id, errors='ignore')
    user_similarities = user_similarities.sort_values(ascending=False)

    # 选择前k个最相似的用户
    top_k_users = user_similarities.iloc[:k]

    # 加权平均
    numer = 0.0
    denom = 0.0
    for neighbor_id, sim in top_k_users.items():
        neighbor_rating = user_movie_matrix.at[neighbor_id, movie_id] if (
                    neighbor_id in user_movie_matrix.index and movie_id in user_movie_matrix.columns) else np.nan
        if not np.isnan(neighbor_rating):
            numer += sim * neighbor_rating
            denom += sim

    if denom == 0:
        return np.nan
    return numer / denom

# 修改 evaluate_model 函数
def evaluate_model_with_attributes(train, test, users, k=5, alpha=0.7):
    print('Start create matrix')
    user_movie_matrix = create_user_movie_matrix(train)
    print('Start calculating similarity')
    time1 = time.time()
    similarity_matrix = calculate_user_similarity_with_attributes(user_movie_matrix, users, alpha=alpha)
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

        pred_rating = predict_rating(user_id, movie_id, user_movie_matrix, similarity_matrix, k=k)

        if not np.isnan(pred_rating):
            preds.append(pred_rating)
            trues.append(true_rating)

    # 计算RMSE和MAE
    preds = np.array(preds)
    trues = np.array(trues)
    rmse = sqrt(np.mean((preds - trues) ** 2))
    mae = np.mean(np.abs(preds - trues))
    return rmse, mae


def recommend_movies(user_id, user_movie_matrix, similarity_matrix, movies, k=5, top_n=10):
    """
    为指定用户推荐电影
    """
    # 存储预测评分
    predicted_ratings = {}

    # 遍历所有电影
    for movie_id in user_movie_matrix.columns:
        # 如果该用户未评分，则进行预测
        if np.isnan(user_movie_matrix.at[user_id, movie_id]):
            predicted_rating = predict_rating(user_id, movie_id, user_movie_matrix, similarity_matrix, k=k)
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
    rmse, mae = evaluate_model_with_attributes(train, test, users, k=5, alpha=alpha)

    print("评估结果：")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print('Start sample recommendation')
    # 构建用户-电影评分矩阵和相似度矩阵
    user_movie_matrix = create_user_movie_matrix(train)
    similarity_matrix = calculate_user_similarity_with_attributes(user_movie_matrix, users, alpha=alpha)

    # 输入目标用户ID
    user_id = 1  # 假设目标用户是ID为1的用户

    # 推荐电影
    recommendations = recommend_movies(user_id, user_movie_matrix, similarity_matrix, movies, k=5, top_n=10)

    print("推荐的电影列表：")
    print(recommendations)

    output_file = 'user-based results with attributes.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('Evaluation Metrics\n')
        f.write(f'RMSE: {rmse}\n')
        f.write(f'MAE: {mae}\n')
        f.write('Sample Recommendation\n')
        f.write(f'Sample UserID: {user_id}\n')
        f.write(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()