import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models import LdaModel
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import webbrowser
import os

# 1. 加载数据
data_path = '../dataset/ml-20m/'  # 请替换为实际路径
movies = pd.read_csv(data_path + 'movies.csv')
tags = pd.read_csv(data_path + 'tags.csv')


# 2. 检查并清理tags数据
# 查看tag列的数据类型和缺失值
print("原始tag列数据类型：", tags['tag'].dtype)
print("原始缺失值数量：", tags['tag'].isnull().sum())

# 删除tag列中的缺失值
tags = tags.dropna(subset=['tag'])

# 确保tag列的所有值都是字符串
tags['tag'] = tags['tag'].astype(str)

# 可选：删除空字符串的标签
tags = tags[tags['tag'].str.strip() != '']

# 查看清理后的数据
print("\n清理后的Tags数据集样本：")
print(tags.head())

# 2. 合并标签
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movie_tags = pd.merge(movie_tags, movies[['movieId', 'title', 'genres']], on='movieId')

# 3. 预处理（仅分词，不移除停用词）
def preprocess(text):
    return simple_preprocess(text, deacc=True)

movie_tags['processed_tags'] = movie_tags['tag'].apply(preprocess)

# 4. 创建词典和语料库
dictionary = corpora.Dictionary(movie_tags['processed_tags'])
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(text) for text in movie_tags['processed_tags']]

# 5. 训练LDA模型
num_topics = 10
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=num_topics,
                     random_state=100,
                     update_every=1,
                     chunksize=100,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)

# 6. 展示主题
for idx, topic in lda_model.print_topics(-1):
    print(f"主题 {idx+1}:")
    print(topic)
    print("\n")

# 8. 可视化
# 准备可视化数据
lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)

# 保存为HTML文件
output_file = 'lda_visualization.html'
pyLDAvis.save_html(lda_vis, output_file)
print(f"LDA可视化结果已保存为 {output_file}")

# 自动在默认浏览器中打开HTML文件
file_path = os.path.abspath(output_file)
webbrowser.open(f'file://{file_path}')
