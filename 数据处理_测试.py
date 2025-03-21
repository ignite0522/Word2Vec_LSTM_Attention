from gensim.models import Word2Vec

# 加载模型
loaded_model = Word2Vec.load("./model/word2vec.model")

# 访问词汇表
vocabulary = loaded_model.wv.index_to_key
print("词汇表：", vocabulary[:10])  # 打印前 10 个单词


similarity = loaded_model.wv.similarity("good", "great")
print("\n相似度 (good vs great):", similarity)

similarity = loaded_model.wv.similarity("love", "great")
print("\n相似度 (love vs great):", similarity)

similar_words = loaded_model.wv.most_similar("love", topn=5)
print("\n与 'love' 最相似的单词：", similar_words)


# 访问单词的词向量
word = "love"
if word in loaded_model.wv:
    print(f"\n单词 '{word}' 存在于词汇表中")
    # 获取该单词的词向量
    vector = loaded_model.wv[word]
    print(f"\n'{word}' 的词向量：", vector)
else:
    print(f"\n单词 '{word}' 不在词汇表中")

# 查看词向量的维度
vector_size = loaded_model.wv.vector_size
print(f"\n词向量的维度是：{vector_size}")
