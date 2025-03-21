import os
from gensim.models import Word2Vec
import numpy as np


# 分词工具
def tokenize(text):
    return text.split()  # 假设文本已经按空格分词。如果是中文，需要换成 jieba 或其他分词工具。


# 加载数据
def load_files_from_dir(directory):
    texts = []
    labels = []
    for label in ["pos", "neg"]:
        label_dir = os.path.join(directory, label)
        for file_name in os.listdir(label_dir):
            with open(os.path.join(label_dir, file_name), "r", encoding="utf-8") as file:
                texts.append(tokenize(file.read()))  # 分词后保存
                labels.append(label)
    return texts, labels


def load_all_files():
    train_texts, train_labels = load_files_from_dir("./aclImdb/train")
    test_texts, test_labels = load_files_from_dir("./aclImdb/test")
    return train_texts, train_labels, test_texts, test_labels


# 训练 Word2Vec 模型
def train_word2vec(sentences, vector_size=100, window=5, min_count=2, workers=4):
    """
    训练 Word2Vec 模型
    :param sentences: 分词后的句子列表（List[List[str]]）
    :param vector_size: 向量维度
    :param window: 上下文窗口大小
    :param min_count: 忽略出现次数小于该值的单词
    :param workers: 线程数
    :return: 训练好的 Word2Vec 模型
    """
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model


# 将句子转换为特征向量
def get_features_by_word2vec(model, sentences):
    features = []
    for sentence in sentences:
        vectors = [model.wv[word] for word in sentence if word in model.wv]
        if vectors:
            features.append(np.mean(vectors, axis=0))  # 平均向量
        else:
            features.append(np.zeros(model.vector_size))  # 如果句子中没有词汇，则返回零向量
    return np.array(features)


# 主函数
if __name__ == "__main__":
    # 加载数据
    print("加载数据中...")
    train_texts, train_labels, test_texts, test_labels = load_all_files()
    print(f"训练集：{len(train_texts)} 条，测试集：{len(test_texts)} 条")

    # 训练 Word2Vec 模型
    print("训练 Word2Vec 模型...")
    model = train_word2vec(train_texts, vector_size=100, window=5, min_count=2, workers=4)

    # 将数据转换为特征向量
    print("提取特征向量...")
    X_train = get_features_by_word2vec(model, train_texts)
    X_test = get_features_by_word2vec(model, test_texts)

    # 输出特征维度
    print(f"训练集特征维度：{X_train.shape}")
    print(f"测试集特征维度：{X_test.shape}")

    # 保存模型
    model.save("word2vec.model")
    print("Word2Vec 模型已保存为 'word2vec.model'")
