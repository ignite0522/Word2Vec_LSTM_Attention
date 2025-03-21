import os
from gensim.models import Word2Vec
import re

# 清洗文本，去除数字和标点符号
def clean_text(text):
    """
    清洗文本，去除数字和标点符号
    :param text: 输入文本
    :return: 清洗后的文本
    """
    text = re.sub(r'\d+', '', text)  # 去除数字
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    return text

# 分词工具
def tokenize(text):
    """
    简单分词函数
    :param text: 输入文本
    :return: 分词后的单词列表
    """
    text = clean_text(text)  # 在分词之前先清洗文本
    return text.split()  # 假设输入文本已经按空格分词。如果是中文，请使用 jieba 或其他分词工具。

# 加载数据
def load_files_from_dir(directory):
    """
    从目录加载数据
    :param directory: 数据目录路径
    :return: 分词后的句子列表
    """
    sentences = []
    for label in ["pos", "neg"]:
        label_dir = os.path.join(directory, label)
        for file_name in os.listdir(label_dir):
            with open(os.path.join(label_dir, file_name), "r", encoding="utf-8") as file:
                sentences.append(tokenize(file.read()))  # 分词后存储为列表
    return sentences

def load_all_files():
    """
    加载训练和测试数据
    :return: 训练文本和测试文本
    """
    train_texts = load_files_from_dir("./aclImdb/train")
    test_texts = load_files_from_dir("./aclImdb/test")
    return train_texts, test_texts

# 训练 Word2Vec 模型
def train_word2vec(sentences, vector_size=100, window=5, min_count=2, workers=4):
    """
    训练 Word2Vec 模型
    :param sentences: 分词后的句子列表（List[List[str]]）
    :param vector_size: 词向量维度
    :param window: 上下文窗口大小
    :param min_count: 忽略出现次数小于该值的单词
    :param workers: 使用的线程数
    :return: 训练好的 Word2Vec 模型
    """
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

# 保存和加载模型
def save_model(model, path="word2vec.model"):
    """
    保存 Word2Vec 模型
    :param model: 训练好的模型
    :param path: 保存路径
    """
    model.save(path)
    print(f"模型已保存到 {path}")

def load_model(path="word2vec.model"):
    """
    加载 Word2Vec 模型
    :param path: 模型路径
    :return: 加载好的模型
    """
    return Word2Vec.load(path)

# 主函数
if __name__ == "__main__":
    # 加载数据
    print("加载数据中...")
    train_texts, test_texts = load_all_files()
    print(f"训练数据：{len(train_texts)} 条，测试数据：{len(test_texts)} 条")

    # 训练 Word2Vec 模型
    print("训练 Word2Vec 模型中...")
    model = train_word2vec(train_texts, vector_size=100, window=5, min_count=2, workers=4)

    # 保存模型
    save_model(model, "./model/word2vec.model")

    # 加载模型
    print("加载保存的模型...")
    loaded_model = load_model("./model/word2vec.model")
    print(f"模型加载成功！词汇表大小：{len(loaded_model.wv)}")
