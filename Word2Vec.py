from gensim.models import Word2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import nltk
from nltk.corpus import stopwords
import re
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 下载并加载 NLTK 的停用词
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 清洗文本，去除数字和标点符号
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # 移除数字
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
    return text

# 加载数据
def load_files_from_dir(directory):
    texts = []
    labels = []
    for label in ["pos", "neg"]:
        label_dir = os.path.join(directory, label)
        for file_name in os.listdir(label_dir):
            with open(os.path.join(label_dir, file_name), "r", encoding="utf-8") as file:
                text = file.read()
                cleaned_text = clean_text(text)  # 清洗文本
                texts.append(cleaned_text)
                labels.append(label)
    return texts, labels

def load_all_files():
    train_texts, train_labels = load_files_from_dir("./aclImdb/train")
    test_texts, test_labels = load_files_from_dir("./aclImdb/test")
    return train_texts, train_labels, test_texts, test_labels

# 将句子转换为句子向量（词向量的均值），并去除停用词
def get_sentence_vectors(texts, word2vec_model, stop_words):
    vectors = []
    for text in texts:
        words = text.split()
        # 去除停用词
        words = [word for word in words if word not in stop_words]
        word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        if word_vectors:  # 如果句子中有词向量
            vectors.append(np.mean(word_vectors, axis=0))  # 计算词向量的均值
        else:  # 如果句子没有词向量，用零向量代替
            vectors.append(np.zeros(word2vec_model.vector_size))
    return np.array(vectors)


def do_rf(X_train, X_test, train_labels, test_labels):   # 分类模型：随机森林 + Word2Vec
    clf = RandomForestClassifier()
    clf.fit(X_train, train_labels)
    predictions = clf.predict(X_test)
    return accuracy_score(test_labels, predictions), confusion_matrix(test_labels, predictions)


def do_xgboost(x_train, x_test, y_train, y_test):  # 使用 XGBoost 分类
    # 将标签从字符串转换为整数
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    xgb_model = xgb.XGBClassifier(eval_metric="logloss")
    xgb_model.fit(x_train, y_train_encoded)
    y_pred = xgb_model.predict(x_test)

    print("\nXGBoost 分类器结果：")
    print(classification_report(y_test_encoded, y_pred))
    print(confusion_matrix(y_test_encoded, y_pred))

# 主程序
if __name__ == "__main__":
    train_texts, train_labels, test_texts, test_labels = load_all_files()

    # 训练 Word2Vec 模型
    all_texts = train_texts + test_texts
    sentences = [text.split() for text in all_texts]  # 将文本拆分成单词列表
    # 如果有训练好的模型，可以加载
    # word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_model = Word2Vec.load("./model/word2vec.model")  # 加载已经训练好的模型

    # 生成句子向量，去除停用词
    X_train = get_sentence_vectors(train_texts, word2vec_model, stop_words)
    X_test = get_sentence_vectors(test_texts, word2vec_model, stop_words)

   # 使用随机森林分类
    accuracy, cm = do_rf(X_train, X_test, train_labels, test_labels)
    print("随机森林分类器结果：")
    print(f"准确率: {accuracy}")
    print(f"混淆矩阵:\n{cm}")

    # 使用 XGBoost 分类
    do_xgboost(X_train, X_test, train_labels, test_labels)
