import os
import re
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 清洗文本
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # 去除数字
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    text = text.lower()  # 转为小写
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
                cleaned_text = clean_text(text)
                texts.append(cleaned_text)
                labels.append(1 if label == "pos" else 0)  # 正类为 1，负类为 0
    return texts, labels

def load_all_files():
    train_texts, train_labels = load_files_from_dir("./aclImdb/train")
    test_texts, test_labels = load_files_from_dir("./aclImdb/test")
    return train_texts, train_labels, test_texts, test_labels

# 将句子转换为固定长度的特征向量
def sentence_to_vector(sentence, model, embedding_dim):
    """将句子转化为向量（词向量平均法）"""
    vectors = [model.wv[word] for word in sentence.split() if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(embedding_dim)

# 主程序
if __name__ == "__main__":
    # 加载数据
    train_texts, train_labels, test_texts, test_labels = load_all_files()

    # 数据预处理：文本向量化
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts + test_texts)

    # 加载预训练的 Word2Vec 模型
    word2vec_model = Word2Vec.load("./model/word2vec.model")
    EMBEDDING_DIM = word2vec_model.vector_size

    # 将文本转化为特征向量
    X_train = np.array([sentence_to_vector(text, word2vec_model, EMBEDDING_DIM) for text in train_texts])
    X_test = np.array([sentence_to_vector(text, word2vec_model, EMBEDDING_DIM) for text in test_texts])
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 创建 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 设置参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'eta': 0.1,
    }

    # 训练模型，使用 early_stopping_rounds
    evals = [(dtrain, 'train'), (dval, 'eval')]
    print("开始训练 XGBoost 模型...")
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=10)

    # 在测试集上进行预测
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]  # 将概率转换为类别标签

    # 评估模型
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))

    print("\n混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))
