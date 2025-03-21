import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Embedding, LSTM


# 加载数据
def load_files_from_dir(directory):
    texts = []
    labels = []
    for label in ["pos", "neg"]:
        label_dir = os.path.join(directory, label)
        for file_name in os.listdir(label_dir):
            with open(os.path.join(label_dir, file_name), "r", encoding="utf-8") as file:
                texts.append(file.read())
                labels.append(label)
    return texts, labels

def load_all_files():
    train_texts, train_labels = load_files_from_dir("./aclImdb/train")
    test_texts, test_labels = load_files_from_dir("./aclImdb/test")
    return train_texts, train_labels, test_texts, test_labels



# 特征提取
def get_features_by_wordbag(train_texts, test_texts):   # 词袋模型向量化器
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test

def get_features_by_wordbag_tfidf(train_texts, test_texts):   # 词袋模型+TF-IDF向量化器
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test


# 分类模型
def do_nb_wordbag(X_train, X_test, train_labels, test_labels):  # 朴素贝叶斯+词袋模型
    clf = MultinomialNB()
    clf.fit(X_train, train_labels)
    predictions = clf.predict(X_test)
    return accuracy_score(test_labels, predictions), confusion_matrix(test_labels, predictions)


def do_rf_doc2vec(X_train, X_test, train_labels, test_labels):   # 随机森林+Doc2Vec向量化器
    clf = RandomForestClassifier()
    clf.fit(X_train, train_labels)
    predictions = clf.predict(X_test)
    return accuracy_score(test_labels, predictions), confusion_matrix(test_labels, predictions)



def do_cnn_wordbag(X_train, X_test, train_labels, test_labels):  # CNN+词袋模型
    model = Sequential([
        Embedding(input_dim=X_train.shape[1], output_dim=128, input_length=X_train.shape[1]),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, np.array(train_labels), epochs=5, batch_size=32)
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)
    return accuracy_score(test_labels, predictions), confusion_matrix(test_labels, predictions)

def do_rnn_wordbag(X_train, X_test, train_labels, test_labels):  # RNN+词袋模型
    model = Sequential([
        Embedding(input_dim=X_train.shape[1], output_dim=128, input_length=X_train.shape[1]),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, np.array(train_labels), epochs=5, batch_size=32)
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)
    return accuracy_score(test_labels, predictions), confusion_matrix(test_labels, predictions)


# 主程序
if __name__ == "__main__":
    train_texts, train_labels, test_texts, test_labels = load_all_files()

    # 使用词袋模型特征
    X_train, X_test = get_features_by_wordbag(train_texts, test_texts)

    # 选择分类模型
    accuracy, cm = do_rf_doc2vec(X_train, X_test, train_labels, test_labels)

    print(f"准确率: {accuracy}")
    print(f"混淆矩阵:\n{cm}")
