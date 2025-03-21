import os
import re
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
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

# 创建嵌入矩阵
def create_embedding_matrix(word_index, word2vec_model, embedding_dim):
    vocab_size = len(word_index) + 1  # +1 因为索引从 1 开始，0 是用于填充的
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    return embedding_matrix

# 主程序
if __name__ == "__main__":
    # 加载数据
    train_texts, train_labels, test_texts, test_labels = load_all_files()

    # 数据预处理：文本向量化
    MAX_NUM_WORDS = 10000  # 词汇表大小
    MAX_SEQ_LEN = 100  # 每个句子的最大长度

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    X_train = tokenizer.texts_to_sequences(train_texts)
    X_test = tokenizer.texts_to_sequences(test_texts)

    # 填充序列到固定长度
    X_train = pad_sequences(X_train, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    X_test = pad_sequences(X_test, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # 加载预训练的 Word2Vec 模型
    word2vec_model = Word2Vec.load("./model/word2vec.model")
    EMBEDDING_DIM = word2vec_model.vector_size

    # 创建嵌入矩阵
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, word2vec_model, EMBEDDING_DIM)

    # 创建双层 LSTM 模型，使用 Word2Vec 嵌入
    model = Sequential([
        Embedding(
            input_dim=embedding_matrix.shape[0],  # 词汇表大小
            output_dim=EMBEDDING_DIM,  # 嵌入维度
            weights=[embedding_matrix],  # 使用预训练嵌入矩阵
            input_length=MAX_SEQ_LEN,  # 输入序列长度
            trainable=False  # 冻结嵌入层
        ),
        LSTM(128, return_sequences=True),  # 第一层 LSTM，返回序列
        LSTM(128, return_sequences=False),  # 第二层 LSTM，不返回序列
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")  # 二分类
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # 训练模型
    BATCH_SIZE = 32
    EPOCHS = 5

    print("开始训练双层 LSTM 模型...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )

    # 测试模型
    print("\n评估模型...")
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    print("\n分类报告：")
    print(classification_report(y_test, y_pred))

    print("\n混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))
