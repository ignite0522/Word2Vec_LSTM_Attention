# Word2Vec+LSTM+Attention恶意评论识别



接着上篇文章遗留的问题，学学word2vec，试着来解决一下

## 统计语言模型：N-gram模型

#### 简介

在word2vec之前，我先来聊聊N-gram模型

简单来说，统计语言模型就是用来计算句子概率的概率模型



这里提到句子🍊概率，那句子概率是啥呢？

举个简单的例子：

假设一个长度为m的句子，包含这些词：![(w_1,w_2,w_3,..,w_m)](assets/8LaFPhxHgewmcBr.png)，那么这个句子的概率（也就是这![m](assets/HC6lhfLuqPrazgF.png)个词共现的概率）是：

![img](assets/skwUOIXWf9RHT7F.png)



还不够简单？那再举个更实际的例子

N-gram模型的主要目的是捕获语言中的短语结构和上下文关系

比如

`I am going to school` 或 `I an going to school`

N-gram 模型计算两种句子的概率：

- P(I am going)>P(I an going)

计算的结果是前者的概率更大，即代表前一个句子更符合实际的语义环境



#### 一、二、三元模型

还有一个要注意的点是：经常提到是几元模型，这里的几元是啥意思？

当 n=1, 一个一元模型（unigram model)即为 ：（这是一个特殊情况，只考虑单个单词出现的概率）

![img](assets/uRbKyoYzVavtflL.png)

当 n=2, 一个二元模型（bigram model)即为 ：

![img](assets/2MdOFhGs7v6V1ty.png)

当 n=3, 一个三元模型（[trigram model](https://zhida.zhihu.com/search?content_id=5320991&content_type=Article&match_order=1&q=trigram+model&zhida_source=entity))即为

![img](assets/QzdrcMmWPTx2fp8.png)



接下来讲讲二元模型：

举个例子：假设我们有个语料库，我们对词语进行构建二元关系

<img src="assets/ajdPzpc5KlRTEs8.png" alt="img" style="zoom: 200%;" />

其中第一行，第二列 表示给定前一个词是 “i” 时，当前词为“want”的情况一共出现了827次



据此，我们便可以算得相应的频率分布表如下。

<img src="assets/e1iT5VxEMZq3YCU.png" alt="img" style="zoom:200%;" />

比如，前一个单词是i，那么下一个单词是want的概率为0.33，下一个单词是eat的概率是0.0036



看到这个是不是一下就想起了你在浏览起搜索时遇到的情况：

你在用谷歌时，输入一个或几个词，**搜索框通常会以下拉菜单的形式给出几个像下图一样的备选，这些备选其实是在猜想你想要搜索的那个词串。**

![img](assets/f5GAyrQTvWPCRp9.png)

这其实就是以N-Gram模型为基础来实现的



#### 局限

但这个模型具有很大的局限性：

首先它考虑当前词时，当前词只与距离它比较近的n个词更加相关(一般n不超过3)，而非前面所有的词都有关

其次，它没有考虑词与词之间内在的联系性，此话怎讲？

```
例如，考虑"the cat is walking in the bedroom"这句话
如果我们在训练语料中看到了很多类似“the dog is walking in the bedroom”或是“the cat is running in the bedroom”这样的句子，那么，哪怕我们此前没有见过这句话"the cat is walking in the bedroom"，也可以从“cat”和“dog”（“walking”和“running”）之间的相似性，推测出这句话的概率
但N-Gram做不到这点
```





## Word2Vec

先来聊聊为啥会出现Word2Vec

在这之前是存在传统的one-hot 编码，但传统的one-hot 编码仅仅只是将词符号化，不包含任何语义信息，还有个最大的痛点就是词的独热表示（one-hot representation）是高维的，且在高维向量中只有一个维度描述了词的语义 (高到什么程度呢？词典有多大就有多少维，一般至少上万的维度)，这是模型训练最不能忍受的，维度极高但有用信息又极其少，可以说是又长又臭

所以目前要解决的就是

1.赋予词语语义信息

2.降低维度



word2vec就横空出世了

**用word2vec训练出来的词向量矩阵，词与词之间是存在语义关系的，而且可以将词向量的纬度从几千几万直接降到几百**



#### 结构

word2vec包含三层：输入层、隐藏层、输出层，**通过从输入层到隐藏层或隐藏层到输出层的权重矩阵去向量化表示词的输入，学习迭代的是两个权重矩阵**，如下图：

![img](assets/GqkKrdnSTmQC8bR.png)





接下来，我们就一起看看word2vec的训练过程(下面以知乎网友 crystalajj 提供的 PPT 为例看一下 CBOW 模型训练流程)

示例句子：**I drink coffee everyday**



1.将上下文词进行 one-hot 表征作为输入：

```
I：        [1,0,0,0]
drink：     [0,1,0,0]
coffee：    ？
everyday： [0,0,0,1]
```

![img](assets/8kzhecL274snQSB.png)



2.然后将 one-hot 表征结果[1,0,0,0]、[**0,1,0,0**]、[0,0,0,1]，分别乘以：3×4的输入层到隐藏层的权重矩阵W「这个矩阵也叫嵌入矩阵，可以随机初始化生成」

![img](assets/nLbDq5oa9rRwBmX.png)

到这里可以看到，维度已经减下来了，这还不明显，如果词典中的单词为上万个，那一下压缩到几百个就明显了

**举例：比如上图中，若词典中有10000个单词，那么每个单词的独热编码就是10000\*1，这个维度为10000**

**第一个矩阵W的形状为200\*10000，第二个矩阵(单词)形状为10000\*1,相乘过后的矩阵形状为200\*1，维度直接断崖式下降**



3.将得到的结果向量求平均作为隐藏层向量：[1, 1.67, 0.33]

![img](assets/9gRKv6JWLz8qorp.png)





4.然后将隐藏层[1, 1.67, 0.33]向量乘以：4×3的隐藏层到输出层的权重矩阵![W'](assets/ovcauxJZAHq6SBr.png)「这个矩阵也是嵌入矩阵，也可以初始化得到」，得到输出向量：[4.01, 2.01, 5.00, 3.34]

![img](assets/uwkKHXL5U4oSzJP.png)



5.最后对输出向量[4.01, 2.01, 5.00, 3.34] 做 softmax 激活处理得到实际输出[0.23, 0.03, 0.62, 0.12]，并将其与真实标签[0, 0, 1, 0]做比较，然后基于损失函数做梯度优化训练

![img](assets/lJYgmxOXS6haptV.png)



这一系列图时真的太详细了，感谢作者

最后还给出了完整图

![img](assets/jPcfN9Gyh738HrD.png)



#### 效果展示

上述过程我们直观的看到了word2vec是怎么降低维度和赋予次之间关联性

那效果如何呢？为了方便展示，这里有一张将128维压缩成2维的图

![img](assets/Oo2vZpt1crBqs3N.png)

可以看到意思相近或者词性相同的词语之间的距离很近



如果图都不能满足你对word2vec的认知欲望的话，那我们就来跑跑代码看看！！

老规矩，先上代码

```py
import os
from gensim.models import Word2Vec
import re

# 清洗文本，去除数字和标点符号
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # 去除数字
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    return text

# 分词工具
def tokenize(text):
    text = clean_text(text)  # 在分词之前先清洗文本
    return text.split()  # 假设输入文本已经按空格分词。如果是中文，请使用 jieba 或其他分词工具。

# 加载数据
def load_files_from_dir(directory):
    sentences = []
    for label in ["pos", "neg"]:
        label_dir = os.path.join(directory, label)
        for file_name in os.listdir(label_dir):
            with open(os.path.join(label_dir, file_name), "r", encoding="utf-8") as file:
                sentences.append(tokenize(file.read()))  # 分词后存储为列表
    return sentences

def load_all_files():
    train_texts = load_files_from_dir("./aclImdb/train")
    test_texts = load_files_from_dir("./aclImdb/test")
    return train_texts, test_texts

# 训练 Word2Vec 模型
def train_word2vec(sentences, vector_size=100, window=5, min_count=2, workers=4):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

# 保存和加载模型
def save_model(model, path="word2vec.model"):
    model.save(path)
    print(f"模型已保存到 {path}")

def load_model(path="word2vec.model"):
    return Word2Vec.load(path)

# 主函数
if __name__ == "__main__":
    print("加载数据中...")
    train_texts, test_texts = load_all_files()
    print(f"训练数据：{len(train_texts)} 条，测试数据：{len(test_texts)} 条")

    print("训练 Word2Vec 模型中...")
    model = train_word2vec(train_texts, vector_size=100, window=5, min_count=2, workers=4)

    save_model(model, "./model/word2vec.model")

    print("加载保存的模型...")
    loaded_model = load_model("./model/word2vec.model")
    print(f"模型加载成功！词汇表大小：{len(loaded_model.wv)}")

```

本次使用的数据集来自互联网电影资料库（Internet Movie Database，IMDB），IMDB是一个关于电影演员、电影、电视节目、电视明星和电影制作的在线数据库。

训练和测试数据各25000条

我们先随便看一条正面评论

```
If you like adult comedy cartoons, like South Park, then this is nearly a similar format about the small adventures of three teenage girls at Bromwell High. Keisha, Natella and Latrina have given exploding sweets and behaved like bitches, I think Keisha is a good leader. There are also small stories going on with the teachers of the school. There's the idiotic principal, Mr. Bip, the nervous Maths teacher and many others. The cast is also fantastic, Lenny Henry's Gina Yashere, EastEnders Chrissie Watts, Tracy-Ann Oberman, Smack The Pony's Doon Mackichan, Dead Ringers' Mark Perry and Blunder's Nina Conti. I didn't know this came from Canada, but it is very good. Very good!
```



可以看到有标点符号，数字之类的

在训练之前先把数据清洗一下，去处文本中的数字和标点符号（因为它们对于语义无实际意义）

```py
# 清洗文本，去除数字和标点符号
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # 去除数字
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    return text
```



接着要将每个评论拆解成一个列表

```py
# 分词工具
def tokenize(text):
    text = clean_text(text)  # 在分词之前先清洗文本
    return text.split()  # 假设输入文本已经按空格分词。如果是中文，请使用 jieba 或其他分词工具。
```



```Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)```即：

vector_size：得到的向量维度为100

Windows：窗口大小，决定每个词上下文的范围（默认5）

min_count：忽略出现频率低于此值的词

workers：  并行线程数，用于加速训练



切记保存训练好的词向量，我训练的字典中的词少，没保存重新跑几分钟就可以啦，遇到超级大的词典训练的时间就是以天计算了



既然训练好了，那就来看看吧

```py
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
```



看看结果

![image-20250103160823792](assets/Zb32rMSnRJTQ5NE.png)



在相似度上，这里我们使用了good和great来比较，love和great来比较

good和great的相似度约为0.76，可以看到是比较相似了

love和great的相似度约为0.33，不相似



在找与love相似的单词时

可以看到其中hate的相似度最高，竟然是反义词恨最高！？

接着就是它的形容词loved，在接着就是enjoy



**这也再次验证了word2vec是可以赋予词与词之间的语义的**

**最后也是最重要的一点，每个词的维度从25000断崖式下降到100，这可以称得上是模型训练上的一大步啊，减少两很多无用且庞大的计算量**







## 恶意评论识别实战



### 一层LSTM

先上代码

```py
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

    # 数据预处理
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

    # 创建 LSTM 模型，使用 Word2Vec 嵌入
    model = Sequential([
        Embedding(
            input_dim=embedding_matrix.shape[0],  # 词汇表大小
            output_dim=EMBEDDING_DIM,  # 嵌入维度
            weights=[embedding_matrix],  # 使用预训练嵌入矩阵
            input_length=MAX_SEQ_LEN,  # 输入序列长度
            trainable=False  # 冻结嵌入层
        ),
        LSTM(128, return_sequences=False),
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

    print("开始训练 LSTM 模型...")
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
```





首先是数据预处理

将每个文本中的单词转换成对应的整数序号并裁剪每个文本使得所有文本的长度一致，以至于能输入神经网络

```py

    # 数据预处理
    MAX_NUM_WORDS = 10000  # 词汇表大小
    MAX_SEQ_LEN = 100  # 每个句子的最大长度

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    X_train = tokenizer.texts_to_sequences(train_texts)
    X_test = tokenizer.texts_to_sequences(test_texts)

    # 填充序列到固定长度
    X_train = pad_sequences(X_train, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    X_test = pad_sequences(X_test, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
```



接着就是在训练好的word2vec基础上得到嵌入矩阵

```
嵌入矩阵是一个二维矩阵，其中每一行对应一个词汇表中的单词的向量表示。
嵌入矩阵的维度是 V × d，其中：
V 是词汇表大小（即词汇表中单词的个数）。
d 是词嵌入的维度（即每个单词向量的长度）。
```

```py
def create_embedding_matrix(word_index, word2vec_model, embedding_dim):
    vocab_size = len(word_index) + 1  # +1 因为索引从 1 开始，0 是用于填充的
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    return embedding_matrix
```



现在每个文本中单词的维度都很高，在进行lstm训练之前，要将离散的单词映射到连续的向量空间中的过程。每个单词用一个固定大小的向量表示（这里就相当于将单词映射为提前训练好的word2vec词向量）

`Embedding` 层，用于将词汇表中的单词转化为对应的词嵌入（即词向量）

```py
Embedding(
    input_dim=embedding_matrix.shape[0],  # 词汇表大小
    output_dim=EMBEDDING_DIM,  # 嵌入维度
    weights=[embedding_matrix],  # 使用预训练嵌入矩阵
    input_length=MAX_SEQ_LEN,  # 输入序列长度
    trainable=False  # 冻结嵌入层,表示在训练过程中不会更新嵌入层的权重，而是使用预训练的词向量
)
```







模型核心

```py
    model = Sequential([
        Embedding(
            input_dim=embedding_matrix.shape[0],  # 词汇表大小
            output_dim=EMBEDDING_DIM,  # 嵌入维度
            weights=[embedding_matrix],  # 使用预训练嵌入矩阵
            input_length=MAX_SEQ_LEN,  # 输入序列长度
            trainable=False  # 冻结嵌入层
        ),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")  # 二分类
    ])
```

包含一个嵌入层，LSTM层，两个全连接层，两个Dropout 层



最后来看看效果吧！

![image-20250103171601975](assets/VeXfEjYGPWrB9LZ.png)

有76%的正确率





### 双层LSTM

```py
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
```



相较于一层LSTM，双层LSTM的代码没变化多少

```py
LSTM(128, return_sequences=True),  # 第一层 LSTM，返回序列
LSTM(128, return_sequences=False),  # 第二层 LSTM，不返回序列
```

解释一下

**`return_sequences=False`**:

- 当设置为 `False` 时，LSTM 层只会返回输入序列的**最后一个时间步**的输出。这样做通常适用于序列的最终分类或回归任务。（所以这里的第二层LSTM设置成False）

**`return_sequences=True`**:

- 当设置为 `True` 时，LSTM 层会返回**整个序列**的输出（即每个时间步的隐藏状态），这在需要进一步处理每个时间步的信息时非常有用。（所以这里的第一次LSTM设置成True，返回整个序列的输出）



看看效果

![image-20250103172548843](assets/4ILCflSpveNOgkh.png)

提升了5%





### 一层LSTM+Attention机制

```py
import os
import re
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
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



    # 自注意力层
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1),
                                     initializer="random_normal", trainable=True)
            self.b = self.add_weight(name="attention_bias", shape=(1,),
                                     initializer="zeros", trainable=True)
            super(AttentionLayer, self).build(input_shape)

        def call(self, inputs):
            scores = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
            attention_weights = tf.nn.softmax(scores, axis=1)
            context_vector = attention_weights * inputs
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector


    # 使用自定义 AttentionLayer
    model = Sequential([
        Embedding(
            input_dim=embedding_matrix.shape[0],  # 词汇表大小
            output_dim=EMBEDDING_DIM,  # 嵌入维度
            weights=[embedding_matrix],  # 使用预训练嵌入矩阵
            input_length=MAX_SEQ_LEN,  # 输入序列长度
            trainable=False  # 冻结嵌入层
        ),
        LSTM(128, return_sequences=True),  # 返回序列供 Attention 使用
        AttentionLayer(),  # 添加自注意力机制
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
    print("开始训练带有自注意力机制的 LSTM 模型...")
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
```

这个模型的核心在于Attention层

```py
    # 自注意力层
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1),
                                     initializer="random_normal", trainable=True)
            self.b = self.add_weight(name="attention_bias", shape=(1,),
                                     initializer="zeros", trainable=True)
            super(AttentionLayer, self).build(input_shape)

        def call(self, inputs):
            scores = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
            attention_weights = tf.nn.softmax(scores, axis=1)
            context_vector = attention_weights * inputs
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector
```

自注意力机制的核心思想是为每个输入序列中的时间步计算一个注意力权重，这些权重反映了其他时间步对于该时间步的重要性，自注意力机制在前一篇文章中有详细讲述，想了解可以看往期文章



看看效果

![截屏2025-01-03 17.30.04](assets/TzUeW6Bm5gh8JuE.png)

又提高1%



最后有试了一下双层LSTM+自注意力机制，效果变差了，不知道为啥

这是个很有意思的地方哈

有以下几种可能：

1.训练超参数设置不当，双层 LSTM 的模型更复杂，对学习率、批量大小等超参数更敏感

2.双层LSTM太复杂，过拟合了

3.Attention 层作用，Attention 层已经为模型提供了更强的表示能力，可能导致双层 LSTM 的额外复杂性成为累赘



那我们就一个一个的排查

看两个的训练过程

单层：

![截屏2025-01-03 17.42.31](assets/trmSo52cHQAnlfK.png)

双层：

![image-20250103174436165](assets/qI9rS6awXmYQR2k.png)



loss：对**训练集**的平均损失值

accuracy：对**训练集**的准确率

val_loss：对**验证集**的平均损失值

val_accuracy：对**验证集**的准确率



先看第一种情况：1.训练超参数设置不当，双层 LSTM 的模型更复杂，对学习率、批量大小等超参数更敏感

由图可以看到：不管是在训练集上还是在测试集上，平均损失值都是在稳步下降的，所以不是参数配置不当的原因



接着看第二种情况：2.双层LSTM太复杂，过拟合了

对比两张图可以看到双层LSTM模型在训练集上的表现是更好的，但在测试集上就表现得较差

所以很可能是双层LSTM太复杂，过拟合了



最后来看看第三种情况：3.Attention 层作用，Attention 层已经为模型提供了更强的表示能力，可能导致双层 LSTM 的额外复杂性成为累赘

问了一下chatgpt，它给出的方案是：尝试调整 Attention 层的位置，例如放在第一层 LSTM 后，那就试试呗

```py
    # 构建模型
    model = Sequential([
        Embedding(
            input_dim=embedding_matrix.shape[0],  
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],  
            input_length=MAX_SEQ_LEN, 
            trainable=False  
        ),
        LSTM(128, return_sequences=True),  # 第一层 LSTM，返回序列供 Attention 使用
        AttentionLayer(),  # Attention 层，提取第一层 LSTM 的重要特征
        tf.keras.layers.Reshape((1, 128)),  # 这样的变形操作适合将数据传入 LSTM 层
        LSTM(64, return_sequences=False),  # 第二层 LSTM，处理 Attention 的重要特征
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")  # 二分类
    ])
```

令我惊喜的事情发生啦

准确率竟然提高了

![image-20250103175801206](assets/h8JRVHsob9qWYct.png)



再看看训练过程：

![image-20250103180055914](assets/RBHNh1xCwzc9TFM.png)

对比之前的单层LSTM，效果明显变好



### 总结

确实是没想到还能再提升一点，起初只是抱着试试看的心态

最后说一下我的猜想：

在 `LSTM → LSTM → Attention` 结构中：

- 第一层 LSTM 的输出直接传递给第二层 LSTM。
- 第二层 LSTM 会进一步对时间序列特征进行抽象和压缩，但可能会丢失一些有用的局部信息。
- Attention 在最后一步才能作用于整个序列输出，无法挽回已经被 LSTM 层压缩或忽略的细节

在 `LSTM → Attention → LSTM` 结构中：

- 第一层 LSTM 的输出会通过 Attention 层提取重要的局部特征，并用加权方式聚焦于关键内容。
- 第二层 LSTM 只需处理这些已经被筛选和加权的关键信息，因此可以更有效地学习深层特征



## **再说简单一点就是`LSTM → Attention → LSTM` 通过早期引入 Attention 聚焦关键信息，减少了特征冗余和信息丢失问题，同时提高了计算效率**
