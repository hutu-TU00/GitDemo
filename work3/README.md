# classify运行结果
<img src="https://github.com/hutu-TU00/GitDemo/blob/main/work3/classify.png" width="800" alt="01 环境搭建的截图一">

# 邮件分类项目

## 核心功能
本项目实现了一个基于朴素贝叶斯的邮件分类系统，能够将邮件分为垃圾邮件(spam)和正常邮件(ham)两类。

## 算法基础
### 多项式朴素贝叶斯分类器
本项目采用多项式朴素贝叶斯(Multinomial Naive Bayes)分类器，其基于以下假设：
1. 特征条件独立性假设：所有特征在给定类别下是条件独立的
2. 多项式分布假设：特征表示词频计数

贝叶斯定理在邮件分类中的应用形式为：
P(Class|Document) ∝ P(Document|Class) * P(Class)

其中：
- P(Class)是类别的先验概率
- P(Document|Class) = ∏P(Word|Class) 基于独立性假设
- P(Class|Document)是给定文档属于某类的后验概率

## 数据处理流程
1. **分词处理**：使用中文分词工具(如jieba)将文本分割为词语序列
2. **停用词过滤**：移除常见无意义词(如"的"、"是"等)
3. **文本清洗**：移除标点符号、数字等非文本内容
4. **特征构建**：根据选择的方法构建特征向量

## 特征构建过程
### 高频词特征选择
1. 统计所有训练文本中词的出现频率
2. 选择出现频率最高的top_k个词作为特征
3. 每个文档表示为这些词的出现次数向量

数学表达：
X[i][j] = count(word_j in document_i)

### TF-IDF特征加权
1. 计算词频(TF): 词在文档中出现的频率
2. 计算逆文档频率(IDF): log(总文档数/包含该词的文档数)
3. TF-IDF = TF * IDF
4. 选择TF-IDF值最高的top_k个特征

数学表达：
X[i][j] = TF(word_j in document_i) * IDF(word_j)

### 差异对比
| 特征类型 | 优点 | 缺点 |
|----------|------|------|
| 高频词 | 计算简单，实现直观 | 忽略词的重要性差异 |
| TF-IDF | 能反映词的重要性，降低常见词权重 | 计算复杂度较高 |

## 特征模式切换方法
在代码中通过`feature_type`参数选择特征提取方式：
```python
# 使用高频词特征
features, vectorizer = get_features(texts, feature_type='frequency')

# 使用TF-IDF特征
features, vectorizer = get_features(texts, feature_type='tfidf')


