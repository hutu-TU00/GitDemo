# Vocabulary类
 
在Vocabulary类中，mask_token对应的索引通过调用add_token方法赋值给self. mask_index 属性。
 
lookup_token方法中，如果self.unk_index >=0，则对未登录词返回 self.unk_index 。
 
调用add_many方法添加多个token时，实际是通过循环调用 add_token 方法实现的。

# CBOWVectorizer类
 
vectorize方法中，当vector_length < 0时，最终向量长度等于 输入文本 的长度。
 
from_dataframe方法构建词表时，会遍历DataFrame中 context 和 target 两列的内容。
 
out_vector[len(indices):]的部分填充为self.cbow_vocab. pad_index 。
 
# CBOWDataset类
 
_max_seq_length通过计算所有context列的 长度 的最大值得出。
 
set_split方法通过self._lookup_dict选择对应的 训练集 和 验证集 。
 
__getitem__返回的字典中，y_target通过查找 target 列的token得到。
 
# 模型结构
 
CBOWClassifier的forward中，x_embedded_sum的计算方式是embedding(x_in). sum (dim=1)。
 
模型输出层fc1的out_features等于 类别数量 参数的值。
 
# 训练流程
 
generate_batches函数通过PyTorch的 DataLoader 类实现批量加载。
 
训练时classifier.train()的作用是启用 训练 和 梯度计算 模式。
 
反向传播前必须执行 optimizer .zero_grad()清空梯度。
 
compute_accuracy中y_pred_indices通过 argmax 方法获取预测类别。
 
# 训练状态管理
 
make_train_state中early_stopping_best_val初始化为 正无穷大（float('inf') ） 。
 
update_train_state在连续 指定次数（patience值 ） 次验证损失未下降时会触发早停。
 
当验证损失下降时，early_stopping_step会被重置为 0 。
 
# 设备与随机种子
 
set_seed_everywhere中与CUDA相关的设置是 torch.cuda.manual_seed_all(seed)。
 
args.device的值根据 torch.cuda.is_available()确定。
 
# 推理与测试
 
get_closest函数中排除计算的目标词本身是通过continue判断word == 目标词 实现的。
 
测试集评估时一定要调用 model.eval() 方法禁用dropout。
 
# 关键参数
 
CBOWClassifier的padding_idx参数默认值为 0 。
 
args中控制词向量维度的参数是 embedding_dim 。
 
学习率调整策略ReduceLROnPlateau的触发条件是验证损失 增加 。