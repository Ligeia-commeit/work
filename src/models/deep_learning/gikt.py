import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GIKT(nn.Module):
    def __init__(self, num_items, num_skills, hidden_dim=128, dropout=0.5):
        """
        Graph-based Interaction Knowledge Tracing模型

        Args:
            num_items: 题目数量
            num_skills: 技能数量
            hidden_dim: 隐藏层维度
            dropout:  dropout概率
        """
        super(GIKT, self).__init__()
        self.num_items = num_items
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim

        # 输入嵌入层
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.interaction_embedding = nn.Embedding(2, hidden_dim)

        # 图神经网络层
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, num_items)

        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, item_seq, interaction_seq, adj_matrix=None):
        """
        前向传播

        Args:
            item_seq: 题目序列，形状为(batch_size, seq_length)
            interaction_seq: 作答结果序列，形状为(batch_size, seq_length)
            adj_matrix: 邻接矩阵，形状为(num_items, num_items)

        Returns:
            output: 预测结果，形状为(batch_size, seq_length, num_items)
        """
        batch_size, seq_length = item_seq.size()

        # 嵌入
        item_embedded = self.item_embedding(item_seq)
        interaction_embedded = self.interaction_embedding(interaction_seq)
        input_embedded = item_embedded + interaction_embedded

        # 图神经网络处理
        # 对每个时间步的输入应用GCN
        gcn_output = []
        for t in range(seq_length):
            # 获取当前时间步的输入
            current_input = input_embedded[:, t, :]

            # 图卷积操作
            x = self.relu(self.gcn1(current_input))
            x = self.gcn2(x)
            gcn_output.append(x.unsqueeze(1))

        # 合并GCN输出
        gcn_output = torch.cat(gcn_output, dim=1)

        # LSTM处理
        lstm_out, _ = self.lstm(gcn_output)

        # 预测
        output = self.sigmoid(self.output_layer(lstm_out))

        return output

    def fit(self, data, q_matrix=None, epochs=10, batch_size=32, learning_rate=0.001):
        """
        训练模型

        Args:
            data: 训练数据，numpy数组，每一行是学生-题目-分数三元组
            q_matrix: Q矩阵（可选）
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        # 准备训练数据：将numpy数组转换为学生序列
        sequences = []
        # 按学生ID分组
        student_ids = np.unique(data[:, 0])
        for student_id in student_ids:
            student_data = data[data[:, 0] == student_id]
            # 按题目ID排序（假设题目ID是按时间顺序分配的）
            student_data = student_data[student_data[:, 1].argsort()]
            item_ids = student_data[:, 1]
            scores = student_data[:, 2]
            if len(item_ids) > 1:  # 只保留长度大于1的序列
                sequences.append({
                    'student_id': student_id,
                    'item_ids': item_ids,
                    'scores': scores
                })

        if not sequences:
            print("No valid sequences found for training")
            return

        # 优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # 损失函数
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            total_loss = 0

            # 随机打乱序列
            np.random.shuffle(sequences)

            # 批处理
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i + batch_size]

                # 构建批次输入和目标
                max_seq_length = max(len(seq['item_ids']) for seq in batch_sequences)
                item_seq = torch.zeros((len(batch_sequences), max_seq_length - 1), dtype=torch.long)
                interaction_seq = torch.zeros((len(batch_sequences), max_seq_length - 1), dtype=torch.long)
                next_item_seq = torch.zeros((len(batch_sequences), max_seq_length - 1), dtype=torch.long)
                target_seq = torch.zeros((len(batch_sequences), max_seq_length - 1), dtype=torch.float)

                for j, seq in enumerate(batch_sequences):
                    seq_length = len(seq['item_ids'])
                    if seq_length > 1:
                        # 构建输入序列
                        item_seq[j, :seq_length - 1] = torch.tensor(seq['item_ids'][:-1], dtype=torch.long)
                        # 当前作答结果作为交互输入
                        interaction_seq[j, :seq_length - 1] = torch.tensor(seq['scores'][:-1], dtype=torch.long)
                        # 下一题ID，用于抽取对应题目的预测概率
                        next_item_seq[j, :seq_length - 1] = torch.tensor(seq['item_ids'][1:], dtype=torch.long)
                        # 目标序列：下一题的作答结果
                        target_seq[j, :seq_length - 1] = torch.tensor(seq['scores'][1:], dtype=torch.float)

                # 前向传播
                output = self.forward(item_seq, interaction_seq)

                pred_seq = output.gather(2, next_item_seq.unsqueeze(-1)).squeeze(-1)
                # 计算损失（形状一致：B*T）
                loss = criterion(pred_seq.view(-1), target_seq.view(-1))

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(sequences) * batch_size:.4f}")

    def predict(self, student_ids, item_ids, q_matrix=None):
        """
        预测学生作答结果

        Args:
            student_ids: 学生ID列表
            item_ids: 题目ID列表
            q_matrix: Q矩阵（可选）

        Returns:
            预测结果列表
        """
        # 这里简化处理，实际应用中需要根据学生的历史作答序列进行预测
        # 这里返回随机预测值作为示例
        return np.random.rand(len(student_ids))