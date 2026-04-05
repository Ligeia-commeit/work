import torch
import torch.nn as nn
import numpy as np


class NeuralCD(nn.Module):
    def __init__(self, num_items, num_skills, hidden_dim=128, dropout=0.5):
        """
        NeuralCD模型（兼顾可解释性的神经认知诊断模型）

        Args:
            num_items: 题目数量
            num_skills: 技能数量
            hidden_dim: 隐藏层维度
            dropout:  dropout概率
        """
        super(NeuralCD, self).__init__()
        self.num_items = num_items
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim

        # 学生能力嵌入
        self.student_embedding = nn.Embedding(10000, hidden_dim)  # 假设最多10000个学生

        # 题目嵌入
        self.item_embedding = nn.Embedding(num_items, hidden_dim)

        # 技能嵌入
        self.skill_embedding = nn.Embedding(num_skills, hidden_dim)

        # 注意力机制（用于技能权重）
        self.attention = nn.Linear(hidden_dim, 1)

        # 预测层
        self.predict_layer = nn.Linear(hidden_dim * 2, 1)

        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, student_ids, item_ids, q_matrix):
        """
        前向传播

        Args:
            student_ids: 学生ID列表，形状为(batch_size,)
            item_ids: 题目ID列表，形状为(batch_size,)
            q_matrix: Q矩阵，形状为(num_items, num_skills)

        Returns:
            output: 预测结果，形状为(batch_size,)
            skill_weights: 技能权重，形状为(batch_size, num_skills)
        """
        batch_size = student_ids.size(0)

        # 获取学生嵌入
        student_embedded = self.student_embedding(student_ids)

        # 获取题目嵌入
        item_embedded = self.item_embedding(item_ids)

        # Q 矩阵统一为 float 张量（兼容 numpy / list；evaluator 常传入 ndarray）
        if not isinstance(q_matrix, torch.Tensor):
            q_matrix = torch.as_tensor(q_matrix, device=item_ids.device, dtype=torch.float32)
        else:
            q_matrix = q_matrix.to(device=item_ids.device, dtype=torch.float32)
        item_q_matrix = q_matrix[item_ids.long()]

        # 计算技能嵌入
        skill_indices = torch.arange(self.num_skills, device=item_ids.device)
        skill_embedded = self.skill_embedding(skill_indices)

        # 计算技能权重
        skill_weights = []
        for i in range(batch_size):
            # 获取当前题目的技能关联（0/1 掩码）
            item_skills = item_q_matrix[i]
            mask = item_skills > 0.5
            relevant_skills = skill_embedded[mask]

            if relevant_skills.shape[0] > 0:
                # (K, H) -> (K, 1) -> (K,)；在技能维上做 softmax，K=1 时也为 1.0
                attention_scores = self.attention(relevant_skills).squeeze(-1)
                weights = torch.softmax(attention_scores, dim=0)
                weighted_skills = (relevant_skills * weights.unsqueeze(1)).sum(dim=0)
            else:
                # 如果没有关联技能，使用零向量
                weighted_skills = torch.zeros(self.hidden_dim, device=item_ids.device)

            skill_weights.append(weighted_skills)

        skill_weights = torch.stack(skill_weights)

        # 组合学生能力和技能权重
        combined = torch.cat([student_embedded, skill_weights], dim=1)

        # 预测
        output = self.sigmoid(self.predict_layer(combined)).squeeze(1)

        return output, skill_weights

    def fit(self, data, q_matrix=None, epochs=10, batch_size=32, learning_rate=0.001):
        """
        训练模型

        Args:
            data: 训练数据，包含学生ID、题目ID和得分
            q_matrix: Q矩阵
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        # 准备训练数据
        student_ids = torch.tensor(data[:, 0], dtype=torch.long)
        item_ids = torch.tensor(data[:, 1], dtype=torch.long)
        scores = torch.tensor(data[:, 2], dtype=torch.float)

        # 优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # 损失函数
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            total_loss = 0

            # 随机打乱数据
            indices = torch.randperm(len(student_ids))
            student_ids = student_ids[indices]
            item_ids = item_ids[indices]
            scores = scores[indices]

            # 批处理
            for i in range(0, len(student_ids), batch_size):
                batch_student_ids = student_ids[i:i + batch_size]
                batch_item_ids = item_ids[i:i + batch_size]
                batch_scores = scores[i:i + batch_size]

                # 前向传播
                output, _ = self.forward(batch_student_ids, batch_item_ids, q_matrix)

                # 计算损失
                loss = criterion(output, batch_scores)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(student_ids) * batch_size:.4f}")

    def predict(self, student_ids, item_ids, q_matrix=None):
        """
        预测学生作答结果

        Args:
            student_ids: 学生ID列表
            item_ids: 题目ID列表
            q_matrix: Q矩阵

        Returns:
            预测结果列表
        """
        # 转换为张量
        student_ids = torch.tensor(student_ids, dtype=torch.long)
        item_ids = torch.tensor(item_ids, dtype=torch.long)

        # 前向传播
        output, _ = self.forward(student_ids, item_ids, q_matrix)

        # 转换为numpy数组
        return output.detach().numpy()