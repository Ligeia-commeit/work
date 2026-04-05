import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.traditional.dina import DINA


class HybridCDM(nn.Module):
    def __init__(self, student_num, item_num, skill_num, hidden_dim=64, dropout=0.5):
        """初始化混合认知诊断模型

        Args:
            student_num: 学生数量
            item_num: 题目数量
            skill_num: 技能数量
            hidden_dim: 隐藏层维度
            dropout:  dropout率
        """
        super(HybridCDM, self).__init__()
        self.student_num = student_num
        self.item_num = item_num
        self.skill_num = skill_num
        self.hidden_dim = hidden_dim

        # 传统DINA模型组件
        self.dina = None

        # 深度学习组件
        self.student_embedding = nn.Embedding(student_num, hidden_dim)
        self.item_embedding = nn.Embedding(item_num, hidden_dim)
        self.skill_embedding = nn.Embedding(skill_num, hidden_dim)

        # 注意力机制
        self.attention = nn.Linear(hidden_dim, 1)

        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # 激活函数和dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, student_ids, item_ids, q_matrix):
        """前向传播

        Args:
            student_ids: 学生ID，形状为(batch_size,)
            item_ids: 题目ID，形状为(batch_size,)
            q_matrix: Q矩阵，形状为(item_num, skill_num)

        Returns:
            预测的答题概率，形状为(batch_size,)
        """
        # 获取嵌入
        student_embed = self.student_embedding(student_ids)
        item_embed = self.item_embedding(item_ids)

        # 计算题目相关的技能嵌入
        batch_size = student_ids.shape[0]
        skill_embeds = []

        for i in range(batch_size):
            item_id = item_ids[i]
            # 获取题目对应的技能
            skills = torch.where(q_matrix[item_id] == 1)[0]
            if len(skills) == 0:
                # 如果题目没有关联技能，使用零向量
                skill_embed = torch.zeros(self.hidden_dim, device=student_embed.device)
            else:
                # 获取技能嵌入
                skill_embed_list = [self.skill_embedding(skill) for skill in skills]
                skill_embed_tensor = torch.stack(skill_embed_list)

                # 计算注意力权重
                attn_input = skill_embed_tensor
                attn_scores = self.attention(attn_input)
                attn_weights = torch.softmax(attn_scores, dim=0)

                # 加权求和得到技能嵌入
                skill_embed = torch.sum(attn_weights * skill_embed_tensor, dim=0)

            skill_embeds.append(skill_embed)

        skill_embeds = torch.stack(skill_embeds)

        # 融合学生嵌入和技能嵌入
        combined = torch.cat([student_embed, skill_embeds], dim=1)

        # 前向传播
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x.squeeze()

    def fit(self, data, q_matrix, batch_size=32, epochs=30, lr=0.001):
        """训练模型

        Args:
            data: 训练数据，包含student_id, item_id, score
            q_matrix: Q矩阵
            batch_size: 批次大小
            epochs: 训练轮数
            lr: 学习率
        """
        # 首先使用DINA模型进行预训练（复用同一份交互数据格式）
        print("Pretraining with DINA model...")
        dina_model = DINA(q_matrix, max_iter=200)
        dina_model.fit(data, q_matrix)
        self.dina = dina_model

        # 转换为张量
        student_ids = torch.tensor(data[:, 0], dtype=torch.long)
        item_ids = torch.tensor(data[:, 1], dtype=torch.long)
        scores = torch.tensor(data[:, 2], dtype=torch.float32)
        q_matrix = torch.tensor(q_matrix, dtype=torch.float32)

        # 数据集和数据加载器
        dataset = torch.utils.data.TensorDataset(student_ids, item_ids, scores)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # 训练
        print("Training hybrid model...")
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (stu_ids, itm_ids, scrs) in enumerate(dataloader):
                # 前向传播
                outputs = self(stu_ids, itm_ids, q_matrix)
                loss = criterion(outputs, scrs)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

    def predict(self, student_ids, item_ids, q_matrix):
        """预测

        Args:
            student_ids: 学生ID
            item_ids: 题目ID
            q_matrix: Q矩阵

        Returns:
            预测的答题概率
        """
        self.eval()
        with torch.no_grad():
            student_ids = torch.tensor(student_ids, dtype=torch.long)
            item_ids = torch.tensor(item_ids, dtype=torch.long)
            q_matrix = torch.tensor(q_matrix, dtype=torch.float32)
            outputs = self(student_ids, item_ids, q_matrix)
        return outputs.numpy()

    def get_params(self):
        """获取模型参数

        Returns:
            模型参数
        """
        params = {}
        if self.dina is not None:
            params['dina_params'] = self.dina.get_params()
        return params