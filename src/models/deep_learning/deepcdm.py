import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DeepCDM(nn.Module):
    def __init__(self, student_num, item_num, skill_num, hidden_dim=64, dropout=0.5):
        """初始化DeepCDM模型
        
        Args:
            student_num: 学生数量
            item_num: 题目数量
            skill_num: 技能数量
            hidden_dim: 隐藏层维度
            dropout:  dropout率
        """
        super(DeepCDM, self).__init__()
        self.student_num = student_num
        self.item_num = item_num
        self.skill_num = skill_num
        self.hidden_dim = hidden_dim
        
        # 学生能力向量
        self.theta = nn.Embedding(student_num, skill_num)
        
        # 题目参数
        self.a = nn.Embedding(item_num, skill_num)
        self.b = nn.Embedding(item_num, 1)
        
        # 全连接层
        self.fc1 = nn.Linear(skill_num, hidden_dim)
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
        # 获取学生能力
        theta = self.theta(student_ids)
        
        # 获取题目参数
        a = self.a(item_ids)
        b = self.b(item_ids).squeeze()
        
        # 计算题目难度
        batch_size = student_ids.shape[0]
        difficulty = []
        for i in range(batch_size):
            item_id = item_ids[i]
            # 获取题目对应的技能
            skills = torch.where(q_matrix[item_id] == 1)[0]
            if len(skills) == 0:
                # 如果题目没有关联技能，使用默认难度
                diff = torch.tensor(0.5, device=theta.device)
            else:
                # 计算技能掌握程度
                skill_mastery = theta[i, skills]
                # 计算题目难度
                diff = torch.sum(a[i, skills] * (1 - skill_mastery)) + b[i]
            difficulty.append(diff)
        
        difficulty = torch.stack(difficulty)
        
        # 计算答对概率
        p = self.sigmoid(-difficulty)
        
        return p
    
    def fit(self, data, q_matrix, batch_size=32, epochs=30, lr=0.001):
        """训练模型
        
        Args:
            data: 训练数据，包含student_id, item_id, score
            q_matrix: Q矩阵
            batch_size: 批次大小
            epochs: 训练轮数
            lr: 学习率
        """
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
                print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")
    
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