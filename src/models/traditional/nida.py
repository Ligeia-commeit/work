import numpy as np
from scipy.optimize import minimize

class NIDA:
    def __init__(self, q_matrix, max_iter=1000, tol=1e-6):
        """初始化NIDA模型
        
        Args:
            q_matrix: Q矩阵，形状为(item_num, skill_num)
            max_iter: 最大迭代次数
            tol: 收敛阈值
        """
        self.q_matrix = q_matrix
        self.item_num, self.skill_num = q_matrix.shape
        self.student_num = None
        self.max_iter = max_iter
        self.tol = tol
        self.g = None  # 猜测参数
        self.s = None  # 失误参数
        self.theta = None  # 学生能力参数
        self.w = None  # 技能权重参数
    
    def _get_required_skills(self, item_idx):
        """获取题目所需的技能
        
        Args:
            item_idx: 题目索引
            
        Returns:
            所需技能的索引列表
        """
        return np.where(self.q_matrix[item_idx] == 1)[0]
    
    def _calculate_p(self, theta):
        """计算学生答对题目的概率
        
        Args:
            theta: 学生能力向量
            
        Returns:
            答对题目的概率矩阵，形状为(student_num, item_num)
        """
        p = np.zeros((self.student_num, self.item_num))
        for i in range(self.item_num):
            required_skills = self._get_required_skills(i)
            if len(required_skills) == 0:
                # 如果题目不需要任何技能，默认答对概率为0.5
                p[:, i] = 0.5
            else:
                # 计算技能掌握程度的加权和
                skill_mastery = np.sum(theta[:, required_skills] * self.w[i, required_skills], axis=1)
                # 归一化
                total_weight = np.sum(self.w[i, required_skills])
                if total_weight > 0:
                    skill_mastery = skill_mastery / total_weight
                # 计算答对概率
                p[:, i] = (1 - self.s[i]) * skill_mastery + self.g[i] * (1 - skill_mastery)
        return p
    
    def _log_likelihood(self, params, data):
        """计算对数似然函数
        
        Args:
            params: 模型参数，包括g、s、w和theta
            data: 观测数据矩阵，形状为(student_num, item_num)
            
        Returns:
            负对数似然值
        """
        g = params[:self.item_num]
        s = params[self.item_num:2*self.item_num]
        w_flat = params[2*self.item_num:2*self.item_num + self.item_num*self.skill_num]
        w = w_flat.reshape(self.item_num, self.skill_num)
        theta_flat = params[2*self.item_num + self.item_num*self.skill_num:]
        theta = theta_flat.reshape(self.student_num, self.skill_num)
        
        # 计算答对概率
        p = np.zeros((self.student_num, self.item_num))
        for i in range(self.item_num):
            required_skills = self._get_required_skills(i)
            if len(required_skills) == 0:
                p[:, i] = 0.5
            else:
                skill_mastery = np.sum(theta[:, required_skills] * w[i, required_skills], axis=1)
                total_weight = np.sum(w[i, required_skills])
                if total_weight > 0:
                    skill_mastery = skill_mastery / total_weight
                p[:, i] = (1 - s[i]) * skill_mastery + g[i] * (1 - skill_mastery)
        
        # 计算对数似然
        log_likelihood = np.sum(data * np.log(p) + (1 - data) * np.log(1 - p))
        return -log_likelihood
    
    def fit(self, data, q_matrix=None):
        """训练NIDA模型
        
        Args:
            data: 观测数据，形状为(num_samples, 3)，包含(student_id, item_id, score)
            q_matrix: Q矩阵（如果需要）
        """
        # 转换数据格式为学生-题目矩阵
        student_ids = np.unique(data[:, 0])
        item_ids = np.unique(data[:, 1])
        self.student_num = len(student_ids)
        
        # 创建学生-题目矩阵
        score_matrix = np.zeros((self.student_num, self.item_num))
        for row in data:
            stu_idx = np.where(student_ids == row[0])[0][0]
            itm_idx = np.where(item_ids == row[1])[0][0]
            score_matrix[stu_idx, itm_idx] = row[2]
        
        # 初始化参数
        g_init = np.random.uniform(0.1, 0.3, self.item_num)
        s_init = np.random.uniform(0.1, 0.3, self.item_num)
        w_init = np.random.uniform(0.1, 1.0, (self.item_num, self.skill_num))
        # 只保留题目所需技能的权重
        for i in range(self.item_num):
            required_skills = self._get_required_skills(i)
            w_init[i, np.setdiff1d(np.arange(self.skill_num), required_skills)] = 0
        theta_init = np.random.randint(0, 2, size=(self.student_num, self.skill_num))
        
        # 合并参数
        params_init = np.concatenate([g_init, s_init, w_init.flatten(), theta_init.flatten()])
        
        # 优化
        bounds = [(0.01, 0.99)] * self.item_num + [(0.01, 0.99)] * self.item_num + [(0, 1)] * (self.item_num * self.skill_num) + [(0, 1)] * (self.student_num * self.skill_num)
        result = minimize(self._log_likelihood, params_init, args=(score_matrix,), bounds=bounds, method='L-BFGS-B', options={'maxiter': self.max_iter, 'gtol': self.tol})
        
        # 提取参数
        params = result.x
        self.g = params[:self.item_num]
        self.s = params[self.item_num:2*self.item_num]
        w_flat = params[2*self.item_num:2*self.item_num + self.item_num*self.skill_num]
        self.w = w_flat.reshape(self.item_num, self.skill_num)
        self.theta = params[2*self.item_num + self.item_num*self.skill_num:].reshape(self.student_num, self.skill_num)
    
    def predict(self, student_ids, item_ids, q_matrix=None):
        """预测学生的答题表现
        
        Args:
            student_ids: 学生ID列表
            item_ids: 题目ID列表
            q_matrix: Q矩阵（如果需要）
            
        Returns:
            预测的答题表现列表
        """
        predictions = []
        for stu_id, itm_id in zip(student_ids, item_ids):
            # 计算学生的能力向量
            # 这里简化处理，假设学生ID直接对应theta的索引
            if stu_id < self.student_num:
                theta = self.theta[stu_id:stu_id+1]
                # 计算该学生答对该题的概率
                p = self._calculate_p(theta)[0, itm_id]
                predictions.append(p)
            else:
                # 如果学生ID超出范围，返回默认概率
                predictions.append(0.5)
        return np.array(predictions)
    
    def get_params(self):
        """获取模型参数
        
        Returns:
            模型参数，包括g、s、w和theta
        """
        return {
            'g': self.g,
            's': self.s,
            'w': self.w,
            'theta': self.theta
        }