import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class DataProcessor:
    def __init__(self):
        pass

    def load_data(self, file_path='D:\\final paper\data\ASSISTments.csv', data_format='csv', dataset_name=None):
        """加载数据

        Args:
            file_path: 数据文件路径
            data_format: 数据格式，支持'csv', 'excel', 'txt'
            dataset_name: 数据集名称，支持'ASSISTments', 'EdNet', 'MOOPer'

        Returns:
            加载的数据
        """
        if data_format == 'csv':
            data = pd.read_csv(file_path)
        elif data_format == 'excel':
            data = pd.read_excel(file_path)
        elif data_format == 'txt':
            data = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f"Unsupported data format: {data_format}")

        # 根据数据集名称进行特定处理
        if dataset_name == 'ASSISTments':
            # ASSISTments 数据集处理
            if 'user_id' in data.columns:
                data = data.rename(columns={'user_id': 'student_id'})
            if 'problem_id' in data.columns:
                data = data.rename(columns={'problem_id': 'item_id'})
            if 'correct' in data.columns:
                data = data.rename(columns={'correct': 'score'})
        elif dataset_name == 'EdNet':
            # EdNet 数据集处理
            if 'uid' in data.columns:
                data = data.rename(columns={'uid': 'student_id'})
            if 'qid' in data.columns:
                data = data.rename(columns={'qid': 'item_id'})
            if 'correct' in data.columns:
                data = data.rename(columns={'correct': 'score'})
        elif dataset_name == 'MOOPer':
            # MOOPer 数据集处理
            if 'user_id' in data.columns:
                data = data.rename(columns={'user_id': 'student_id'})
            if 'item_id' in data.columns:
                pass  # 已经是正确的列名
            if 'score' in data.columns:
                pass  # 已经是正确的列名

        return data

    def preprocess(self, data, student_id_col='student_id', item_id_col='item_id', score_col='score',
                   timestamp_col=None, q_matrix=None):
        """预处理数据

        Args:
            data: 原始数据
            student_id_col: 学生ID列名
            item_id_col: 题目ID列名
            score_col: 得分列名
            timestamp_col: 时间戳列名（如果有的话）
            q_matrix: Q矩阵（如果有的话）

        Returns:
            processed_data: 处理后的数据
            q_matrix: Q矩阵（如果有的话）
        """
        # 确保数据类型正确
        data[student_id_col] = data[student_id_col].astype(int)
        data[item_id_col] = data[item_id_col].astype(int)
        data[score_col] = data[score_col].astype(int)

        # 按时间戳排序（如果提供了时间戳列）
        if timestamp_col is not None:
            data = data.sort_values(by=[student_id_col, timestamp_col])

        # 处理Q矩阵
        if q_matrix is None:
            # 尝试从数据中提取Q矩阵
            q_matrix = self._extract_q_matrix(data, item_id_col)

        return data, q_matrix

    def _extract_q_matrix(self, data, item_id_col):
        """从数据中提取Q矩阵

        Args:
            data: 原始数据
            item_id_col: 题目ID列名

        Returns:
            Q矩阵
        """
        # 查找可能的技能列
        skill_cols = [col for col in data.columns if re.search(r'skill|knowledge|concept', col.lower())]

        if skill_cols:
            # 提取Q矩阵
            q_matrix = data[[item_id_col] + skill_cols].drop_duplicates().sort_values(by=item_id_col)
            q_matrix = q_matrix[skill_cols].values
            return q_matrix
        else:
            # 如果没有找到技能列，返回None
            return None

    def validate_q_matrix(self, q_matrix):
        """校验Q矩阵

        Args:
            q_matrix: Q矩阵

        Returns:
            校验结果字典
        """
        validation_result = {
            'valid': True,
            'errors': []
        }

        # 检查Q矩阵是否为None
        if q_matrix is None:
            validation_result['valid'] = False
            validation_result['errors'].append('Q矩阵为None')
            return validation_result

        # 检查Q矩阵的维度
        if len(q_matrix.shape) != 2:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Q矩阵维度不正确，期望2维，实际{len(q_matrix.shape)}维')
            return validation_result

        # 检查Q矩阵中的值是否为0或1
        if not np.all(np.isin(q_matrix, [0, 1])):
            validation_result['valid'] = False
            validation_result['errors'].append('Q矩阵中包含非0或1的值')

        # 检查每个题目是否至少关联一个技能
        item_skill_count = np.sum(q_matrix, axis=1)
        zero_skill_items = np.where(item_skill_count == 0)[0]
        if len(zero_skill_items) > 0:
            validation_result['valid'] = False
            validation_result['errors'].append(f'题目{zero_skill_items}未关联任何技能')

        # 检查每个技能是否至少被一个题目关联
        skill_item_count = np.sum(q_matrix, axis=0)
        zero_item_skills = np.where(skill_item_count == 0)[0]
        if len(zero_item_skills) > 0:
            validation_result['errors'].append(f'技能{zero_item_skills}未被任何题目关联')

        return validation_result

    def build_q_matrix(self, data, item_id_col='item_id', method='text', skill_names=None):
        """自动构建Q矩阵

        Args:
            data: 原始数据
            item_id_col: 题目ID列名
            method: 构建方法，支持'text'（基于题目文本）、'difficulty'（基于题目难度）、'performance'（基于学生表现）
            skill_names: 技能名称列表（如果提供）

        Returns:
            构建的Q矩阵
        """
        # 获取所有题目ID
        item_ids = sorted(data[item_id_col].unique())
        num_items = len(item_ids)

        if method == 'text':
            # 基于题目文本构建Q矩阵
            # 假设数据中包含题目文本列
            text_cols = [col for col in data.columns if re.search(r'text|content|question', col.lower())]
            if not text_cols:
                raise ValueError('数据中没有找到题目文本列')

            # 提取题目文本
            item_texts = []
            for item_id in item_ids:
                item_data = data[data[item_id_col] == item_id]
                if not item_data.empty:
                    text = ' '.join([str(item_data[col].values[0]) for col in text_cols if col in item_data.columns])
                    item_texts.append(text)
                else:
                    item_texts.append('')

            # 如果没有提供技能名称，基于文本聚类自动生成
            if skill_names is None:
                # 使用TF-IDF向量化
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                text_vectors = vectorizer.fit_transform(item_texts).toarray()

                # 使用K-means聚类生成技能
                from sklearn.cluster import KMeans
                num_skills = min(10, num_items // 5)
                kmeans = KMeans(n_clusters=num_skills, random_state=42)
                clusters = kmeans.fit_predict(text_vectors)

                # 构建Q矩阵
                q_matrix = np.zeros((num_items, num_skills))
                for i, cluster in enumerate(clusters):
                    q_matrix[i, cluster] = 1
            else:
                # 基于提供的技能名称构建Q矩阵
                num_skills = len(skill_names)
                q_matrix = np.zeros((num_items, num_skills))

                # 简单的关键词匹配
                for i, text in enumerate(item_texts):
                    for j, skill in enumerate(skill_names):
                        if skill.lower() in text.lower():
                            q_matrix[i, j] = 1

        elif method == 'difficulty':
            # 基于题目难度构建Q矩阵
            # 计算题目难度
            item_difficulty = self.estimate_item_difficulty(data, item_id_col)

            # 将题目难度分为几个等级，每个等级对应一个技能
            difficulty_values = np.array([item_difficulty[item_id] for item_id in item_ids])
            num_skills = min(5, num_items // 10)
            q_matrix = np.zeros((num_items, num_skills))

            # 将难度值分为num_skills个区间
            bins = np.linspace(0, 1, num_skills + 1)
            for i, difficulty in enumerate(difficulty_values):
                skill_id = np.digitize(difficulty, bins) - 1
                skill_id = min(skill_id, num_skills - 1)
                q_matrix[i, skill_id] = 1

        elif method == 'performance':
            # 基于学生表现构建Q矩阵
            # 使用因子分析或主成分分析
            from sklearn.decomposition import FactorAnalysis

            # 构建学生-题目矩阵
            student_ids = sorted(data['student_id'].unique())
            num_students = len(student_ids)
            score_matrix = np.zeros((num_students, num_items))

            for i, student_id in enumerate(student_ids):
                student_data = data[data['student_id'] == student_id]
                for j, item_id in enumerate(item_ids):
                    item_data = student_data[student_data[item_id_col] == item_id]
                    if not item_data.empty:
                        score_matrix[i, j] = item_data['score'].values[0]

            # 使用因子分析
            num_skills = min(10, num_items // 5)
            fa = FactorAnalysis(n_components=num_skills, random_state=42)
            fa.fit(score_matrix)

            # 构建Q矩阵
            loadings = fa.components_.T
            q_matrix = np.zeros((num_items, num_skills))
            for i in range(num_items):
                # 选择负荷最大的几个技能
                top_skills = np.argsort(loadings[i])[-1:]
                q_matrix[i, top_skills] = 1

        else:
            raise ValueError(f"Unsupported method: {method}")

        # 确保每个题目至少关联一个技能
        item_skill_count = np.sum(q_matrix, axis=1)
        zero_skill_items = np.where(item_skill_count == 0)[0]
        for item_idx in zero_skill_items:
            # 为没有关联技能的题目随机关联一个技能
            q_matrix[item_idx, np.random.randint(q_matrix.shape[1])] = 1

        return q_matrix

    def construct_student_sequences(self, data, student_id_col='student_id', item_id_col='item_id', score_col='score',
                                    timestamp_col=None, max_sequence_length=100):
        """按时间戳构造学生序列

        Args:
            data: 预处理后的数据
            student_id_col: 学生ID列名
            item_id_col: 题目ID列名
            score_col: 得分列名
            timestamp_col: 时间戳列名（如果有的话）
            max_sequence_length: 最大序列长度

        Returns:
            学生序列列表
        """
        sequences = []

        # 按学生分组
        student_groups = data.groupby(student_id_col)

        for student_id, group in student_groups:
            # 按时间戳排序（如果提供了时间戳列）
            if timestamp_col is not None:
                group = group.sort_values(by=timestamp_col)

            # 提取题目ID和得分
            item_ids = group[item_id_col].values
            scores = group[score_col].values

            # 构造序列
            if len(item_ids) > max_sequence_length:
                # 如果序列长度超过最大长度，截断为多个序列
                for i in range(0, len(item_ids), max_sequence_length):
                    end_idx = min(i + max_sequence_length, len(item_ids))
                    sequence = {
                        'student_id': student_id,
                        'item_ids': item_ids[i:end_idx],
                        'scores': scores[i:end_idx]
                    }
                    sequences.append(sequence)
            else:
                # 否则，构造单个序列
                sequence = {
                    'student_id': student_id,
                    'item_ids': item_ids,
                    'scores': scores
                }
                sequences.append(sequence)

        return sequences

    def handle_missing_values(self, data, strategy='drop'):
        """处理缺失值

        Args:
            data: 原始数据
            strategy: 处理策略，支持'drop'（删除）或'fill'（填充）

        Returns:
            处理后的数据
        """
        if strategy == 'drop':
            # 删除包含缺失值的行
            data = data.dropna()
        elif strategy == 'fill':
            # 填充缺失值
            # 对于数值列，使用均值填充
            numeric_cols = data.select_dtypes(include=['number']).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
            # 对于分类列，使用众数填充
            categorical_cols = data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                data[col] = data[col].fillna(data[col].mode()[0])
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        return data

    def handle_outliers(self, data, columns, method='iqr'):
        """处理异常值

        Args:
            data: 原始数据
            columns: 需要处理异常值的列
            method: 处理方法，支持'iqr'（四分位距）或'z-score'（Z分数）

        Returns:
            处理后的数据
        """
        for col in columns:
            if method == 'iqr':
                # 使用四分位距方法
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            elif method == 'z-score':
                # 使用Z分数方法
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores <= 3]
            else:
                raise ValueError(f"Unsupported method: {method}")

        return data

    def vectorize_item_text(self, item_texts, max_features=1000):
        """题目文本向量化

        Args:
            item_texts: 题目文本列表
            max_features: 最大特征数

        Returns:
            向量化后的特征矩阵
        """
        vectorizer = TfidfVectorizer(max_features=max_features)
        return vectorizer.fit_transform(item_texts).toarray()

    def estimate_item_difficulty(self, data, item_id_col='item_id', score_col='score'):
        """估计题目难度

        Args:
            data: 原始数据
            item_id_col: 题目ID列名
            score_col: 得分列名

        Returns:
            题目难度字典
        """
        # 计算每个题目的平均得分（得分越高，难度越低）
        item_difficulty = data.groupby(item_id_col)[score_col].mean().to_dict()
        # 转换为难度值（0-1，值越大难度越高）
        for item_id in item_difficulty:
            item_difficulty[item_id] = 1 - item_difficulty[item_id]

        return item_difficulty

    def split_data(self, data, test_size=0.2, validation_size=0.1, random_state=42, split_by='student'):
        """分割数据为训练集、验证集和测试集

        Args:
            data: 原始数据
            test_size: 测试集比例
            validation_size: 验证集比例
            random_state: 随机种子
            split_by: 分割方式，支持'student'（按学生分割）或'entry'（按条目分割）

        Returns:
            train_data: 训练集
            validation_data: 验证集
            test_data: 测试集
        """
        from sklearn.model_selection import train_test_split

        if split_by == 'student':
            # 按学生分割
            student_ids = data['student_id'].unique()
            train_students, test_students = train_test_split(student_ids, test_size=test_size + validation_size,
                                                             random_state=random_state)
            validation_students, test_students = train_test_split(test_students,
                                                                  test_size=test_size / (test_size + validation_size),
                                                                  random_state=random_state)

            train_data = data[data['student_id'].isin(train_students)]
            validation_data = data[data['student_id'].isin(validation_students)]
            test_data = data[data['student_id'].isin(test_students)]
        elif split_by == 'entry':
            # 按条目分割
            train_data, temp_data = train_test_split(data, test_size=test_size + validation_size,
                                                     random_state=random_state)
            validation_data, test_data = train_test_split(temp_data,
                                                          test_size=test_size / (test_size + validation_size),
                                                          random_state=random_state)
        else:
            raise ValueError(f"Unsupported split_by: {split_by}")

        return train_data, validation_data, test_data

    def scale_features(self, features):
        """特征标准化

        Args:
            features: 特征矩阵

        Returns:
            标准化后的特征矩阵
        """
        scaler = StandardScaler()
        return scaler.fit_transform(features)

    def generate_statistics(self, data):
        """生成数据统计报告

        Args:
            data: 原始数据

        Returns:
            统计报告字典
        """
        statistics = {
            'num_students': data['student_id'].nunique(),
            'num_items': data['item_id'].nunique(),
            'num_responses': len(data),
            'average_score': data['score'].mean(),
            'score_distribution': data['score'].value_counts().to_dict(),
            'student_response_count': data.groupby('student_id').size().describe().to_dict(),
            'item_response_count': data.groupby('item_id').size().describe().to_dict(),
            'item_difficulty': self.estimate_item_difficulty(data)
        }

        return statistics