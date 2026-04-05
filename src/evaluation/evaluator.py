import numpy as np
import time
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Evaluator:
    def __init__(self):
        pass

    def evaluate_prediction(self, y_true, y_pred):
        """评估预测性能

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            评估指标字典
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size == 0:
            return {
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "log_loss": float("nan"),
                "roc_auc": float("nan"),
            }
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 二值化预测值（阈值0.5）
        y_pred_binary = (y_pred >= 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred_binary, average='macro', zero_division=0),
            'log_loss': log_loss(y_true, y_pred, labels=[0, 1])
        }
        # AUC 在测试集只有单一类别时会报错；此时置为 NaN
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        except Exception:
            metrics['roc_auc'] = float('nan')
        return metrics

    def _split_interactions(self, data, test_ratio=0.2, rng=None):
        data = np.asarray(data)
        if rng is None:
            rng = np.random.default_rng(42)
        idx = rng.permutation(len(data))
        test_size = int(round(test_ratio * len(data)))
        test_idx = idx[:test_size]
        train_idx = idx[test_size:]
        return data[train_idx], data[test_idx]

    def evaluate_stability(self, model_builder, data, q_matrix, n_runs=10, base_seed=42):
        """评估模型稳定性

        Args:
            model_builder: 无参函数/可调用对象，返回一个“全新”的模型实例
            data: 数据集
            q_matrix: Q矩阵
            n_runs: 运行次数
            base_seed: 基础随机种子

        Returns:
            稳定性评估结果
        """
        metrics_list = []
        if n_runs <= 0:
            return {}

        for i in range(n_runs):
            rng = np.random.default_rng(base_seed + i)
            train_data, test_data = self._split_interactions(data, test_ratio=0.2, rng=rng)

            model = model_builder()
            if hasattr(model, 'fit'):
                model.fit(train_data, q_matrix)

            # 预测
            y_true = test_data[:, 2]
            y_pred = model.predict(test_data[:, 0], test_data[:, 1], q_matrix)

            # 评估
            metrics = self.evaluate_prediction(y_true, y_pred)
            metrics_list.append(metrics)

        if not metrics_list:
            return {}

        # 计算指标的标准差（跨运行）
        stability_metrics = {}
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list]
            stability_metrics[f'{metric}_std'] = float(np.std(values))

        return stability_metrics

    def evaluate_interpretability(self, model):
        """评估模型可解释性

        Args:
            model: 模型对象

        Returns:
            可解释性评估结果
        """
        interpretability_score = 0

        # 检查模型是否提供参数解释
        if hasattr(model, 'get_params'):
            params = model.get_params()
            if 'theta' in params:
                interpretability_score += 0.3
            if 'g' in params and 's' in params:
                interpretability_score += 0.3

        # 检查模型是否为传统模型（通常更可解释）
        model_name = model.__class__.__name__
        if model_name in ['DINA', 'DINO', 'NIDA']:
            interpretability_score += 0.4

        return {'interpretability_score': interpretability_score}

    def evaluate_computational_cost(self, model, train_data, test_data, q_matrix):
        """评估计算成本

        Args:
            model: 模型对象
            train_data: 训练集（交互三元组）
            test_data: 测试集（交互三元组）
            q_matrix: Q矩阵

        Returns:
            计算成本评估结果
        """
        # 该函数假设 evaluate() 已经完成训练；这里只测“预测耗时”
        # 为了能对比训练成本，evaluate() 会单独计时 fit()。
        training_time = float('nan')

        start_time = time.perf_counter()
        if hasattr(model, 'predict'):
            _ = model.predict(test_data[:, 0], test_data[:, 1], q_matrix)
        prediction_time = time.perf_counter() - start_time

        return {
            'training_time': training_time,
            'prediction_time': prediction_time
        }

    def evaluate_teaching_applicability(self, model):
        """评估教学适配性

        Args:
            model: 模型对象

        Returns:
            教学适配性评估结果
        """
        applicability_score = 0

        # 检查模型是否提供学生能力估计
        if hasattr(model, 'get_params'):
            params = model.get_params()
            if 'theta' in params:
                applicability_score += 0.5

        # 检查模型是否为传统模型（通常更适合教学应用）
        model_name = model.__class__.__name__
        if model_name in ['DINA', 'DINO', 'NIDA']:
            applicability_score += 0.3

        # 检查模型是否提供题目参数
        if hasattr(model, 'get_params'):
            params = model.get_params()
            if 'g' in params and 's' in params:
                applicability_score += 0.2

        return {'applicability_score': applicability_score}

    def evaluate(self, model_builder, data, q_matrix, seed=42, n_stability_runs=10):
        """综合评估模型

        Args:
            model_builder: 无参函数/可调用对象，返回一个“全新”的模型实例
            data: 数据集
            q_matrix: Q矩阵
            seed: 随机种子（用于单次 train/test 切分）
            n_stability_runs: 稳定性重复次数

        Returns:
            综合评估结果
        """
        data = np.asarray(data)
        if len(data) == 0:
            return {
                "prediction": self.evaluate_prediction(np.array([]), np.array([])),
                "stability": {},
                "interpretability": {"interpretability_score": 0.0},
                "computational_cost": {"training_time": float("nan"), "prediction_time": float("nan")},
                "teaching_applicability": {"applicability_score": 0.0},
            }

        rng = np.random.default_rng(seed)
        train_data, test_data = self._split_interactions(data, test_ratio=0.2, rng=rng)
        if len(test_data) == 0:
            return {
                "prediction": self.evaluate_prediction(np.array([]), np.array([])),
                "stability": self.evaluate_stability(
                    model_builder, data, q_matrix, n_runs=n_stability_runs, base_seed=seed
                ),
                "interpretability": {"interpretability_score": 0.0},
                "computational_cost": {"training_time": float("nan"), "prediction_time": float("nan")},
                "teaching_applicability": {"applicability_score": 0.0},
            }

        model = model_builder()

        # 训练模型
        training_time = float('nan')
        if hasattr(model, 'fit'):
            start_time = time.perf_counter()
            model.fit(train_data, q_matrix)
            training_time = time.perf_counter() - start_time

        # 预测
        y_true = test_data[:, 2]
        start_time = time.perf_counter()
        y_pred = model.predict(test_data[:, 0], test_data[:, 1], q_matrix)
        prediction_time = time.perf_counter() - start_time

        # 评估各项指标
        prediction_metrics = self.evaluate_prediction(y_true, y_pred)
        stability_metrics = self.evaluate_stability(model_builder, data, q_matrix, n_runs=n_stability_runs,
                                                    base_seed=seed)
        interpretability_metrics = self.evaluate_interpretability(model)
        computational_metrics = self.evaluate_computational_cost(model, train_data, test_data, q_matrix)
        computational_metrics['training_time'] = training_time
        computational_metrics['prediction_time'] = prediction_time
        teaching_metrics = self.evaluate_teaching_applicability(model)

        # 综合评估结果
        evaluation_results = {
            'prediction': prediction_metrics,
            'stability': stability_metrics,
            'interpretability': interpretability_metrics,
            'computational_cost': computational_metrics,
            'teaching_applicability': teaching_metrics
        }

        return evaluation_results