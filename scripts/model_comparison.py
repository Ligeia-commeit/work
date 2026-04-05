"""
认知诊断模型比较框架
用于在PyCharm上运行的优化版本
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from typing import List, Tuple, Callable, Any
import warnings
import argparse

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"

# Windows 控制台 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 忽略不必要的警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

print("=" * 60)
print("认知诊断模型比较框架")
print("=" * 60)
print(f"Python version: {sys.version.split()[0]}")
print(f"Project root: {_PROJECT_ROOT}")
print(f"Working directory: {os.getcwd()}")
print()

# 创建实验目录
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"[OK] Experiments directory ready: {EXPERIMENTS_DIR}")

# 设置随机种子以确保可复现性
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
print(f"[OK] Random seed set to {RANDOM_SEED}")

print("\n" + "=" * 60)
print("导入模块")
print("=" * 60)

# 导入传统模型
print("\n[1] 导入传统认知诊断模型...")
try:
    from src.preprocessing.data_processor import DataProcessor

    print("    ✓ DataProcessor")
except Exception as e:
    print(f"    ✗ DataProcessor: {e}")
    DataProcessor = None

try:
    from src.models.traditional.dina import DINA

    print("    ✓ DINA")
except Exception as e:
    print(f"    ✗ DINA: {e}")
    DINA = None

try:
    from src.models.traditional.dino import DINO

    print("    ✓ DINO")
except Exception as e:
    print(f"    ✗ DINO: {e}")
    DINO = None

try:
    from src.models.traditional.nida import NIDA

    print("    ✓ NIDA")
except Exception as e:
    print(f"    ✗ NIDA: {e}")
    NIDA = None

# 导入评估器
print("\n[2] 导入评估器...")
try:
    from src.evaluation.evaluator import Evaluator

    print("    ✓ Evaluator")
except Exception as e:
    print(f"    ✗ Evaluator: {e}")
    Evaluator = None

# 导入深度学习模型
print("\n[3] 导入深度学习框架...")
torch_available = False
try:
    import torch

    torch_available = True
    print(f"    ✓ PyTorch {torch.__version__}")
    print(f"      CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"    ✗ PyTorch not available: {e}")
    print("      深度学习模型将被跳过")

# 导入深度学习模型
deep_models = {}
if torch_available:
    print("\n[4] 导入深度学习模型...")

    model_imports = [
        ('CDGK', 'src.models.deep_learning.cdgk', 'CDGK'),
        ('DeepCDM', 'src.models.deep_learning.deepcdm', 'DeepCDM'),
        ('DKT', 'src.models.deep_learning.dkt', 'DKT'),
        ('DKVMN', 'src.models.deep_learning.dkvmn', 'DKVMN'),
        ('SAKT', 'src.models.deep_learning.sakt', 'SAKT'),
        ('SAINT', 'src.models.deep_learning.saint', 'SAINT'),
        ('GIKT', 'src.models.deep_learning.gikt', 'GIKT'),
        ('NeuralCD', 'src.models.deep_learning.neuralcd', 'NeuralCD'),
        ('HybridCDM', 'src.models.hybrid.hybrid_cdm', 'HybridCDM'),
    ]

    for name, module, class_name in model_imports:
        try:
            exec(f"from {module} import {class_name}")
            deep_models[name] = eval(class_name)
            print(f"    ✓ {name}")
        except Exception as e:
            print(f"    ✗ {name}: {e}")

print("\n" + "=" * 60)
print("模块导入完成")
print("=" * 60)


def generate_synthetic_data(student_num=50, item_num=20, skill_num=5):
    """生成合成数据"""
    print(f"\n生成合成数据: {student_num}学生, {item_num}题目, {skill_num}技能")

    # 生成Q矩阵
    q_matrix = np.random.randint(0, 2, size=(item_num, skill_num))
    # 确保每个题目至少关联一个技能
    for i in range(item_num):
        if np.sum(q_matrix[i]) == 0:
            q_matrix[i, np.random.randint(skill_num)] = 1

    # 生成学生能力向量
    theta = np.random.randint(0, 2, size=(student_num, skill_num))

    # 生成题目参数
    g = np.random.uniform(0.1, 0.3, item_num)
    s = np.random.uniform(0.1, 0.3, item_num)

    # 生成答题数据
    data = []
    for student_id in range(student_num):
        for item_id in range(item_num):
            # 计算学生是否掌握了所有所需技能
            required_skills = np.where(q_matrix[item_id] == 1)[0]
            master_all = np.all(theta[student_id, required_skills] == 1)
            # 计算答对概率
            if master_all:
                p = 1 - s[item_id]
            else:
                p = g[item_id]
            # 生成答题结果
            score = 1 if np.random.random() < p else 0
            data.append([student_id, item_id, score])

    data = np.array(data)
    print(f"数据生成完成: {len(data)}条记录")

    return data, q_matrix


def compare_models(
        interactions_csv: str | None = None,
        q_matrix_csv: str | None = None,
        package_dir: str | None = None,
        dataset_name: str | None = None,
        use_synthetic: bool = True,
):
    """比较不同认知诊断模型的性能"""
    print("\n" + "=" * 60)
    print("开始模型比较实验")
    print("=" * 60)

    # 数据：真实 CSV / 标准化数据包 / 合成数据
    print("\n[Step 1] 准备数据...")
    if package_dir:
        from src.preprocessing.real_data import load_from_package_dir
        data, q_matrix = load_from_package_dir(package_dir)
        print(f"    已从数据包加载: {package_dir}")
    elif interactions_csv:
        from src.preprocessing.real_data import load_interactions_and_q
        data, q_matrix = load_interactions_and_q(
            interactions_csv,
            q_matrix_path=q_matrix_csv,
            dataset_name=dataset_name,
        )
        print(f"    已从真实数据加载: {interactions_csv}")
        if q_matrix_csv:
            print(f"    Q 矩阵: {q_matrix_csv}")
    elif use_synthetic:
        data, q_matrix = generate_synthetic_data(student_num=50, item_num=20, skill_num=5)
    else:
        print("错误: 请指定 --interactions 或 --package-dir，或使用默认合成数据（不加 --no-synthetic）")
        return

    student_num = len(np.unique(data[:, 0]))
    item_num = len(np.unique(data[:, 1]))
    skill_num = q_matrix.shape[1]
    print(f"    学生数={student_num}, 题目数={item_num}, 技能数={skill_num}, 记录数={len(data)}")

    # 初始化评估器
    print("\n[Step 2] 初始化评估器...")
    if Evaluator is None:
        print("错误: Evaluator模块未成功导入")
        return
    evaluator = Evaluator()

    # 构建模型列表
    print("\n[Step 3] 构建模型列表...")
    model_builders: List[Tuple[str, Callable[[], Any]]] = []

    # 添加传统模型
    if DINA is not None:
        model_builders.append(('DINA', lambda: DINA(q_matrix, max_iter=200)))
        print("    ✓ DINA (传统模型)")
    if DINO is not None:
        model_builders.append(('DINO', lambda: DINO(q_matrix, max_iter=200)))
        print("    ✓ DINO (传统模型)")
    if NIDA is not None:
        model_builders.append(('NIDA', lambda: NIDA(q_matrix, max_iter=200)))
        print("    ✓ NIDA (传统模型)")

    # 添加深度学习模型
    if torch_available and deep_models:
        print("\n    深度学习模型:")
        for name, model_class in deep_models.items():
            try:
                if name in ['CDGK', 'DeepCDM', 'HybridCDM']:
                    model_builders.append((name, lambda cls=model_class: cls(student_num, item_num, skill_num)))
                else:
                    model_builders.append((name, lambda cls=model_class: cls(item_num, skill_num)))
                print(f"        ✓ {name}")
            except Exception as e:
                print(f"        ✗ {name}: {e}")

    print(f"\n共 {len(model_builders)} 个模型待评估")

    # 评估模型
    print("\n[Step 4] 开始评估模型...")
    results = {}

    for idx, (model_name, model_builder) in enumerate(model_builders, 1):
        print(f"\n    [{idx}/{len(model_builders)}] 评估 {model_name}...")
        print("-" * 40)

        try:
            evaluation_result = evaluator.evaluate(
                model_builder,
                data,
                q_matrix,
                seed=RANDOM_SEED,
                n_stability_runs=3
            )
            results[model_name] = evaluation_result
            print(f"    ✓ {model_name} 评估完成")

            # 打印关键指标
            pred = evaluation_result.get('prediction', {})
            if pred:
                acc = pred.get('accuracy', 0)
                f1 = pred.get('f1', 0)
                auc = pred.get('roc_auc', 0)
                print(f"      Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        except Exception as e:
            print(f"    ✗ {model_name} 评估失败: {e}")
            import traceback
            traceback.print_exc()

    # 分析结果
    print("\n" + "=" * 60)
    print("[Step 5] 分析结果")
    print("=" * 60)

    if results:
        analyze_results(results)
    else:
        print("没有成功的评估结果")

    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)


def analyze_results(results):
    """分析模型比较结果"""
    print("\n预测性能:")
    print("-" * 60)

    prediction_metrics = {}
    for model_name, result in results.items():
        pred = result.get('prediction', {})
        if pred:
            prediction_metrics[model_name] = pred
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {pred.get('accuracy', 0):.4f}")
            print(f"  Precision: {pred.get('precision', 0):.4f}")
            print(f"  Recall:    {pred.get('recall', 0):.4f}")
            print(f"  F1 Score:  {pred.get('f1', 0):.4f}")
            print(f"  AUC:       {pred.get('roc_auc', 0):.4f}")

    print("\n\n可解释性评分:")
    print("-" * 60)
    for model_name, result in results.items():
        interp = result.get('interpretability', {})
        score = interp.get('interpretability_score', 0)
        print(f"  {model_name}: {score:.2f}")

    print("\n计算成本:")
    print("-" * 60)
    for model_name, result in results.items():
        cost = result.get('computational_cost', {})
        train_time = cost.get('training_time', 0)
        pred_time = cost.get('prediction_time', 0)
        print(f"  {model_name}: 训练 {train_time:.4f}s, 预测 {pred_time:.6f}s")

    print("\n教学适配性:")
    print("-" * 60)
    for model_name, result in results.items():
        teach = result.get('teaching_applicability', {})
        score = teach.get('applicability_score', 0)
        print(f"  {model_name}: {score:.2f}")

    # 可视化结果
    print("\n生成可视化图表...")
    visualize_results(results)
    print(f"图表已保存到: {EXPERIMENTS_DIR / 'model_comparison.png'}")


def visualize_results(results):
    """可视化模型比较结果"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cognitive Diagnosis Model Comparison', fontsize=14, fontweight='bold')

        # 提取指标
        model_names = list(results.keys())

        # 1. Accuracy
        accuracy = [results[m]['prediction'].get('accuracy', 0) for m in model_names]
        axes[0, 0].bar(model_names, accuracy, color='steelblue')
        axes[0, 0].set_title('Accuracy', fontweight='bold')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. F1 Score
        f1_scores = [results[m]['prediction'].get('f1', 0) for m in model_names]
        axes[0, 1].bar(model_names, f1_scores, color='darkorange')
        axes[0, 1].set_title('F1 Score', fontweight='bold')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. 可解释性和教学适配性
        interpretability = [results[m]['interpretability'].get('interpretability_score', 0) for m in model_names]
        applicability = [results[m]['teaching_applicability'].get('applicability_score', 0) for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35
        axes[1, 0].bar(x - width / 2, interpretability, width, label='Interpretability', color='green')
        axes[1, 0].bar(x + width / 2, applicability, width, label='Applicability', color='purple')
        axes[1, 0].set_title('Interpretability & Teaching Applicability', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)

        # 4. 训练时间
        training_times = [results[m]['computational_cost'].get('training_time', 0) for m in model_names]
        axes[1, 1].bar(model_names, training_times, color='crimson')
        axes[1, 1].set_title('Training Time (seconds)', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        out_path = EXPERIMENTS_DIR / "model_comparison.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ 图表已保存: {out_path}")

    except Exception as e:
        print(f"  ✗ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


def _parse_args():
    p = argparse.ArgumentParser(
        description="认知诊断模型对比：默认合成数据；可用真实 CSV 或标准化数据包。"
    )
    p.add_argument(
        "--interactions",
        default=None,
        help="答题记录 CSV（列含 student_id, item_id, score；可用 --dataset 做列名映射）",
    )
    p.add_argument(
        "--q-matrix",
        default=None,
        dest="q_matrix",
        help="Q 矩阵 CSV：item_id + 各技能 0/1 列",
    )
    p.add_argument(
        "--package-dir",
        default=None,
        help="build_public_dataset_package.py 生成的目录（含 all_interactions.csv 与 q_matrix.csv）",
    )
    p.add_argument(
        "--dataset",
        default=None,
        choices=("ASSISTments", "EdNet", "MOOPer"),
        help="公开数据集名称，用于 DataProcessor 列名映射",
    )
    p.add_argument(
        "--no-synthetic",
        action="store_true",
        help="若未提供真实数据则报错（禁止回退到合成数据）",
    )
    return p.parse_args()


if __name__ == "__main__":
    try:
        args = _parse_args()
        compare_models(
            interactions_csv=args.interactions,
            q_matrix_csv=args.q_matrix,
            package_dir=args.package_dir,
            dataset_name=args.dataset,
            use_synthetic=not args.no_synthetic,
        )
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n\n程序执行出错: {e}")
        import traceback

        traceback.print_exc()
