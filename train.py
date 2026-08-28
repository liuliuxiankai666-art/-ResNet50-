import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, auc
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from resnet50_1 import ResNet50WithGLCM, EnhancedDynamicConfusionClassifier
from dataload_qianyi01 import load_data_split

import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTI_GPU = True
GPU_MEMORY_FRACTION = 0.95

# ===== 路径与超参数（按需修改） =====
TRANSFER_LEARNING = False
# 预训练权重、源数据集和目标数据集路径
PRETRAINED_SOURCE_WEIGHT_PATH = os.path.normpath(r"E:\\\\\\需要修改")
SOURCE_DATA_ROOT = os.path.normpath(r"E:\A_sujuji\IMAGE\yuanxing_0.1-0.8_1024-512-8")
TARGET_DATA_ROOT = os.path.normpath(r"E:\A_sujuji\IMAGE\zhouchengtao_0.1-0.8-512-9")
# 迁移学习冻结与分组学习率
FREEZE_LAYERS = True
FREEZE_BACKBONE_UNTIL = 8
LR_BACKBONE_TARGET = 5e-6
LR_OTHERS_TARGET = 1e-4

SOURCE_TRAIN_DIR = os.path.join(SOURCE_DATA_ROOT, "train")
SOURCE_VAL_DIR = os.path.join(SOURCE_DATA_ROOT, "val")
SOURCE_TEST_DIR = os.path.join(SOURCE_DATA_ROOT, "test")

TARGET_TRAIN_DIR = os.path.join(TARGET_DATA_ROOT, "train")
TARGET_VAL_DIR = os.path.join(TARGET_DATA_ROOT, "val")
TARGET_TEST_DIR = os.path.join(TARGET_DATA_ROOT, "test")

# 训练超参数
EPOCHS_SOURCE = 150#训练轮次
EPOCHS_TARGET = 150
BATCH_SIZE = 8
LR_SOURCE = 1e-4
SEED = 2026  # 三次重复实验可依次设置为 46、2026、3407
WEIGHT_DECAY = 1e-5
PATIENCE = 20#耐力值一般为15或20就行
CLASS_NAMES = ['Ra0.05', 'Ra0.1', 'Ra0.2', 'Ra0.4', 'Ra0.8', 'Ra1.6']
NUM_CLASSES = len(CLASS_NAMES)
TARGET_CLASS_NAMES = ['Ra0.1', 'Ra0.2', 'Ra0.4', 'Ra0.8']
TARGET_TO_SOURCE_LABEL = {'Ra0.1': 1, 'Ra0.2': 2, 'Ra0.4': 3, 'Ra0.8': 4}
TARGET_LABELS = [TARGET_TO_SOURCE_LABEL[name] for name in TARGET_CLASS_NAMES]
NUM_TARGET_CLASSES = len(TARGET_CLASS_NAMES)
DISTURB_LABELS = [0, 5]

MAX_LOSS_THRESHOLD = 1e5
LOW_ACC_THRESHOLD = 25.0
STAGNATION_PATIENCE = 10

LR_FACTOR = 0.5
MIN_LR = 1e-8#最低学习率
ADJUST_INTERVAL = 2
LOSS_BASED_PATIENCE = 5

MID_TRAIN_ACC = 85.0
LATE_TRAIN_ACC = 95.0
EARLY_STAGNATION = 10
MID_STAGNATION = 5
LATE_STAGNATION = 2

REAL_TIME_PLOT_INTERVAL = 1
PLOT_DPI = 300

# 检查点保存参数
SAVE_CHECKPOINT = True
CHECKPOINT_INTERVAL = 15
DEFAULT_PRETRAINED_NAME = "source_best_model.pth"

def set_seed(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

def save_csv_log(row, csv_path, header=False):
    df = pd.DataFrame([row])
    if "Learning_Rate" in df.columns:
        df["Learning_Rate"] = pd.to_numeric(df["Learning_Rate"], errors='coerce')
    df.to_csv(csv_path, mode='a', header=header, index=False)

def plot_metric(values, title, ylabel, save_path, dpi=PLOT_DPI):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(values) + 1), values, linewidth=2, marker='o', markersize=2)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_compare_metric(train_values, val_values, title, ylabel, save_path, dpi=PLOT_DPI):
    plt.figure(figsize=(8, 4))
    epochs = range(1, len(train_values) + 1)
    plt.plot(epochs, train_values, linewidth=2, marker='o', markersize=2, label=f"Train {ylabel}", color='#2E86AB')
    plt.plot(epochs, val_values, linewidth=2, marker='s', markersize=2, label=f"Val {ylabel}", color='#A23B72')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def compute_map(y_true, y_pred_probs, num_classes, eps=1e-8, filter_labels=None):
    average_precisions = []
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    y_true_one_hot = np.zeros((len(y_true), num_classes))
    valid_indices = y_true < num_classes
    y_true_one_hot[valid_indices, y_true[valid_indices].astype(int)] = 1

    target_cls = filter_labels if filter_labels is not None else range(num_classes)
    for cls in target_cls:
        if cls >= num_classes:
            average_precisions.append(0.0)
            continue
        y_cls_true = y_true_one_hot[:, cls]
        y_cls_prob = y_pred_probs[:, cls]

        if np.sum(y_cls_true) == 0:
            average_precisions.append(0.0)
            continue

        precision, recall, _ = precision_recall_curve(y_cls_true, y_cls_prob)
        average_precisions.append(auc(recall, precision))

    return np.mean(average_precisions) if average_precisions else 0.0

def init_realtime_plot(interactive=True):
    if not interactive:
        return None, None, None, None
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    train_line, = ax.plot([], [], linewidth=2, marker='o', markersize=2, label="Train Accuracy (%)", color='#2E86AB')
    val_line, = ax.plot([], [], linewidth=2, marker='s', markersize=2, label="Val Accuracy (%)", color='#A23B72')
    ax.set_title("Real-Time Train & Validation Accuracy", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlim(1, 10)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig, ax, train_line, val_line

def update_realtime_plot(fig, ax, train_line, val_line, train_accs, val_accs):
    if fig is None:
        return
    epochs = range(1, len(train_accs) + 1)
    train_line.set_data(epochs, train_accs)
    val_line.set_data(epochs, val_accs)
    ax.set_xlim(1, len(train_accs) + 1)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()
    time.sleep(REAL_TIME_PLOT_INTERVAL)

def calculate_valid_acc(y_true, y_pred, valid_labels=TARGET_LABELS, disturb_labels=DISTURB_LABELS):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    valid_mask = np.isin(y_true, valid_labels) & ~np.isin(y_pred, disturb_labels)
    if np.sum(valid_mask) == 0:
        return 0.0
    return accuracy_score(y_true[valid_mask], y_pred[valid_mask]) * 100

def load_pretrained_weights(model, pretrained_path, device):
    state_dict = torch.load(pretrained_path, map_location=device)
    if isinstance(model, nn.DataParallel) and not any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    elif not isinstance(model, nn.DataParallel) and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    return model.load_state_dict(state_dict, strict=False)

def evaluate(model, loader, criterion, num_classes=NUM_CLASSES, class_names=CLASS_NAMES, desc="Evaluating",
             is_target_dataset=False):
    model.eval()
    y_true, y_pred, y_pred_probs = [], [], []
    val_loss = 0.0

    with torch.no_grad():
        for imgs, glcms, labels in tqdm(loader, desc=desc, leave=False, ncols=100):
            imgs, glcms, labels = imgs.to(device), glcms.to(device), labels.to(device)
            outputs = model(imgs, glcms)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_probs.extend(probs.cpu().numpy())

    avg_val_loss = val_loss / len(loader) if len(loader) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred) * 100 if len(y_true) > 0 else 0.0
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    valid_acc = calculate_valid_acc(y_true_np, y_pred_np) if is_target_dataset else acc

    if len(y_true) == 0:
        return acc, valid_acc, avg_val_loss, 0.0, 0.0, 0.0, 0.0, {}, np.zeros((num_classes, num_classes)), [], [], [], 0, 0.0, {}

    report_full = classification_report(
        y_true_np, y_pred_np, target_names=class_names, output_dict=True, zero_division=0, labels=range(num_classes)
    )
    cm = confusion_matrix(y_true_np, y_pred_np, labels=range(num_classes))

    if is_target_dataset:
        valid_mask = np.isin(y_true_np, TARGET_LABELS)
        report_valid = classification_report(
            y_true_np[valid_mask], y_pred_np[valid_mask],
            target_names=TARGET_CLASS_NAMES, output_dict=True, zero_division=0, labels=TARGET_LABELS
        )
        macro_p = report_valid['macro avg']['precision']
        macro_r = report_valid['macro avg']['recall']
        macro_f1 = report_valid['macro avg']['f1-score']
    else:
        macro_p = report_full['macro avg']['precision']
        macro_r = report_full['macro avg']['recall']
        macro_f1 = report_full['macro avg']['f1-score']

    map_score = compute_map(y_true_np, np.array(y_pred_probs), num_classes, filter_labels=TARGET_LABELS if is_target_dataset else None) * 100

    cross_mis_count, cross_mis_rate, mis_distribution = 0, 0.0, {}
    if is_target_dataset:
        valid_sample_mask = np.isin(y_true_np, TARGET_LABELS)
        cross_mis_mask = valid_sample_mask & np.isin(y_pred_np, DISTURB_LABELS)
        cross_mis_count = np.sum(cross_mis_mask)
        cross_mis_rate = (cross_mis_count / np.sum(valid_sample_mask) * 100) if np.sum(valid_sample_mask) > 0 else 0.0
        mis_distribution = {label: np.sum((y_pred_np == label) & cross_mis_mask) for label in DISTURB_LABELS}

    return acc, valid_acc, avg_val_loss, macro_p, macro_r, macro_f1, map_score, report_full, cm, y_true, y_pred, y_pred_probs, cross_mis_count, cross_mis_rate, mis_distribution

def generate_complete_report(
        total_time, epochs_trained, best_epoch,
        train_metrics, val_metrics, test_metrics,
        best_lr, lr_adjust_records, save_path,
        stage_name="训练",
        lr_config=None,
        best_val_report=None,
        best_val_cm=None,
        best_val_y_true=None,
        best_val_y_pred=None,
        class_names=None
):
    lr_config = lr_config or {}
    num_classes = len(class_names) if class_names else 0
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"📊 模型{stage_name}与评估综合报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"保存路径: {os.path.dirname(save_path)}\n")
        f.write(f"设备: {device} (多GPU: {MULTI_GPU})\n")
        f.write(f"随机数种子: {SEED}\n")
        f.write(f"总训练轮次: {epochs_trained}/{EPOCHS_SOURCE if '源数据集' in stage_name else EPOCHS_TARGET}\n")
        f.write(f"总训练时间: {total_time:.2f}秒 (约{total_time / 60:.2f}分钟)\n")
        f.write(f"最佳验证模型所在轮次: {best_epoch + 1}\n")
        f.write(f"最佳模型对应学习率: {best_lr:.2e}\n")
        f.write(f"类别配置: {num_classes}类 ({', '.join(class_names)})\n")

        if "目标数据集" in stage_name:
            f.write(f"微调配置: {'冻结backbone前' + str(FREEZE_BACKBONE_UNTIL) + '个组件' if FREEZE_LAYERS else '不冻结'}\n")
            if lr_config.get('is_grouped', False):
                f.write(f"分组学习率: Backbone={lr_config.get('lr_backbone', 0):.2e}, 其他模块={lr_config.get('lr_others', 0):.2e}\n")
            else:
                f.write(f"学习率: {lr_config.get('lr', 0):.2e}\n")
            f.write(f"目标数据集说明: 实际4类，映射为源数据集标签1-4，排除干扰类Ra0.05(0)/Ra1.6(5)\n")
            f.write(f"有效类准确率: 仅计算真实标签为目标4类且预测标签非干扰类的样本准确率\n")
        f.write("=" * 80 + "\n\n")

        f.write("📉 学习率调整记录\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'调整轮次':<10} {'调整前LR':<15} {'调整后LR':<15} {'调整原因':<30}\n")
        f.write("-" * 70 + "\n")
        for rec in lr_adjust_records:
            f.write(f"{rec['epoch']:<10} {rec['lr_before']:<15.2e} {rec['lr_after']:<15.2e} {rec['reason']:<30}\n")
        if not lr_adjust_records:
            f.write("无主动调整记录（仅依赖损失驱动调度）\n")
        f.write("\n")

        f.write("📈 训练集指标汇总\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Epoch':<8} {'Loss':<10} {'Accuracy(%)':<15} {'Learning Rate':<15}\n")
        f.write("-" * 50 + "\n")
        for epoch in range(epochs_trained):
            f.write(f"{epoch + 1:<8} {train_metrics['losses'][epoch]:<10.4f} {train_metrics['accs'][epoch]:<15.2f} {train_metrics['lrs'][epoch]:<15.2e}\n")
        f.write(f"\n最终训练指标:\n")
        f.write(f"  平均损失: {np.mean(train_metrics['losses']):.4f}\n")
        f.write(f"  最终准确率: {train_metrics['accs'][-1]:.2f}%\n")
        f.write(f"  最终学习率: {train_metrics['lrs'][-1]:.2e}\n\n")

        f.write("📈 验证集指标汇总\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Epoch':<8} {'Loss':<10} {'Accuracy(%)':<15} {'Valid_Acc(%)':<15} {'MAP(%)':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Cross-MisRate(%)':<15}\n")
        f.write("-" * 100 + "\n")
        for epoch in range(epochs_trained):
            cross_mis_rate = val_metrics.get('cross_mis_rates', [0.0] * epochs_trained)[epoch]
            valid_acc = val_metrics.get('valid_accs', [0.0] * epochs_trained)[epoch]
            f.write(f"{epoch + 1:<8} {val_metrics['losses'][epoch]:<10.4f} {val_metrics['accs'][epoch]:<15.2f} {valid_acc:<15.2f} {val_metrics['maps'][epoch]:<10.2f} {val_metrics['precisions'][epoch]:<12.4f} {val_metrics['recalls'][epoch]:<12.4f} {val_metrics['f1s'][epoch]:<12.4f} {cross_mis_rate:<15.2f}\n")
        f.write(f"\n🏆 最佳验证指标 (Epoch {best_epoch + 1}):\n")
        f.write(f"  损失: {val_metrics['losses'][best_epoch]:.4f}\n")
        f.write(f"  全量准确率: {val_metrics['accs'][best_epoch]:.2f}%\n")
        f.write(f"  有效类准确率: {val_metrics['valid_accs'][best_epoch]:.2f}%\n")
        f.write(f"  MAP: {val_metrics['maps'][best_epoch]:.2f}%\n")
        f.write(f"  宏平均精确率: {val_metrics['precisions'][best_epoch]:.4f}\n")
        f.write(f"  宏平均召回率: {val_metrics['recalls'][best_epoch]:.4f}\n")
        f.write(f"  宏平均F1分数: {val_metrics['f1s'][best_epoch]:.4f}\n")
        if "目标数据集" in stage_name:
            f.write(f"  跨数据集错分率: {val_metrics['cross_mis_rates'][best_epoch]:.2f}%\n")
        f.write(f"  对应学习率: {train_metrics['lrs'][best_epoch]:.2e}\n\n")

        if best_val_report and best_val_y_true and best_val_y_pred:
            f.write("📋 最佳验证轮次分类报告（6类全量）:\n")
            f.write("-" * 60 + "\n")
            f.write(classification_report(best_val_y_true, best_val_y_pred, target_names=class_names, zero_division=0, labels=range(num_classes)) + "\n")

            if "目标数据集" in stage_name:
                target_mask = np.isin(best_val_y_true, TARGET_LABELS)
                y_true_target = np.array(best_val_y_true)[target_mask]
                y_pred_target = np.array(best_val_y_pred)[target_mask]
                if len(y_true_target) > 0:
                    f.write("📋 最佳验证轮次-目标4类专项报告（排除干扰类）:\n")
                    f.write("-" * 60 + "\n")
                    f.write(classification_report(y_true_target, y_pred_target, target_names=TARGET_CLASS_NAMES, labels=TARGET_LABELS, zero_division=0) + "\n")

        if best_val_cm is not None:
            f.write("📊 最佳验证轮次混淆矩阵（6类全量）:\n")
            f.write("-" * 60 + "\n")
            f.write(np.array2string(best_val_cm, formatter={'int': lambda x: f"{x:4d}"}))
            f.write("\n\n")

        f.write("🧪 测试集评估结果 (基于最佳验证模型)\n")
        f.write("-" * 40 + "\n")
        f.write("整体指标（6类全量）:\n")
        f.write(f"  准确率: {test_metrics['acc']:.2f}%\n")
        f.write(f"  有效类准确率: {test_metrics['valid_acc']:.2f}%\n")
        f.write(f"  MAP: {test_metrics['map']:.2f}%\n")
        f.write(f"  宏平均精确率: {test_metrics['macro_p']:.4f}\n")
        f.write(f"  宏平均召回率: {test_metrics['macro_r']:.4f}\n")
        f.write(f"  宏平均F1分数: {test_metrics['macro_f1']:.4f}\n")

        if "目标数据集" in stage_name:
            f.write(f"\n🔍 跨数据集错分统计（预测为源额外类Ra0.05/Ra1.6）:\n")
            f.write(f"  错分样本数: {test_metrics['cross_mis_count']}个\n")
            f.write(f"  错分率: {test_metrics['cross_mis_rate']:.2f}%\n")
            f.write(f"  错分分布:\n")
            f.write(f"    - 预测为Ra0.05（源额外类）: {test_metrics['mis_distribution'].get(0, 0)}个\n")
            f.write(f"    - 预测为Ra1.6（源额外类）: {test_metrics['mis_distribution'].get(5, 0)}个\n")

        f.write("\n分类报告（6类全量）:\n")
        f.write(classification_report(test_metrics['y_true'], test_metrics['y_pred'], target_names=class_names, zero_division=0, labels=range(num_classes)) + "\n")
        if "目标数据集" in stage_name:
            target_mask = np.isin(test_metrics['y_true'], TARGET_LABELS)
            y_true_target = np.array(test_metrics['y_true'])[target_mask]
            y_pred_target = np.array(test_metrics['y_pred'])[target_mask]
            if len(y_true_target) > 0:
                f.write("🎯 测试集-目标4类专项分类报告（排除干扰类）:\n")
                f.write("-" * 60 + "\n")
                f.write(classification_report(y_true_target, y_pred_target, target_names=TARGET_CLASS_NAMES, labels=TARGET_LABELS, zero_division=0) + "\n")

        f.write("混淆矩阵（6类全量）:\n")
        f.write(np.array2string(test_metrics['cm'], formatter={'int': lambda x: f"{x:4d}"}))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("✅ 报告结束\n")

def train_stage(save_dir, config, data_dirs, epochs, lr, stage_name, class_names, num_classes,
                pretrained_path=None, is_target_stage=False):

    set_seed(SEED)
    tqdm.write(f"🎲 {stage_name} - 当前随机数种子：{SEED}")
    SAVE_MODEL_PATH = config['SAVE_MODEL_PATH']
    CSV_PATH = config['CSV_PATH']
    COMPLETE_REPORT_PATH = config['COMPLETE_REPORT_PATH']
    STOP_LOG_PATH = config['STOP_LOG_PATH']
    ERROR_LOG_PATH = config['ERROR_LOG_PATH']
    CHECKPOINT_PATH = os.path.join(save_dir, f"checkpoint_{stage_name.replace(' ', '_')}.pth")

    best_val_results = {'report': None, 'cm': None, 'y_true': None, 'y_pred': None}
    training_completed = False

    try:

        train_loader, val_loader, test_loader = load_data_split(
            data_dirs['train'], data_dirs['val'], data_dirs['test'],
            batch_size=BATCH_SIZE, is_target_dataset=is_target_stage, target_to_source_label=TARGET_TO_SOURCE_LABEL
        )
        tqdm.write(f"✅ {stage_name} - 数据加载完成：训练集{len(train_loader.dataset)}样本，验证集{len(val_loader.dataset)}样本，测试集{len(test_loader.dataset)}样本")
    except Exception as e:
        with open(STOP_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(f"终止时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"终止阶段: {stage_name} - 数据加载\n")
            f.write(f"终止原因: 数据加载失败，错误信息：{str(e)}\n")
        raise RuntimeError(f"{stage_name} - 数据加载失败：{str(e)}") from e

    model = ResNet50WithGLCM(num_classes=NUM_CLASSES, pretrained_backbone=False).to(device)

    if pretrained_path and os.path.exists(pretrained_path):
        tqdm.write(f"📥 {stage_name} - 加载预训练模型：{pretrained_path}")
        missing_keys, unexpected_keys = load_pretrained_weights(model, pretrained_path, device)
        if missing_keys:
            tqdm.write(f"⚠️ {stage_name} - 模型加载缺失参数: {missing_keys[:5]}...")
        if unexpected_keys:
            tqdm.write(f"⚠️ {stage_name} - 模型加载多余参数: {unexpected_keys[:5]}...")
        if is_target_stage:
            tqdm.write(f"🔄 {stage_name} - 模型权重加载完成，保持6类输出（含源额外类Ra0.05/Ra1.6）")

    lr_config = {'lr': lr, 'is_grouped': False}
    if is_target_stage and FREEZE_LAYERS:
        tqdm.write(f"🔒 {stage_name} - 执行模型冻结：冻结backbone前{FREEZE_BACKBONE_UNTIL}个组件")
        model.freeze_layers(freeze_backbone_until=FREEZE_BACKBONE_UNTIL)
        lr_config = {'is_grouped': True, 'lr_backbone': LR_BACKBONE_TARGET, 'lr_others': LR_OTHERS_TARGET}

    if MULTI_GPU and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        tqdm.write(f"✅ {stage_name} - 启用多GPU训练：共{torch.cuda.device_count()}个GPU")

    if lr_config['is_grouped']:
        params_groups = model.module.get_trainable_params_groups(LR_BACKBONE_TARGET, LR_OTHERS_TARGET) if isinstance(model, nn.DataParallel) else model.get_trainable_params_groups(LR_BACKBONE_TARGET, LR_OTHERS_TARGET)
        optimizer = optim.AdamW(params_groups, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss()
    loss_based_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=LOSS_BASED_PATIENCE, min_lr=MIN_LR)

    lr_adjust_records = []
    last_adjust_epoch = -ADJUST_INTERVAL
    best_val_acc = 0.0
    best_val_map = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    total_time = 0.0
    epochs_stagnant = 0

    train_metrics = {'losses': [], 'accs': [], 'lrs': []}
    val_metrics = {
        'losses': [], 'accs': [], 'valid_accs': [],
        'precisions': [], 'recalls': [], 'f1s': [], 'maps': [],
        'cross_mis_counts': [], 'cross_mis_rates': [], 'mis_distributions': []
    }

    interactive_env = plt.get_backend() not in ['Agg', 'pdf', 'svg']
    fig, ax, train_line, val_line = init_realtime_plot(interactive=interactive_env)

    csv_headers = {
        "Epoch": "Epoch", "Train_Loss": "Train_Loss", "Train_Acc(%)": "Train_Acc(%)",
        "Val_Loss": "Val_Loss", "Val_Acc(%)": "Val_Acc(%)", "Val_Valid_Acc(%)": "Val_Valid_Acc(%)",
        "Val_MAP(%)": "Val_MAP(%)", "Val_Precision": "Val_Precision", "Val_Recall": "Val_Recall",
        "Val_F1": "Val_F1", "Learning_Rate": "Learning_Rate", "LR_Adjust_Reason": "LR_Adjust_Reason"
    }
    if is_target_stage:
        csv_headers.update({"Val_Cross_Mis_Count": "Val_Cross_Mis_Count", "Val_Cross_Mis_Rate(%)": "Val_Cross_Mis_Rate(%)"})
    save_csv_log(csv_headers, CSV_PATH, header=True)

    tqdm.write("=" * 80)
    tqdm.write(f"🚀 {stage_name} - 训练配置信息")
    tqdm.write("=" * 80)
    tqdm.write(f"📌 学习率调度规则：")
    tqdm.write(f"   - 前期（训练准确率<{MID_TRAIN_ACC}%）：验证指标连续{EARLY_STAGNATION}轮未提升 → 调整")
    tqdm.write(f"   - 中期（{MID_TRAIN_ACC}%≤训练准确率<{LATE_TRAIN_ACC}%）：验证指标连续{MID_STAGNATION}轮未提升 → 调整")
    tqdm.write(f"   - 后期（训练准确率≥{LATE_TRAIN_ACC}%）：验证指标连续{LATE_STAGNATION}轮未提升 → 调整")
    tqdm.write(f"   - 调整因子：{LR_FACTOR}，最小学习率：{MIN_LR:.2e}，调整间隔≥{ADJUST_INTERVAL}轮")
    tqdm.write(f"📌 其他配置：")
    tqdm.write(f"   - 随机数种子：{SEED}")
    tqdm.write(f"   - 批次大小：{BATCH_SIZE}，权重衰减：{WEIGHT_DECAY:.2e}")
    tqdm.write(f"   - 早停容忍：{PATIENCE}轮，异常损失阈值：{MAX_LOSS_THRESHOLD:.1e}")
    if lr_config['is_grouped']:
        tqdm.write(f"   - 学习率配置：分组学习率（Backbone: {LR_BACKBONE_TARGET:.2e}, 其他模块: {LR_OTHERS_TARGET:.2e}）")
    else:
        tqdm.write(f"   - 学习率配置：统一学习率（{lr:.2e}）")
    if is_target_stage and FREEZE_LAYERS:
        tqdm.write(f"   - 冻结配置：冻结backbone前{FREEZE_BACKBONE_UNTIL}个组件（仅训练深层和顶层）")
    tqdm.write(f"   - 类别列表：{', '.join(class_names)}（共{num_classes}类）")
    if is_target_stage:
        tqdm.write(f"   - 目标数据集标签映射：{TARGET_TO_SOURCE_LABEL}（模型输出6类）")
        tqdm.write(f"   - 干扰类排除：Ra0.05(0)/Ra1.6(5)，仅计算有效类准确率")
        tqdm.write(f"   - 最佳模型更新规则：仅有效类准确率提升时更新，MAP提升不更新")
    tqdm.write("=" * 80 + "\n")

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        lr_adjust_reason = "无"

        for imgs, glcms, labels in tqdm(train_loader, desc=f"{stage_name} - Epoch {epoch + 1}/{epochs}", ncols=100):
            imgs, glcms, labels = imgs.to(device), glcms.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, glcms)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            ce_loss = criterion(outputs, labels)
            try:
                reg_loss = model.module.get_regularization_loss() if isinstance(model, nn.DataParallel) else model.get_regularization_loss()
            except AttributeError:
                reg_loss = torch.tensor(0.0, device=device)
                if epoch == 0:
                    tqdm.write("⚠️ 模型未实现get_regularization_loss方法，正则化损失设为0")
            loss = ce_loss + reg_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_time = time.time() - start_time
        total_time += epoch_time
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_acc = (correct / total * 100) if total > 0 else 0.0
        updated_lr = current_lr

        train_metrics['losses'].append(epoch_loss)
        train_metrics['accs'].append(train_acc)
        train_metrics['lrs'].append(updated_lr)

        if epoch_loss > MAX_LOSS_THRESHOLD:
            with open(STOP_LOG_PATH, "w", encoding="utf-8") as f:
                f.write(f"终止时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"终止阶段: {stage_name} - Epoch {epoch + 1}\n")
                f.write(f"终止原因: 训练损失异常（当前损失={epoch_loss:.2f} > 阈值{MAX_LOSS_THRESHOLD}）\n")
                f.write(f"已训练轮次: {epoch + 1}/{epochs}\n")
                f.write(f"当前训练指标: 损失={epoch_loss:.4f}, 准确率={train_acc:.2f}%, 学习率={updated_lr:.2e}\n")
            tqdm.write(f"\n❌ {stage_name} - 训练损失异常，终止训练！")
            if fig is not None:
                fig.savefig(os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_realtime_plot.png"), dpi=PLOT_DPI)
                plt.ioff()
                plt.close(fig)
            break

        if train_acc < LOW_ACC_THRESHOLD:
            epochs_stagnant += 1
            tqdm.write(f"⚠️ {stage_name} - 训练停滞警告：准确率{train_acc:.2f}% < 阈值{LOW_ACC_THRESHOLD}%，已连续{epochs_stagnant}/{STAGNATION_PATIENCE}轮")
            if epochs_stagnant >= STAGNATION_PATIENCE:
                with open(STOP_LOG_PATH, "w", encoding="utf-8") as f:
                    f.write(f"终止时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"终止阶段: {stage_name} - Epoch {epoch + 1}\n")
                    f.write(f"终止原因: 连续{STAGNATION_PATIENCE}轮训练准确率低于{LOW_ACC_THRESHOLD}%（训练停滞）\n")
                    f.write(f"已训练轮次: {epoch + 1}/{epochs}\n")
                tqdm.write(f"\n❌ {stage_name} - 训练停滞，终止训练！")
                if fig is not None:
                    fig.savefig(os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_realtime_plot.png"), dpi=PLOT_DPI)
                plt.ioff()
                plt.close(fig)
                break
        else:
            epochs_stagnant = 0

        val_acc, val_valid_acc, val_loss, val_precision, val_recall, val_f1, val_map, val_report, val_cm, val_y_true, val_y_pred, _, cross_mis_count, cross_mis_rate, mis_distribution = evaluate(
            model, val_loader, criterion, num_classes, class_names, desc=f"{stage_name} - Validating Epoch {epoch + 1}", is_target_dataset=is_target_stage
        )

        val_metrics['losses'].append(val_loss)
        val_metrics['accs'].append(val_acc)
        val_metrics['valid_accs'].append(val_valid_acc)
        val_metrics['precisions'].append(val_precision)
        val_metrics['recalls'].append(val_recall)
        val_metrics['f1s'].append(val_f1)
        val_metrics['maps'].append(val_map)
        if is_target_stage:
            val_metrics['cross_mis_counts'].append(cross_mis_count)
            val_metrics['cross_mis_rates'].append(cross_mis_rate)
            val_metrics['mis_distributions'].append(mis_distribution)
        else:
            val_metrics['cross_mis_counts'].append(0)
            val_metrics['cross_mis_rates'].append(0.0)
            val_metrics['mis_distributions'].append({})

        metric_improved = False
        improve_reason = ""
        if is_target_stage:
            if val_valid_acc > best_val_acc + 1e-5:
                best_val_acc = val_valid_acc
                improve_reason = "有效类准确率提升"
                metric_improved = True
        else:
            if val_acc > best_val_acc + 1e-5:
                best_val_acc = val_acc
                improve_reason = "全量准确率提升"
                metric_improved = True

        map_improved = val_map > best_val_map + 1e-5
        map_info = f"（MAP同步提升至{val_map:.2f}%）" if map_improved else ""
        if map_improved:
            best_val_map = val_map

        if metric_improved:
            best_epoch = epoch
            epochs_no_improve = 0
            best_val_results['report'] = val_report
            best_val_results['cm'] = val_cm
            best_val_results['y_true'] = val_y_true
            best_val_results['y_pred'] = val_y_pred
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, SAVE_MODEL_PATH)
            tqdm.write(f"📌 {stage_name} - 最佳模型更新！Epoch {epoch + 1} - 验证{improve_reason}：{val_valid_acc if is_target_stage else val_acc:.2f}% {map_info}")
            if is_target_stage:
                tqdm.write(f"   - 跨数据集错分率：{cross_mis_rate:.2f}%（{cross_mis_count}个样本错分到Ra0.05/Ra1.6）")
        else:
            epochs_no_improve += 1
            if map_improved:
                tqdm.write(f"📌 {stage_name} - Epoch {epoch + 1} - MAP提升至{val_map:.2f}%，但准确率未提升，不更新最佳模型")

        current_epoch = epoch + 1
        adjust_needed = False
        adjust_reason = ""
        stage = "前期" if train_acc < MID_TRAIN_ACC else "中期" if train_acc < LATE_TRAIN_ACC else "后期"
        stagnation_threshold = EARLY_STAGNATION if stage == "前期" else MID_STAGNATION if stage == "中期" else LATE_STAGNATION

        if current_lr > MIN_LR and epochs_no_improve >= stagnation_threshold and (current_epoch - last_adjust_epoch) >= ADJUST_INTERVAL:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * LR_FACTOR, MIN_LR)
            new_lr = optimizer.param_groups[0]['lr']
            lr_adjust_records.append({"epoch": current_epoch, "lr_before": current_lr, "lr_after": new_lr, "reason": f"{stage}-验证指标连续{epochs_no_improve}轮未提升"})
            lr_adjust_reason = adjust_reason
            last_adjust_epoch = current_epoch
            updated_lr = new_lr
            epochs_no_improve = 0
            tqdm.write(f"🔧 {stage_name} - 学习率调整（准确率驱动）：{current_lr:.2e} → {new_lr:.2e}，原因：{adjust_reason}")

        prev_lr_for_loss = updated_lr
        loss_based_scheduler.step(val_loss)
        post_loss_lr = optimizer.param_groups[0]['lr']
        if post_loss_lr < prev_lr_for_loss and post_loss_lr >= MIN_LR:
            lr_adjust_records.append({"epoch": current_epoch, "lr_before": prev_lr_for_loss, "lr_after": post_loss_lr, "reason": f"验证损失连续{LOSS_BASED_PATIENCE}轮无下降（损失驱动）"})
            lr_adjust_reason = "验证损失无下降（损失驱动）"
            last_adjust_epoch = current_epoch
            updated_lr = post_loss_lr
            tqdm.write(f"🔧 {stage_name} - 学习率调整（损失驱动）：{prev_lr_for_loss:.2e} → {post_loss_lr:.2e}，原因：验证损失无下降")

        train_metrics['lrs'][-1] = updated_lr

        reg_loss_val = reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
        log_msg = (
            f"[{current_epoch}/{epochs}] Loss: {epoch_loss:.4f} (CE: {ce_loss.item():.4f}, Reg: {reg_loss_val:.6f}) | "
            f"Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )
        if is_target_stage:
            log_msg += f" | Val Valid Acc: {val_valid_acc:.2f}%"
        log_msg += f" | Val MAP: {val_map:.2f}% | P: {val_precision:.4f} R: {val_recall:.4f} F1: {val_f1:.4f}"
        if is_target_stage:
            log_msg += f" | Cross-Mis: {cross_mis_count}({cross_mis_rate:.1f}%)"
        log_msg += f" | LR: {updated_lr:.2e} | Adjust: {lr_adjust_reason} | Time: {epoch_time:.1f}s"
        tqdm.write(log_msg)

        csv_row = {
            "Epoch": current_epoch, "Train_Loss": epoch_loss, "Train_Acc(%)": train_acc,
            "Val_Loss": val_loss, "Val_Acc(%)": val_acc, "Val_Valid_Acc(%)": val_valid_acc,
            "Val_MAP(%)": val_map, "Val_Precision": val_precision, "Val_Recall": val_recall,
            "Val_F1": val_f1, "Learning_Rate": updated_lr, "LR_Adjust_Reason": lr_adjust_reason
        }
        if is_target_stage:
            csv_row.update({"Val_Cross_Mis_Count": cross_mis_count, "Val_Cross_Mis_Rate(%)": cross_mis_rate})
        save_csv_log(csv_row, CSV_PATH)

        update_realtime_plot(fig, ax, train_line, val_line, train_metrics['accs'], val_metrics['valid_accs'] if is_target_stage else val_metrics['accs'])

        if SAVE_CHECKPOINT and (current_epoch % CHECKPOINT_INTERVAL == 0):
            checkpoint = {
                'seed': SEED,
                'epoch': current_epoch, 'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_val_acc': best_val_acc, 'best_val_map': best_val_map,
                'train_metrics': train_metrics, 'val_metrics': val_metrics, 'lr_adjust_records': lr_adjust_records,
                'lr_config': lr_config, 'best_val_results': best_val_results
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            tqdm.write(f"💾 {stage_name} - 断点保存：Epoch {current_epoch} → {CHECKPOINT_PATH}")

        if epochs_no_improve >= PATIENCE:
            with open(STOP_LOG_PATH, "w", encoding="utf-8") as f:
                f.write(f"终止时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"终止阶段: {stage_name} - Epoch {current_epoch}\n")
                f.write(f"终止原因: 验证集连续{PATIENCE}轮无提升（早停机制）\n")
                f.write(f"已训练轮次: {current_epoch}/{epochs}\n")
                f.write(f"最佳模型轮次: {best_epoch + 1}，最佳验证{'有效类' if is_target_stage else '全量'}准确率: {best_val_acc:.2f}%，最佳验证MAP: {best_val_map:.2f}%\n")
                if is_target_stage:
                    f.write(f"最佳模型跨数据集错分率: {val_metrics['cross_mis_rates'][best_epoch]:.2f}%\n")
            tqdm.write(f"\n⚠️ {stage_name} - 验证集 {PATIENCE} 轮无提升，提前停止训练。")
            if fig is not None:
                fig.savefig(os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_realtime_plot.png"), dpi=PLOT_DPI)
                plt.ioff()
                plt.close(fig)
            break

        if epoch == epochs - 1:
            training_completed = True

    if training_completed and fig is not None:
        fig.savefig(os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_realtime_plot.png"), dpi=PLOT_DPI)
        plt.ioff()
        plt.close(fig)

    epochs_trained = len(train_metrics['losses'])
    if epochs_trained > 0 and os.path.exists(SAVE_MODEL_PATH):
        state_dict = torch.load(SAVE_MODEL_PATH, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        tqdm.write(f"\n✅ {stage_name} - 加载最佳模型：{SAVE_MODEL_PATH}")

        acc, valid_acc, _, macro_p, macro_r, macro_f1, map_score, _, cm, y_true, y_pred, _, cross_mis_count, cross_mis_rate, mis_distribution = evaluate(
            model, test_loader, criterion, num_classes, class_names, desc=f"{stage_name} - Testing", is_target_dataset=is_target_stage
        )

        model.train()
        with torch.no_grad():
            model_for_confusion = model.module if isinstance(model, nn.DataParallel) else model
            model_for_confusion.update_confusion_pairs(y_true, y_pred)
        model.eval()

        test_metrics = {
            'acc': acc, 'valid_acc': valid_acc, 'map': map_score,
            'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
            'cm': cm, 'y_true': y_true, 'y_pred': y_pred,
            'cross_mis_count': cross_mis_count, 'cross_mis_rate': cross_mis_rate, 'mis_distribution': mis_distribution
        }

        tqdm.write("\n" + "=" * 60)
        tqdm.write(f"🧪 {stage_name} - 测试集最终结果")
        tqdm.write("=" * 60)
        tqdm.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0, labels=range(num_classes)))
        tqdm.write(f"测试集全量准确率: {acc:.2f}%")
        tqdm.write(f"测试集有效类准确率: {valid_acc:.2f}%")
        tqdm.write(f"测试集MAP: {map_score:.2f}%")
        if is_target_stage:
            tqdm.write(f"\n🔍 跨数据集错分统计（预测为源额外类Ra0.05/Ra1.6）：")
            tqdm.write(f"  错分样本数: {cross_mis_count}个")
            tqdm.write(f"  错分率: {cross_mis_rate:.2f}%")
            tqdm.write(f"  错分分布:")
            tqdm.write(f"    - Ra0.05（源额外类）: {mis_distribution.get(0, 0)}个")
            tqdm.write(f"    - Ra1.6（源额外类）: {mis_distribution.get(5, 0)}个")
        tqdm.write("\n混淆矩阵（6类全量）:")
        tqdm.write(np.array2string(cm, formatter={'int': lambda x: f"{x:4d}"}))

        try:
            model_for_stats = model.module if isinstance(model, nn.DataParallel) else model
            confusion_pairs = model_for_stats.classifier[-1].confusion_pairs
            temp_stats = model_for_stats.get_temperature_stats()
            tqdm.write(f"\n混淆对统计: {list(confusion_pairs.keys())}")
            tqdm.write(f"温度统计: {temp_stats}")
        except AttributeError as e:
            tqdm.write(f"⚠️ 无法获取混淆对/温度统计：{str(e)}")
        tqdm.write("=" * 60 + "\n")

        generate_complete_report(
            total_time=total_time, epochs_trained=epochs_trained, best_epoch=best_epoch,
            train_metrics=train_metrics, val_metrics=val_metrics, test_metrics=test_metrics,
            best_lr=train_metrics['lrs'][best_epoch], lr_adjust_records=lr_adjust_records,
            save_path=COMPLETE_REPORT_PATH, stage_name=stage_name, lr_config=lr_config,
            best_val_report=best_val_results['report'], best_val_cm=best_val_results['cm'],
            best_val_y_true=best_val_results['y_true'], best_val_y_pred=best_val_results['y_pred'],
            class_names=class_names
        )
        tqdm.write(f"✅ {stage_name} - 综合报告已保存至: {COMPLETE_REPORT_PATH}")

        test_report_path = os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_test_report.txt")
        with open(test_report_path, "w", encoding="utf-8") as f:
            f.write(f"{stage_name} - 测试集分类报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试集全量准确率: {acc:.2f}%\n")
            f.write(f"测试集有效类准确率: {valid_acc:.2f}%\n")
            f.write(f"测试集MAP: {map_score:.2f}%\n")
            if is_target_stage:
                f.write(f"跨数据集错分率: {cross_mis_rate:.2f}%\n")
                f.write(f"跨数据集错分分布: {mis_distribution}\n\n")
            else:
                f.write("\n")
            try:
                f.write(                f"混淆对统计: {list(confusion_pairs.keys())}\n")
                f.write(f"温度统计: {temp_stats}\n\n")
            except:
                f.write("混淆对/温度统计：无法获取\n\n")
            f.write("6类全量分类报告:\n")
            f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0, labels=range(num_classes)))
            if is_target_stage:
                target_mask = np.isin(y_true, TARGET_LABELS)
                y_true_target = np.array(y_true)[target_mask]
                y_pred_target = np.array(y_pred)[target_mask]
                if len(y_true_target) > 0:
                    f.write("\n\n目标4类专项分类报告（排除干扰类）:\n")
                    f.write(classification_report(
                        y_true_target, y_pred_target,
                        target_names=TARGET_CLASS_NAMES,
                        labels=TARGET_LABELS,
                        zero_division=0
                    ))
            f.write("\n\n混淆矩阵（6类全量）:\n")
            f.write(np.array2string(cm, formatter={'int': lambda x: f"{x:4d}"}))

        tqdm.write(f"\n📊 {stage_name} - 生成可视化图表...")

        plot_metric(train_metrics['lrs'], f"{stage_name} - Learning Rate Change", "Learning Rate",
                    os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_learning_rate.png"))
        plot_metric(train_metrics['losses'], f"{stage_name} - Train Loss Curve", "Loss",
                    os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_train_loss.png"))
        plot_metric(val_metrics['losses'], f"{stage_name} - Validation Loss Curve", "Loss",
                    os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_val_loss.png"))
        plot_metric(train_metrics['accs'], f"{stage_name} - Train Accuracy Curve", "Accuracy (%)",
                    os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_train_acc.png"))
        plot_metric(val_metrics['accs'], f"{stage_name} - Validation Total Accuracy Curve", "Accuracy (%)",
                    os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_val_total_acc.png"))
        plot_metric(val_metrics['valid_accs'], f"{stage_name} - Validation Valid Accuracy Curve", "Valid Accuracy (%)",
                    os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_val_valid_acc.png"))
        plot_metric(val_metrics['maps'], f"{stage_name} - Validation MAP Curve", "MAP (%)",
                    os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_val_map.png"))
        if is_target_stage:
            plot_metric(val_metrics['cross_mis_rates'],
                        f"{stage_name} - Validation Cross-Dataset Misclassification Rate", "Misclassification Rate (%)",
                        os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_val_cross_mis_rate.png"))

        plot_compare_metric(
            train_metrics['losses'], val_metrics['losses'],
            f"{stage_name} - Train vs Validation Loss", "Loss",
            os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_train_val_loss_compare.png")
        )
        plot_compare_metric(
            train_metrics['accs'], val_metrics['valid_accs'],
            f"{stage_name} - Train Accuracy vs Validation Valid Accuracy", "Accuracy (%)",
            os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_train_val_valid_acc_compare.png")
        )
        plot_compare_metric(
            val_metrics['accs'], val_metrics['valid_accs'],
            f"{stage_name} - Validation Total vs Valid Accuracy", "Accuracy (%)",
            os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_val_total_vs_valid_acc.png")
        )
        plot_compare_metric(
            val_metrics['valid_accs'], val_metrics['maps'],
            f"{stage_name} - Validation Valid Accuracy vs MAP", "Score (%)",
            os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_val_valid_acc_vs_map.png")
        )

        plt.figure(figsize=(10, 8))
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm. Blues)
        plt.title(f"{stage_name} - Confusion Matrix (6 Classes Full)", fontsize=12, fontweight='bold')
        plt.colorbar(im, shrink=0.8, label="Sample Count")
        plt.xticks(ticks=range(num_classes), labels=class_names, rotation=45, ha='right')
        plt.yticks(ticks=range(num_classes), labels=class_names)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=9, fontweight='bold')
        if is_target_stage:
            target_indices = TARGET_LABELS
            plt.hlines(y=[min(target_indices)-0.5, max(target_indices)+0.5], xmin=0.5, xmax=num_classes-0.5, colors='red', linestyles='--', linewidth=2)
            plt.vlines(x=[min(target_indices)-0.5, max(target_indices)+0.5], ymin=0.5, ymax=num_classes-0.5, colors='red', linestyles='--', linewidth=2)
            plt.xlabel("Predicted Label (Including Disturb Classes: Ra0.05/Ra1.6)", fontsize=10)
            plt.ylabel("True Label (Target 4 Classes)", fontsize=10)
        else:
            plt.xlabel("Predicted Label", fontsize=10)
            plt.ylabel("True Label", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_test_confusion_matrix_6class.png"), dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        if best_val_results['cm'] is not None:
            plt.figure(figsize=(10, 8))
            im = plt.imshow(best_val_results['cm'], interpolation='nearest', cmap=plt.cm.Greens)
            plt.title(f"{stage_name} - Confusion Matrix (Best Validation Epoch {best_epoch + 1}, 6 Classes Full)", fontsize=12, fontweight='bold')
            plt.colorbar(im, shrink=0.8, label="Sample Count")
            plt.xticks(ticks=range(num_classes), labels=class_names, rotation=45, ha='right')
            plt.yticks(ticks=range(num_classes), labels=class_names)
            thresh = best_val_results['cm'].max() / 2.
            for i in range(best_val_results['cm'].shape[0]):
                for j in range(best_val_results['cm'].shape[1]):
                    plt.text(j, i, format(best_val_results['cm'][i, j], 'd'),
                             ha="center", va="center",
                             color="white" if best_val_results['cm'][i, j] > thresh else "black",
                             fontsize=9, fontweight='bold')
            if is_target_stage:
                target_indices = TARGET_LABELS
                plt.hlines(y=[min(target_indices)-0.5, max(target_indices)+0.5], xmin=0.5, xmax=num_classes-0.5, colors='red', linestyles='--', linewidth=2)
                plt.vlines(x=[min(target_indices)-0.5, max(target_indices)+0.5], ymin=0.5, ymax=num_classes-0.5, colors='red', linestyles='--', linewidth=2)
                plt.xlabel("Predicted Label (Including Disturb Classes: Ra0.05/Ra1.6)", fontsize=10)
                plt.ylabel("True Label (Target 4 Classes)", fontsize=10)
            else:
                plt.xlabel("Predicted Label", fontsize=10)
                plt.ylabel("True Label", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{stage_name.replace(' ', '_')}_best_val_confusion_matrix_6class.png"), dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            tqdm.write(f"✅ {stage_name} - 最佳验证轮次6类混淆矩阵图已保存")

        tqdm.write(f"✅ {stage_name} - 所有图表已保存至: {save_dir}")
    else:
        tqdm.write(f"\n⚠️ {stage_name} - 无有效训练轮次，未生成指标图表和报告")

    return SAVE_MODEL_PATH

if __name__ == '__main__':
    set_seed(SEED)
    print(f"🎲 全局随机数种子：{SEED}")

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        tqdm.write("🖥️ GPU配置信息：")
        tqdm.write(f"  使用设备: {device}")
        tqdm.write(f"  GPU数量: {torch.cuda.device_count()}")
        tqdm.write(f"  内存限制比例: {GPU_MEMORY_FRACTION * 100}%")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            tqdm.write(f"  GPU {i}: {props.name} (总内存: {props.total_memory / 1024 ** 3:.2f} GB)")

    total_save_dir = f"Transfer_Learning_ResNet50_GLCM_seed{SEED}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(total_save_dir, exist_ok=True)
    tqdm.write(f"📁 迁移学习实验总保存目录：{os.path.abspath(total_save_dir)}")

    with open(os.path.join(total_save_dir, "transfer_learning_config.txt"), "w", encoding="utf-8") as f:
        f.write(f"迁移学习实验启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write("📋 迁移学习整体配置:\n")
        f.write(f"  - 迁移学习模式: {'启用' if TRANSFER_LEARNING else '禁用'}\n")
        f.write(f"  - 随机数种子: {SEED}\n")
        f.write(f"  - 预训练源模型路径: {PRETRAINED_SOURCE_WEIGHT_PATH}\n")
        f.write(f"  - 源数据集路径: {SOURCE_DATA_ROOT}（仅模式禁用时使用）\n")
        f.write(f"  - 目标数据集路径: {TARGET_DATA_ROOT}\n")
        f.write(f"  - 源数据集训练轮次: {EPOCHS_SOURCE}（仅模式禁用时使用）\n")
        f.write(f"  - 目标数据集微调轮次: {EPOCHS_TARGET}\n")
        f.write(f"  - 模型输出维度: {NUM_CLASSES}类（与源数据集一致，含Ra0.05/Ra1.6）\n")
        f.write(f"  - 源数据集类别: {NUM_CLASSES}类 ({', '.join(CLASS_NAMES)})\n")
        f.write(f"  - 目标数据集类别: {NUM_TARGET_CLASSES}类 ({', '.join(TARGET_CLASS_NAMES)})\n")
        f.write(f"  - 目标→源标签映射: {TARGET_TO_SOURCE_LABEL}\n")
        f.write(f"  - 干扰类: Ra0.05(0)/Ra1.6(5)（评估时排除）\n")
        f.write(f"  - 有效类准确率: 仅统计真实标签为目标4类且预测标签非干扰类的准确率\n")
        f.write(f"  - 最佳模型更新规则: 仅有效类准确率提升时更新，MAP提升不更新\n")
        f.write(f"  - 源数据集学习率: {LR_SOURCE:.2e}（仅模式禁用时使用）\n")
        f.write(f"  - 目标数据集学习率: 分组学习率（Backbone: {LR_BACKBONE_TARGET:.2e}, 其他模块: {LR_OTHERS_TARGET:.2e}）\n")
        f.write(f"  - 模型冻结配置: {'启用' if FREEZE_LAYERS else '禁用'}（冻结backbone前{FREEZE_BACKBONE_UNTIL}个组件）\n")
        f.write(f"  - 设备: {device} (多GPU启用: {MULTI_GPU})\n")
        f.write(f"  - 批次大小: {BATCH_SIZE}\n")
        f.write("=" * 60 + "\n")

    try:
        pretrained_model_path = None

        if TRANSFER_LEARNING:
            tqdm.write("\n" + "=" * 80)
            tqdm.write("📌 迁移学习模式：启用（直接加载预训练源模型权重）")
            tqdm.write("=" * 80)

            if not os.path.exists(PRETRAINED_SOURCE_WEIGHT_PATH):
                raise FileNotFoundError(f"预训练源模型权重文件不存在：{PRETRAINED_SOURCE_WEIGHT_PATH}")
            pretrained_model_path = PRETRAINED_SOURCE_WEIGHT_PATH
            tqdm.write(f"✅ 已找到预训练源模型权重：{os.path.abspath(pretrained_model_path)}")
            tqdm.write(f"⚠️ 跳过源数据集训练阶段，直接进行目标数据集微调")
            tqdm.write(f"⚠️ 模型保持6类输出，评估时排除干扰类Ra0.05/Ra1.6")
        else:

            tqdm.write("\n" + "=" * 80)
            tqdm.write("📌 迁移学习模式：禁用（执行完整两阶段训练）")
            tqdm.write("=" * 80)

            tqdm.write("\n" + "=" * 80)
            tqdm.write("📌 第一阶段：源数据集训练（6类全量）")
            tqdm.write("=" * 80)

            source_save_dir = os.path.join(total_save_dir, "stage1_source_training")
            os.makedirs(source_save_dir, exist_ok=True)

            source_config = {
                'SAVE_MODEL_PATH': os.path.join(source_save_dir, DEFAULT_PRETRAINED_NAME),
                'CSV_PATH': os.path.join(source_save_dir, "source_metrics_log.csv"),
                'COMPLETE_REPORT_PATH': os.path.join(source_save_dir, "source_complete_report.txt"),
                'STOP_LOG_PATH': os.path.join(source_save_dir, "source_stop_reason.txt"),
                'ERROR_LOG_PATH': os.path.join(source_save_dir, "source_error_log.txt")
            }

            pretrained_model_path = train_stage(
                save_dir=source_save_dir,
                config=source_config,
                data_dirs={'train': SOURCE_TRAIN_DIR, 'val': SOURCE_VAL_DIR, 'test': SOURCE_TEST_DIR},
                epochs=EPOCHS_SOURCE,
                lr=LR_SOURCE,
                stage_name="源数据集训练",
                class_names=CLASS_NAMES,
                num_classes=NUM_CLASSES,
                is_target_stage=False
            )

            tqdm.write(f"\n✅ 第一阶段完成！预训练模型已保存至：{os.path.abspath(pretrained_model_path)}")

        tqdm.write("\n" + "=" * 80)
        tqdm.write("📌 第二阶段：目标数据集微调（排除干扰类Ra0.05/Ra1.6）")
        tqdm.write("=" * 80)

        target_save_dir = os.path.join(total_save_dir, "stage2_target_finetuning")
        os.makedirs(target_save_dir, exist_ok=True)

        target_config = {
            'SAVE_MODEL_PATH': os.path.join(target_save_dir, "target_best_model.pth"),
            'CSV_PATH': os.path.join(target_save_dir, "target_metrics_log.csv"),
            'COMPLETE_REPORT_PATH': os.path.join(target_save_dir, "target_complete_report.txt"),
            'STOP_LOG_PATH': os.path.join(target_save_dir, "target_stop_reason.txt"),
            'ERROR_LOG_PATH': os.path.join(target_save_dir, "target_error_log.txt")
        }

        target_best_model_path = train_stage(
            save_dir=target_save_dir,
            config=target_config,
            data_dirs={'train': TARGET_TRAIN_DIR, 'val': TARGET_VAL_DIR, 'test': TARGET_TEST_DIR},
            epochs=EPOCHS_TARGET,
            lr=LR_OTHERS_TARGET,
            stage_name="目标数据集微调",
            class_names=CLASS_NAMES,
            num_classes=NUM_CLASSES,
            pretrained_path=pretrained_model_path,
            is_target_stage=True
        )

        tqdm.write(f"\n✅ 第二阶段完成！目标数据集最佳微调模型已保存至：{os.path.abspath(target_best_model_path)}")

        with open(os.path.join(total_save_dir, "TRANSFER_LEARNING_SUMMARY_REPORT.txt"), "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("🚀 迁移学习实验（ResNet50+GLCM双分支）整体总结报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"实验总目录: {os.path.abspath(total_save_dir)}\n")
            f.write("=" * 80 + "\n\n")

            f.write("📋 实验核心配置回顾:\n")
            f.write(f"  - 模型结构: ResNet50+GLCM双分支，输出6类（与源数据集一致）\n")
            f.write(f"  - 随机数种子: {SEED}\n")
            f.write(f"  - 迁移模式: {'直接加载预训练权重' if TRANSFER_LEARNING else '完整两阶段训练'}\n")
            f.write(f"  - 目标任务: {NUM_TARGET_CLASSES}类粗糙度分类（{', '.join(TARGET_CLASS_NAMES)}）\n")
            f.write(f"  - 干扰类处理: 排除Ra0.05(0)/Ra1.6(5)，仅计算有效类准确率\n")
            f.write(f"  - 最佳模型更新: 仅有效类准确率提升时更新，MAP提升不更新\n")
            f.write(f"  - 冻结配置: 冻结backbone前{FREEZE_BACKBONE_UNTIL}个组件\n")
            f.write(f"  - 分组学习率: Backbone={LR_BACKBONE_TARGET:.2e}, 其他模块={LR_OTHERS_TARGET:.2e}\n\n")

            f.write("📊 关键结果汇总:\n")
            if not TRANSFER_LEARNING:
                f.write(f"  1. 源数据集训练模型: {os.path.basename(pretrained_model_path)}\n")
            f.write(f"  2. 目标数据集微调模型: {os.path.basename(target_best_model_path)}\n")
            f.write(f"  3. 详细指标查看: \n")
            f.write(f"     - 目标数据集训练日志: {os.path.join(target_save_dir, 'target_metrics_log.csv')}\n")
            f.write(f"     - 目标数据集完整报告: {os.path.join(target_save_dir, 'target_complete_report.txt')}\n")
            f.write(f"     - 有效类准确率曲线: {os.path.join(target_save_dir, '目标数据集微调_val_valid_acc.png')}\n")
            f.write(f"     - 6类混淆矩阵图: {os.path.join(target_save_dir, '目标数据集微调_test_confusion_matrix_6class.png')}\n")
            f.write(f"     - 跨数据集错分率曲线: {os.path.join(target_save_dir, '目标数据集微调_val_cross_mis_rate.png')}\n\n")

            f.write("🔍 模型性能验证重点:\n")
            f.write(f"  - 核心指标: 目标4类的有效类准确率、F1-score（参考专项报告）\n")
            f.write(f"  - 跨数据集错分率: 越低越好，反映模型对目标类别的区分能力\n")
            f.write(f"  - 混淆矩阵: 关注目标样本是否集中在对应类别，是否存在大量错分到Ra0.05/Ra1.6的情况\n\n")

            f.write("✅ 实验结论:\n")
            f.write(f"  1. 迁移学习流程完成，模型成功适配目标数据集{NUM_TARGET_CLASSES}类分类任务\n")
            f.write(f"  2. 启用干扰类排除后，有效类准确率更能反映模型在目标任务上的实际性能\n")
            f.write(f"  3. 最佳模型仅基于有效类准确率更新，确保了模型优化方向与目标任务一致\n")
            f.write(f"  4. 若跨数据集错分率较高（>5%），建议调整：\n")
            f.write(f"     - 减少冻结层数（如FREEZE_BACKBONE_UNTIL=6），让模型学习更多目标专属特征\n")
            f.write(f"     - 增大数据增强强度，提升模型泛化能力\n")
            f.write(f"     - 调整分类头温度参数，优化难分样本区分效果\n\n")

            f.write("📝 后续优化建议:\n")
            f.write(f"  - 对比不同冻结层数（4/6/8）的微调效果，选择最优配置\n")
            f.write(f"  - 分析错分到Ra0.05/Ra1.6的样本特征，补充针对性数据增强\n")
            f.write(f"  - 尝试融合更多纹理特征（如LBP），进一步提升分类边界清晰度\n")
            f.write(f"  - 对比不同评估指标（全量准确率vs有效类准确率）对模型性能的影响\n")
            f.write("=" * 80 + "\n")

        tqdm.write("\n" + "=" * 80)
        tqdm.write("🎉 迁移学习实验全流程执行完毕！")
        tqdm.write(f"📁 所有实验结果已保存至：{os.path.abspath(total_save_dir)}")
        tqdm.write(f"📋 整体总结报告：{os.path.join(total_save_dir, 'TRANSFER_LEARNING_SUMMARY_REPORT.txt')}")
        tqdm.write("=" * 80 + "\n")

    except Exception as e:

        error_msg = f"迁移学习实验执行失败：{str(e)}"
        tqdm.write(f"\n❌ {error_msg}")

        error_log_path = os.path.join(total_save_dir, "GLOBAL_EXCEPTION_LOG.txt") if 'total_save_dir' in locals() else "GLOBAL_EXCEPTION_LOG.txt"
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"错误发生时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"错误类型: {type(e).__name__}\n")
            f.write(f"错误信息: {str(e)}\n\n")
            import traceback
            f.write("详细堆栈信息:\n")
            f.write(traceback.format_exc())

        raise RuntimeError(error_msg) from e