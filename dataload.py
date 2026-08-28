import os
import cv2
import torch
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.feature import graycomatrix, graycoprops
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.feature import local_binary_pattern
import warnings
import multiprocessing as mp

warnings.filterwarnings('ignore')

# ===================== 核心配置 =====================
# 目标数据集类别（4类）如果有迁移学习任务再使用
TARGET_CLASS_NAMES = ['Ra0.1', 'Ra0.2', 'Ra0.4', 'Ra0.8']
# 源数据集类别（6类）
SOURCE_CLASS_NAMES = ['Ra0.05', 'Ra0.1', 'Ra0.2', 'Ra0.4', 'Ra0.8', 'Ra1.6']

# 【默认缓存根路径】（新增核心配置）
DEFAULT_GLCM_CACHE_ROOT = r"E:\A_cucaodu-python\ResNet50-NEW250904\迁移学习网络搭建\迁移学习2.0\缓存文件\glcm"

# 基于默认根路径自动拼接子路径（无需手动写完整路径）
SOURCE_GLCM_CACHE_ROOT = os.path.join(DEFAULT_GLCM_CACHE_ROOT, "source_dataset_cache008")
TARGET_GLCM_CACHE_ROOT = os.path.join(DEFAULT_GLCM_CACHE_ROOT, "target_dataset_cache009")

# 并行配置
MAX_WORKERS = 12

# 灰度图加权三通道系数（R/G/B）
GRAY_CHANNEL_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# GLCM参数
GLCM_PARAMS = {
    'distances': [1, 2, 3, 4],
    'angles': [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4],
    'levels': 256,
    'symmetric': True,
    'normed': True
}
GLCM_FEATURE_TYPES = ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity', 'ASM', 'entropy', 'de']

# ===================== 配置输出 =====================
if mp.current_process().name == "MainProcess":
    print(f"📌 GLCM配置：8个方向 × 4个距离 × 8种特征 = 256维特征")
    print(f"📌 GLCM方向：{[round(angle * 180 / np.pi) for angle in GLCM_PARAMS['angles']]}°")
    print(f"📌 GLCM距离：{GLCM_PARAMS['distances']}像素")
    print(f"📌 并行配置：{MAX_WORKERS}个进程同时处理")
    print(f"📌 默认缓存根路径：{DEFAULT_GLCM_CACHE_ROOT}")
    print(f"📌 源数据集（6类）缓存目录：{SOURCE_GLCM_CACHE_ROOT}，类别：{SOURCE_CLASS_NAMES}")
    print(f"📌 目标数据集（4类）缓存目录：{TARGET_GLCM_CACHE_ROOT}，类别：{TARGET_CLASS_NAMES}")
    print(f"✅ 自动识别+强制指定双保险，缓存路径精准切换！")


# ===================== 自动识别数据集类型 =====================
def auto_detect_dataset_type(data_dir):
    data_dir = os.path.normpath(data_dir)
    sub_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    sub_dirs_set = set(sub_dirs)

    if sub_dirs_set == set(TARGET_CLASS_NAMES):
        return 'target'
    elif 'Ra0.05' in sub_dirs or 'Ra1.6' in sub_dirs:
        return 'source'
    elif sub_dirs_set.issubset(set(TARGET_CLASS_NAMES)):
        return 'target'
    else:
        raise ValueError(
            f"❌ 无法识别数据集类型！\n"
            f"  当前目录下的类别：{sub_dirs}\n"
            f"  源数据集类别：{SOURCE_CLASS_NAMES}\n"
            f"  目标数据集类别：{TARGET_CLASS_NAMES}"
        )


# ===================== 数据集配置 =====================
def get_dataset_config(dataset_type):
    if dataset_type == 'source':
        return SOURCE_CLASS_NAMES, SOURCE_GLCM_CACHE_ROOT
    elif dataset_type == 'target':
        return TARGET_CLASS_NAMES, TARGET_GLCM_CACHE_ROOT
    else:
        raise ValueError(f"❌ 不支持的数据集类型：{dataset_type}")


# ===================== GLCM预处理 =====================
def optimize_preprocess(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # 滤波和增强可根据需要开启或添加
    # gray = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=50)
    # gray = cv2.medianBlur(gray, 5)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = clahe.apply(gray)

    return gray.astype(np.uint8)


# ===================== 图像读取 =====================
def read_image_compatible(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if len(img.shape) == 2:
        # 灰度图按 R/G/B 权重构造三通道
        gray = img.astype(np.float32)
        img_rgb = np.stack(
            [gray * GRAY_CHANNEL_WEIGHTS[i] for i in range(3)],
            axis=-1
        )
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_bgr = img[:, :, :3]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_rgb


# ===================== GLCM特征提取 =====================
def compute_glcm_features(gray_img):
    gray = np.clip(gray_img, 0, 255).astype(np.uint8)
    glcm = graycomatrix(
        gray,
        distances=GLCM_PARAMS['distances'],
        angles=GLCM_PARAMS['angles'],
        levels=GLCM_PARAMS['levels'],
        symmetric=GLCM_PARAMS['symmetric'],
        normed=GLCM_PARAMS['normed']
    )

    features_list = []
    for dist_idx in range(len(GLCM_PARAMS['distances'])):
        for angle_idx in range(len(GLCM_PARAMS['angles'])):
            p = glcm[:, :, dist_idx, angle_idx]
            contrast = graycoprops(glcm, 'contrast')[dist_idx, angle_idx]
            correlation = graycoprops(glcm, 'correlation')[dist_idx, angle_idx]
            energy = graycoprops(glcm, 'energy')[dist_idx, angle_idx]
            homogeneity = graycoprops(glcm, 'homogeneity')[dist_idx, angle_idx]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[dist_idx, angle_idx]
            asm = graycoprops(glcm, 'ASM')[dist_idx, angle_idx]
            entropy = -np.sum(p * np.log2(p + 1e-10)) if np.sum(p) > 0 else 0.0
            max_gray = p.shape[0]
            p_diff = np.zeros(max_gray)
            for k in range(max_gray):
                p_diff[k] = np.sum(p[np.abs(np.arange(max_gray)[:, None] - np.arange(max_gray)) == k])
            p_diff = p_diff / (np.sum(p_diff) + 1e-10)
            de = -np.sum(p_diff * np.log2(p_diff + 1e-10)) if np.sum(p_diff) > 0 else 0.0
            features = np.array([contrast, correlation, energy, homogeneity,
                                 dissimilarity, asm, entropy, de], dtype=np.float32)
            features_list.append(features)

    all_features = np.hstack(features_list)
    all_features = (all_features - np.mean(all_features)) / (np.std(all_features) + 1e-10)
    return all_features


# ===================== 单图像处理函数 =====================
def process_single_image(img_path):
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️  无法读取图像，跳过：{os.path.basename(img_path)}")
            return None
        gray = optimize_preprocess(img)
        glcm_feat = compute_glcm_features(gray)
        return glcm_feat
    except Exception as e:
        print(f"⚠️  处理图像出错，跳过：{os.path.basename(img_path)}，错误：{str(e)[:50]}")
        return None


# ===================== GLCM特征预计算 =====================
def precompute_glcm_features(root_dir, cache_dir, class_names):
    root_dir = os.path.normpath(root_dir)
    os.makedirs(cache_dir, exist_ok=True)
    path_to_idx = {}
    features_list = []

    print(f"\n⚠️  未找到有效缓存，启动并行预计算（{MAX_WORKERS}进程）")
    print(f"📌 预计算根目录：{root_dir}")
    print(f"📌 缓存保存目录：{cache_dir}")
    print(f"📌 当前数据集类别：{class_names}（共{len(class_names)}类）")

    all_img_paths = []
    class_img_count = {}
    for cls in class_names:
        cls_dir = os.path.normpath(os.path.join(root_dir, cls))
        if not os.path.isdir(cls_dir):
            print(f"⚠️  类别文件夹不存在，跳过：{cls_dir}")
            class_img_count[cls] = 0
            continue
        img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        class_img_count[cls] = len(img_files)
        cls_img_paths = [os.path.normpath(os.path.join(cls_dir, img_name)) for img_name in img_files]
        all_img_paths.extend(cls_img_paths)

    if 'Ra0.05' in class_img_count and class_img_count['Ra0.05'] == 0:
        print(f"⚠️  警告：Ra0.05类别文件夹存在但无有效图像！路径：{os.path.join(root_dir, 'Ra0.05')}")
    if 'Ra1.6' in class_img_count and class_img_count['Ra1.6'] == 0:
        print(f"⚠️  警告：Ra1.6类别文件夹存在但无有效图像！路径：{os.path.join(root_dir, 'Ra1.6')}")

    print(f"\n✅ 共收集 {len(all_img_paths)} 张图像，启动并行处理...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_img = {executor.submit(process_single_image, img_path): img_path for img_path in all_img_paths}
        for idx, future in enumerate(as_completed(future_to_img), 1):
            glcm_feat = future.result()
            img_path = future_to_img[future]
            if glcm_feat is not None:
                features_list.append(glcm_feat)
                path_to_idx[img_path] = len(features_list) - 1
            if len(features_list) % 100 == 0:
                print(f"✅ 已处理 {len(features_list)}/{len(all_img_paths)} 张图像")

    np.save(os.path.join(cache_dir, "features.npy"), np.array(features_list))
    with open(os.path.join(cache_dir, "path_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(path_to_idx, f, indent=2, ensure_ascii=False)

    print(f"\n✅ GLCM特征预计算完成！")
    print(f"📌 缓存保存路径：{cache_dir}")
    print(f"📌 有效样本数：{len(features_list)} 个")

    if len(features_list) == 0:
        raise RuntimeError(
            "GLCM特征预计算失败：0 个有效样本。请检查前面的图像处理错误。"
        )

    print(f"📌 特征维度：{features_list[0].shape[0]} 维")
    print(f"📌 每类图像统计：{class_img_count}")
    return len(features_list)


# ===================== Dataset =====================
class DualBranchDataset(Dataset):
    def __init__(self, root_dir, cache_type, transform=None, dataset_type=None,
                 is_target_dataset=False, target_to_source_label=None):
        self.root_dir = os.path.normpath(root_dir)
        self.cache_type = cache_type.lower()
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.is_target_dataset = is_target_dataset
        self.target_to_source_label = target_to_source_label or {}

        if dataset_type is None:
            self.dataset_type = auto_detect_dataset_type(self.root_dir)
        else:
            self.dataset_type = dataset_type
            print(f"🔒 强制指定数据集类型：{self.dataset_type}（跳过自动识别）")

        self.class_names, self.cache_root = get_dataset_config(self.dataset_type)
        self.cache_dir = os.path.normpath(os.path.join(self.cache_root, self.cache_type))

        print(f"📌 当前数据集类型：{self.dataset_type}")
        print(f"📌 当前数据集配置：{len(self.class_names)}类 → {self.class_names}")
        print(f"📌 标签映射状态：{'启用' if self.is_target_dataset and self.target_to_source_label else '禁用'}")
        if self.is_target_dataset and self.target_to_source_label:
            print(f"📌 标签映射规则：{self.target_to_source_label}")
        print(f"📌 当前缓存读写路径：{self.cache_dir}")

        self._collect_image_paths()
        self._load_or_rebuild_cache()

    def _collect_image_paths(self):
        for cls in self.class_names:
            cls_dir = os.path.normpath(os.path.join(self.root_dir, cls))
            if not os.path.isdir(cls_dir):
                print(f"⚠️  类别文件夹不存在，跳过：{cls_dir}")
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    img_path = os.path.normpath(os.path.join(cls_dir, img_name))
                    self.image_paths.append(img_path)

                    if self.is_target_dataset and self.target_to_source_label:
                        mapped_label = self.target_to_source_label.get(cls, -1)
                        if mapped_label == -1:
                            raise ValueError(f"❌ 类别 {cls} 无对应映射规则！映射字典：{self.target_to_source_label}")
                        self.labels.append(mapped_label)
                    else:
                        self.labels.append(self.class_names.index(cls))

        if len(self.image_paths) == 0:
            raise ValueError(
                f"❌ 未找到任何图像！根目录：{self.root_dir}，支持格式：jpg/png/jpeg/bmp，类别：{self.class_names}")

        print(f"✅ 数据集扫描完成：{self.cache_type}集共 {len(self.image_paths)} 个样本")
        label_count = {}
        if self.is_target_dataset and self.target_to_source_label:
            reverse_map = {v: k for k, v in self.target_to_source_label.items()}
            for label in set(self.labels):
                count = self.labels.count(label)
                label_count[f"{reverse_map.get(label, '未知')}→{label}"] = count
        else:
            for cls in self.class_names:
                count = self.labels.count(self.class_names.index(cls))
                label_count[cls] = count
        print(f"📌 标签分布（映射后）：{label_count}")

    def _load_or_rebuild_cache(self):
        print(f"\n📌 检查{self.cache_type}集GLCM缓存：{self.cache_dir}")
        try:
            self._load_existing_cached_features()
        except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as e:
            print(f"⚠️  缓存加载失败：{str(e)}")
            print(f"📌 自动重建{self.cache_type}集GLCM缓存...")
            precompute_glcm_features(
                root_dir=self.root_dir,
                cache_dir=self.cache_dir,
                class_names=self.class_names
            )
            self._load_existing_cached_features()

    def _load_existing_cached_features(self):
        features_path = os.path.join(self.cache_dir, "features.npy")
        mapping_path = os.path.join(self.cache_dir, "path_to_idx.json")

        if not os.path.exists(features_path) or not os.path.exists(mapping_path):
            raise FileNotFoundError(f"缓存文件缺失！路径：{features_path} 或 {mapping_path}")

        self.features = np.load(features_path)
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.path_to_idx = json.load(f)

        self.path_to_idx = {os.path.normpath(k): v for k, v in self.path_to_idx.items()}

        missing_imgs = []
        for img_path in self.image_paths:
            norm_img_path = os.path.normpath(img_path)
            if norm_img_path not in self.path_to_idx:
                missing_imgs.append(img_path)

        if missing_imgs:
            raise ValueError(f"{len(missing_imgs)} 个样本未找到对应特征（数据集与缓存不匹配）")

        if self.features.shape[1] != 256:
            raise ValueError(f"特征维度不匹配（当前 {self.features.shape[1]} 维，期望 256 维）")

        print(f"✅ {self.cache_type}集缓存加载成功！")
        print(f"  - 特征维度：{self.features.shape[1]} 维")
        print(f"  - 匹配样本数：{len(self.image_paths)} 个")
        print(f"  - 缓存读取路径：{self.cache_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        norm_img_path = os.path.normpath(img_path)

        img = read_image_compatible(img_path)
        if img is None:
            raise ValueError(f"❌ 无法读取图像：{img_path}")

        if self.transform:
            img = self.transform(img)

        feat_idx = self.path_to_idx[norm_img_path]
        glcm_features = self.features[feat_idx]
        glcm_tensor = torch.tensor(glcm_features, dtype=torch.float32)

        return img, glcm_tensor, label


# ===================== 数据加载 =====================
def load_data_split(train_dir, val_dir, test_dir, batch_size=8, num_workers=8, force_dataset_type=None,
                    is_target_dataset=False, target_to_source_label=None):
    if force_dataset_type is not None:
        dataset_type = force_dataset_type
        print(f"🔒 全局强制指定数据集类型：{dataset_type}")
    else:
        dataset_type = auto_detect_dataset_type(train_dir)

    _, cache_root = get_dataset_config(dataset_type)
    dataset_name = "源数据集（6类）" if dataset_type == "source" else "目标数据集（4类）"

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("=" * 80)
    print(f"📊 开始加载双分支粗糙度数据集（{dataset_name}）")
    print(f"📌 图像配置：输入尺寸(512,512)，ImageNet标准化")
    print(f"📌 GLCM配置：256维特征（8方向×4距离×8特征）")
    print(f"📌 缓存根路径：{cache_root}")
    print(f"📌 批次大小：{batch_size}，工作线程数：{num_workers}")
    print(f"📌 标签映射状态：{'启用' if is_target_dataset and target_to_source_label else '禁用'}")
    if is_target_dataset and target_to_source_label:
        print(f"📌 全局标签映射规则：{target_to_source_label}")
    print("=" * 80)

    print("\n[训练集]")
    train_set = DualBranchDataset(
        train_dir, cache_type='train', transform=transform, dataset_type=dataset_type,
        is_target_dataset=is_target_dataset, target_to_source_label=target_to_source_label
    )
    print("\n[验证集]")
    val_set = DualBranchDataset(
        val_dir, cache_type='val', transform=transform, dataset_type=dataset_type,
        is_target_dataset=is_target_dataset, target_to_source_label=target_to_source_label
    )
    print("\n[测试集]")
    test_set = DualBranchDataset(
        test_dir, cache_type='test', transform=transform, dataset_type=dataset_type,
        is_target_dataset=is_target_dataset, target_to_source_label=target_to_source_label
    )

    def create_dataloader(dataset, shuffle, batch_size, num_workers):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=min(num_workers, os.cpu_count() - 1),
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

    train_loader = create_dataloader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = create_dataloader(val_set, shuffle=False, batch_size=batch_size , num_workers=num_workers // 2)
    test_loader = create_dataloader(test_set, shuffle=False, batch_size=batch_size , num_workers=num_workers // 2)

    print("\n" + "=" * 80)
    print("✅ 所有数据集加载完成！")
    print(f"📌 数据集类型：{dataset_name}")
    print(f"📌 类别配置（映射后）：{len(SOURCE_CLASS_NAMES)}类 → {SOURCE_CLASS_NAMES}")
    print(f"📌 缓存根路径：{cache_root}")
    print(f"📌 输出格式：(图像tensor, GLCM特征tensor, 标签)")
    print(f"📌 图像形状：(batch_size, 3, 512, 512)")
    print(f"📌 GLCM特征形状：(batch_size, 256)")
    print(f"📌 标签范围：{min(train_set.labels)}~{max(train_set.labels)}（源数据集6类索引）")
    print(f"📌 训练集：{len(train_set)} 样本，{len(train_loader)} 批次")
    print(f"📌 验证集：{len(val_set)} 样本，{len(val_loader)} 批次")
    print(f"📌 测试集：{len(test_set)} 样本，{len(test_loader)} 批次")
    print("=" * 80)
    return train_loader, val_loader, test_loader


