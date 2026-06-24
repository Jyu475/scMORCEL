

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Robust 标准化与 0-1 映射工具
# -----------------------------
def _robust_median_mad(x: np.ndarray):
    x = np.asarray(x).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return med, mad


def _robust_zscore(
    x: np.ndarray,
    median: float,
    mad: float,
    eps: float = 1e-12,
    use_consistency_const: bool = True,
):
   
    x = np.asarray(x)
    denom = (1.4826 * mad) if use_consistency_const else mad
    denom = denom + eps
    return (x - median) / denom


def _sigmoid_to_01(z: np.ndarray, clip: float = 50.0):
 
    z = np.asarray(z)
    z = np.clip(z, -clip, clip)
    return 1.0 / (1.0 + np.exp(-z))


# -----------------------------
# Mahalanobis Detector
# -----------------------------
class MahalanobisDetector:


    def __init__(self):
        self.class_means = {}
        self.precision_matrix = None
        self.num_classes = 0

    def extract_features(self, model, data_loader, device):

        model.eval()
        features_list = []
        labels_list = []

        features_hook = []

        def hook_fn(module, input, output):
            features_hook.append(output.detach())

        handle = None

        if hasattr(model, "layer3"):
            handle = model.layer3.register_forward_hook(hook_fn)
        elif hasattr(model, "elu3"):
            handle = model.elu3.register_forward_hook(hook_fn)
        else:
            for _, module in model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    handle = module.register_forward_hook(hook_fn)
                    break

        if handle is None:
            raise RuntimeError("无法为模型注册 hook（未找到合适的层），请检查模型结构。")

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                _ = model(batch_x)

                if len(features_hook) > 0:
                    feat = features_hook[-1]
                    features_list.append(feat.cpu().numpy())
                    labels_list.append(batch_y.numpy())
                    features_hook.clear()

        handle.remove()

        if len(features_list) == 0:
            raise RuntimeError("无法提取特征，请检查模型结构或数据加载器输出。")

        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        return features, labels

    def fit(self, train_loader, model, device, verbose=True):

        if verbose:
            print("\n[Mahalanobis] 提取训练集特征...")

        features, labels = self.extract_features(model, train_loader, device)

        if verbose:
            print(f"  特征维度: {features.shape}")

        self.num_classes = len(np.unique(labels))
        if verbose:
            print(f"  类别数: {self.num_classes}")

        class_features = {}
        for c in range(self.num_classes):
            mask = labels == c
            class_features[c] = features[mask]
            self.class_means[c] = class_features[c].mean(axis=0)
            if verbose:
                print(f"  类别 {c}: {class_features[c].shape[0]} 样本")

        if verbose:
            print("\n[Mahalanobis] 计算协方差矩阵...")

        centered_features = []
        for c in range(self.num_classes):
            centered = class_features[c] - self.class_means[c]
            centered_features.append(centered)
        centered_features = np.vstack(centered_features)

        cov_matrix = np.cov(centered_features.T)

        reg_coeff = 1e-4
        cov_matrix += reg_coeff * np.eye(cov_matrix.shape[0])

        try:
            self.precision_matrix = np.linalg.inv(cov_matrix)
            if verbose:
                print(f"  ✓ 协方差矩阵维度: {cov_matrix.shape}")
        except np.linalg.LinAlgError:
            if verbose:
                print("  ⚠️ 协方差矩阵奇异，使用伪逆")
            self.precision_matrix = np.linalg.pinv(cov_matrix)

    def compute_distances(self, data_loader, model, device, verbose=True):

        if verbose:
            print("\n[Mahalanobis] 计算数据集距离...")

        features, _ = self.extract_features(model, data_loader, device)

        distances = []
        for i in range(len(features)):
            feat = features[i]
            min_dist = float("inf")

            for c in range(self.num_classes):
                diff = feat - self.class_means[c]
                dist = np.sqrt(diff @ self.precision_matrix @ diff.T)
                if dist < min_dist:
                    min_dist = dist

            distances.append(min_dist)

        distances = np.array(distances)

        if verbose:
            print(f"  距离范围: [{distances.min():.4f}, {distances.max():.4f}]")
            print(f"  平均距离: {distances.mean():.4f}")

        return distances



def compute_energy_score(data_loader, model, device, temperature=1.0, verbose=True):

    if verbose:
        print("\n[Energy] 计算Energy分数...")

    model.eval()
    energy_scores = []

    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
            energy_scores.extend(energy.cpu().numpy())

    energy_scores = np.array(energy_scores)

    if verbose:
        print(f"  能量范围: [{energy_scores.min():.4f}, {energy_scores.max():.4f}]")
        print(f"  平均能量: {energy_scores.mean():.4f}")

    return energy_scores

def combined_mahalanobis_energy(
    train_loader,
    test_loader,
    model,
    device,
    alpha=0.5,
    temperature=1.0,
    verbose=True,
    eps=1e-12,
    use_consistency_const=True,
):

    if verbose:
        print("\n" + "=" * 60)
        print("组合Mahalanobis + Energy OOD检测（Robust-Z + 0-1）")
        print("=" * 60)

    mahal_detector = MahalanobisDetector()
    mahal_detector.fit(train_loader, model, device, verbose=verbose)

    mahal_train = mahal_detector.compute_distances(train_loader, model, device, verbose=verbose)
    mahal_test = mahal_detector.compute_distances(test_loader, model, device, verbose=verbose)

    energy_train = compute_energy_score(train_loader, model, device, temperature, verbose=verbose)
    energy_test = compute_energy_score(test_loader, model, device, temperature, verbose=verbose)

    m_id_train = -mahal_train
    m_id_test = -mahal_test

    e_id_train = -energy_train
    e_id_test = -energy_test

    if verbose:
        print("\n[Robust-Z] 计算训练集 median/MAD 并标准化...")

    med_m, mad_m = _robust_median_mad(m_id_train)
    med_e, mad_e = _robust_median_mad(e_id_train)

    z_m_test = _robust_zscore(
        m_id_test, med_m, mad_m, eps=eps, use_consistency_const=use_consistency_const
    )
    z_e_test = _robust_zscore(
        e_id_test, med_e, mad_e, eps=eps, use_consistency_const=use_consistency_const
    )


    m_01 = _sigmoid_to_01(z_m_test)
    e_01 = _sigmoid_to_01(z_e_test)

    z_combined = alpha * z_m_test + (1.0 - alpha) * z_e_test
    combined_score = _sigmoid_to_01(z_combined)

    if verbose:
        print(f"\n[组合] 权重: Mahalanobis={alpha:.2f}, Energy={1-alpha:.2f}")
        print(f"  Mahalanobis(z) : mean={z_m_test.mean():.4f}, std={z_m_test.std():.4f}")
        print(f"  Energy(z)      : mean={z_e_test.mean():.4f}, std={z_e_test.std():.4f}")
        print(f"  Combined(z)    : mean={z_combined.mean():.4f}, std={z_combined.std():.4f}")
        print(f"  Combined(0-1)  : min={combined_score.min():.4f}, max={combined_score.max():.4f}, mean={combined_score.mean():.4f}")
        print("=" * 60)

    components = {
        "mahalanobis": m_01,
        "energy": e_01,

        "mahalanobis_z": z_m_test,
        "energy_z": z_e_test,
        "combined_z": z_combined,

        "mahalanobis_raw": mahal_test,   
        "energy_raw": energy_test,      

        "calibration": {
            "median_mahal_id": float(med_m),
            "mad_mahal_id": float(mad_m),
            "median_energy_id": float(med_e),
            "mad_energy_id": float(mad_e),
            "use_consistency_const": bool(use_consistency_const),
            "eps": float(eps),
            "alpha": float(alpha),
            "temperature": float(temperature),
        },
    }

    return combined_score, components