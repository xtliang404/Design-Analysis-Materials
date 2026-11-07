import numpy as np

# 高斯模糊判断矩阵 (μ, σ)
fuzzy_matrix = np.array([
    [(0.5, 0), (0.8, 0), (0.9, 0)],
    [(0.2, 0), (0.5, 0), (0.6, 0.05)],
    [(0.1, 0), (0.4, 0.05), (0.5, 0)]
])


# 计算权重向量（几何均值法）
def compute_weights(fuzzy_matrix):
    n = fuzzy_matrix.shape[0]
    geometric_means = np.array([
        np.exp(np.mean(np.log([m[0] for m in row]))) for row in fuzzy_matrix
    ])
    W = geometric_means / np.sum(geometric_means)  # 归一化
    return W


# 计算 λ_max
def compute_lambda_max(fuzzy_matrix, W):
    n = fuzzy_matrix.shape[0]

    # 提取模糊数的均值（μ值）作为判断矩阵
    crisp_matrix = np.array([[fuzzy_matrix[i][j][0] for j in range(n)] for i in range(n)])

    # 计算最大特征值
    eigenvalues, _ = np.linalg.eig(crisp_matrix)
    lambda_max = max(eigenvalues.real)  # 取实部最大特征值

    return lambda_max


# 计算 CI 和 CR
def compute_CI_CR(fuzzy_matrix, W):
    n = fuzzy_matrix.shape[0]
    lambda_max = compute_lambda_max(fuzzy_matrix, W)

    # 确保 lambda_max >= n
    if lambda_max < n:
        print("⚠ 警告: 计算的 λ_max 小于矩阵阶数 n，可能有计算错误")
        lambda_max = n  # 直接调整，避免 CI 计算错误

    CI = (lambda_max - n) / (n - 1)

    # Saaty 1980 版 RI 值
    RI_values = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    RI = RI_values.get(n, 1.45)

    CR = CI / RI if RI != 0 else 0
    return CI, CR


# 计算相容性指标 I_sigma
def compute_I_sigma(fuzzy_matrix, k=0.2):
    n = fuzzy_matrix.shape[0]
    I_sigma = 0

    for i in range(n):
        for j in range(n):
            mu_ij, sigma_ij = fuzzy_matrix[i][j]
            mu_ji, sigma_ji = fuzzy_matrix[j][i]
            I_sigma += abs(mu_ij + mu_ji - 1) + k * (sigma_ij + sigma_ji)

    I_sigma /= n ** 2  # 归一化
    return I_sigma


# 计算权重
W = compute_weights(fuzzy_matrix)
CI, CR = compute_CI_CR(fuzzy_matrix, W)
I_sigma = compute_I_sigma(fuzzy_matrix, k=0.2)  # 这里 k=0.2

print("权重向量:", W)
print("一致性指标 CI:", CI)
print("一致性比率 CR:", CR)
print("相容性指标 I_sigma:", I_sigma)
