import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 新增：导入进度条
from recommenders import itemcf_recommend, content_recommend, hybrid_recommend

def evaluate_model(model, rating_matrix, sim_df1, sim_df2=None, alpha=0.7, k=10, test=None, sample_users=None):
    """
    评估模型，返回平均Precision@K、Recall@K、F1@K
    :param sample_users: 采样用户数（None表示用所有用户）
    """
    precision_list = []
    recall_list = []
    f1_list = []

    # 【优化1】采样测试用户，减少计算量
    test_users = test["user_id"].unique()
    if sample_users is not None and len(test_users) > sample_users:
        np.random.seed(42)
        test_users = np.random.choice(test_users, size=sample_users, replace=False)

    # 【优化2】加入进度条
    for user_id in tqdm(test_users, desc=f"评估 {model} 模型"):
        if user_id not in rating_matrix.index:
            continue
        user_test = test[test["user_id"] == user_id]
        liked_movies = user_test[user_test["rating"] >= 4]["movie_id"].tolist()
        if len(liked_movies) == 0:
            continue

        if model == "itemcf":
            rec_movies, _ = itemcf_recommend(user_id, rating_matrix, sim_df1, k=k)
        elif model == "content":
            rec_movies, _ = content_recommend(user_id, rating_matrix, sim_df1, k=k)
        elif model == "hybrid":
            rec_movies, _ = hybrid_recommend(user_id, rating_matrix, sim_df1, sim_df2, alpha=alpha, k=k)
        else:
            raise ValueError("model must be itemcf/content/hybrid")

        tp = len(set(rec_movies) & set(liked_movies))
        precision = tp / k if k > 0 else 0
        recall = tp / len(liked_movies) if len(liked_movies) > 0 else 0
        # 计算F1分数（避免除零）
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

def grid_search_alpha(rating_matrix, item_sim_df, content_sim_df, test, alphas=[0.0,0.2,0.4,0.6,0.8,0.95,1.0], k=10, sample_users=100):
    """网格搜索最优融合权重α（扩展搜索范围）"""
    best_alpha = 0
    best_f1 = 0  # 改用F1作为最优指标
    results = []

    for a in tqdm(alphas, desc="网格搜索 α"):
        p, r, f1 = evaluate_model("hybrid", rating_matrix, item_sim_df, content_sim_df, alpha=a, k=k, test=test, sample_users=sample_users)
        results.append((a, p, r, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = a

    # 可视化（新增F1曲线）
    a_list = [x[0] for x in results]
    p_list = [x[1] for x in results]
    r_list = [x[2] for x in results]
    f1_list = [x[3] for x in results]
    
    plt.figure(figsize=(12,6))
    plt.plot(a_list, p_list, marker="o", label=f"Precision@{k}")
    plt.plot(a_list, r_list, marker="s", label=f"Recall@{k}")
    plt.plot(a_list, f1_list, marker="^", label=f"F1@{k}")  # 新增F1曲线
    plt.xlabel("Alpha (ItemCF Weight)")
    plt.ylabel("Metric")
    plt.title("Alpha vs Recommendation Metrics (Extended Range)")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/alpha_tuning_extended.png")
    plt.close()

    return best_alpha, best_f1, results