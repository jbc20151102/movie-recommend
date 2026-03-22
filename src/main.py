import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_clean_data
from feature_engineering import build_feature_matrices
from similarity import compute_similarity_matrices
from evaluation import evaluate_model, grid_search_alpha

# ===================== 全局配置（缓存控制核心）=====================
# 强制重新训练（True=重新计算所有结果，False=优先加载缓存）
FORCE_RETRAIN = False
# 缓存文件夹路径
CACHE_DIR = "cache"
# 缓存文件名定义（统一管理）
CACHE_FILES = {
    "similarity": os.path.join(CACHE_DIR, "similarity_matrices.pkl"),
    "evaluation": os.path.join(CACHE_DIR, "evaluation_metrics.pkl"),
    "alpha": os.path.join(CACHE_DIR, "best_alpha.pkl"),
    "rating_matrix": os.path.join(CACHE_DIR, "rating_matrix.pkl"),
    "content_matrix": os.path.join(CACHE_DIR, "content_matrix.pkl")
}

# 设置中文字体（避免中文乱码）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ===================== 缓存工具函数（新增核心）=====================
def save_cache(data, file_path):
    """保存缓存文件（pickle格式）"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ 缓存已保存至：{file_path}")

def load_cache(file_path):
    """加载缓存文件，不存在则返回None"""
    if os.path.exists(file_path) and not FORCE_RETRAIN:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ 从缓存加载：{file_path}")
        return data
    return None

# ===================== 全局推荐核心函数（保持不变）=====================
def global_movie_recommend(ratings_df, movies_df, top_k=10, min_rating_count=10):
    """生成全局热门电影推荐榜单"""
    movie_global_metrics = ratings_df.groupby("movie_id").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()
    # 过滤冷门电影
    movie_global_metrics = movie_global_metrics[movie_global_metrics["rating_count"] >= min_rating_count]
    
    # 计算归一化热度和综合得分
    max_count = movie_global_metrics["rating_count"].max()
    movie_global_metrics["count_norm"] = movie_global_metrics["rating_count"] / max_count if max_count > 0 else 0
    movie_global_metrics["global_score"] = movie_global_metrics["avg_rating"] * 0.7 + (movie_global_metrics["count_norm"] * 5) * 0.3
    
    # 排序取Top-K并拼接电影名称
    movie_global_metrics_sorted = movie_global_metrics.sort_values("global_score", ascending=False).head(top_k)
    movie_global_metrics_sorted = pd.merge(
        movie_global_metrics_sorted,
        movies_df[["movie_id", "title"]],
        on="movie_id",
        how="left"
    )
    # 整理结果格式
    result = movie_global_metrics_sorted[["title", "avg_rating", "rating_count", "global_score"]].reset_index(drop=True)
    return result

# ===================== 可视化函数（完整保留）=====================
def plot_data_distribution(ratings, save_path):
    """Visualization 1: Basic data distribution (rating + user/movie rating count)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 评分分布直方图
    sns.histplot(ratings["rating"], bins=10, ax=axes[0], color="#1f77b4")
    axes[0].set_title("Movie Rating Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Count")
    
    # Top50用户评分数量
    user_rating_count = ratings["user_id"].value_counts().head(50)
    sns.barplot(x=user_rating_count.index, y=user_rating_count.values, ax=axes[1], color="#ff7f0e")
    axes[1].set_title("Top 50 Users - Rating Count Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("User ID")
    axes[1].set_ylabel("Rating Count")
    axes[1].tick_params(axis="x", rotation=90)
    
    # Top50电影评分数量
    movie_rating_count = ratings["movie_id"].value_counts().head(50)
    sns.barplot(x=movie_rating_count.index, y=movie_rating_count.values, ax=axes[2], color="#2ca02c")
    axes[2].set_title("Top 50 Movies - Rating Count Distribution", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Movie ID")
    axes[2].set_ylabel("Rating Count")
    axes[2].tick_params(axis="x", rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 数据分布图表已保存至：{save_path}")

def plot_global_recommendation(global_rec_result, save_path):
    """Visualization 2: Global Recommendation List (Dual-axis: Rating + Popularity)"""
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    x = range(len(global_rec_result))
    # 平均评分（柱状图）
    bars = ax1.bar(x, global_rec_result["avg_rating"], color="#1f77b4", alpha=0.7, label="Average Rating")
    ax1.set_xlabel("Movie Rank", fontsize=12)
    ax1.set_ylabel("Average Rating", fontsize=12, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(global_rec_result["title"], rotation=45, ha="right")
    
    # 评分人数（折线图，双轴）
    ax2 = ax1.twinx()
    ax2.plot(x, global_rec_result["rating_count"], color="#ff7f0e", marker="o", linewidth=2, label="Rating Count")
    ax2.set_ylabel("Rating Count", fontsize=12, color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f"{height:.2f}", ha="center", va="bottom", fontsize=10)
    
    # 标题和图例
    plt.title("Global Top 10 Recommended Movies - Avg Rating vs Rating Count", fontsize=14, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 全局推荐榜单图表已保存至：{save_path}")

def plot_model_evaluation(itemcf_metrics, content_metrics, save_path):
    """Visualization 3: Model Evaluation Metrics Comparison"""
    metrics = ["Precision", "Recall", "F1-Score"]  # 优化F1为F1-Score，符合行业规范
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, itemcf_metrics, width, label="ItemCF", color="#1f77b4")
    rects2 = ax.bar(x + width/2, content_metrics, width, label="Content-Based", color="#ff7f0e")
    
    # 样式设置
    ax.set_title("ItemCF vs Content-Based - Evaluation Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Evaluation Metrics", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # 添加数值标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.4f}",
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom")
    add_labels(rects1)
    add_labels(rects2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 模型评估对比图表已保存至：{save_path}")

def plot_rating_vs_popularity(ratings, save_path):
    """Visualization 4: Correlation between Movie Rating and Popularity"""
    # 计算每部电影的评分和热度
    movie_metrics = ratings.groupby("movie_id").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()
    # 过滤冷门电影
    movie_metrics = movie_metrics[movie_metrics["rating_count"] >= 10]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x="rating_count", y="avg_rating", data=movie_metrics, 
                    alpha=0.6, color="#2ca02c", ax=ax)
    # 添加趋势线
    sns.regplot(x="rating_count", y="avg_rating", data=movie_metrics, 
                scatter=False, color="#d62728", ax=ax)
    
    ax.set_title("Movie Average Rating vs Rating Count (Popularity)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rating Count (Popularity)", fontsize=12)
    ax.set_ylabel("Average Rating", fontsize=12)
    # 计算相关系数并标注
    corr = movie_metrics["avg_rating"].corr(movie_metrics["rating_count"])
    ax.text(0.05, 0.95, f"Correlation Coefficient: {corr:.2f}", transform=ax.transAxes, 
            fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 评分-热度相关性图表已保存至：{save_path}")

def plot_alpha_tuning(alpha_results, save_path):
    """Visualization 5: α Hyperparameter Tuning Results"""
    alphas = [item[0] for item in alpha_results]
    f1_scores = [item[1] for item in alpha_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=alphas, y=f1_scores, marker="o", linewidth=2, color="#9467bd", ax=ax)
    # 标注最优α
    best_alpha = alphas[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    ax.scatter(best_alpha, best_f1, color="#d62728", s=100, label=f"Best α={best_alpha} (F1-Score={best_f1:.4f})")
    
    ax.set_title("Hybrid Recommendation - α Tuning (F1-Score Variation)", fontsize=14, fontweight="bold")
    ax.set_xlabel("α (ItemCF Weight)", fontsize=12)
    ax.set_ylabel("F1-Score @10", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ α调优图表已保存至：{save_path}")

# ===================== 主函数（核心改造：缓存逻辑）=====================
def main():
    # ---------------------- 1. 初始化文件夹 ----------------------
    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    TOP_K = 10
    MIN_RATING_COUNT = 10

    # ---------------------- 2. 数据加载 ----------------------
    print("=== 正在加载和清洗数据... ===")
    ratings, movies, train, test = load_and_clean_data()
    movie_id_to_name = movies.set_index("movie_id")["title"].to_dict()
    
    print(f"  - 总评分样本数：{len(ratings)}，训练集：{len(train)}，测试集：{len(test)}")
    print(f"  - 总用户数：{ratings['user_id'].nunique()}，总电影数：{ratings['movie_id'].nunique()}")

    # ---------------------- 3. 特征矩阵（优先加载缓存）----------------------
    print("\n=== 正在构建/加载特征矩阵... ===")
    # 加载评分矩阵缓存
    rating_matrix = load_cache(CACHE_FILES["rating_matrix"])
    content_matrix = load_cache(CACHE_FILES["content_matrix"])
    genre_cols = None
    
    if rating_matrix is None or content_matrix is None:
        # 缓存不存在则重新计算
        rating_matrix, content_matrix, genre_cols = build_feature_matrices(train, movies)
        # 保存缓存
        save_cache(rating_matrix, CACHE_FILES["rating_matrix"])
        save_cache(content_matrix, CACHE_FILES["content_matrix"])
    
    print(f"  - 评分矩阵形状：{rating_matrix.shape}，内容特征矩阵形状：{content_matrix.shape}")
    # 保存样本到reports
    rating_matrix.iloc[:10, :10].to_csv("reports/rating_matrix_sample.csv")
    content_matrix.iloc[:10, :10].to_csv("reports/content_matrix_sample.csv")

    # ---------------------- 4. 相似度矩阵（优先加载缓存）----------------------
    print("\n=== 正在计算/加载相似度矩阵... ===")
    sim_data = load_cache(CACHE_FILES["similarity"])
    
    if sim_data is None:
        item_sim_df, content_sim_df = compute_similarity_matrices(rating_matrix, content_matrix)
        sim_data = (item_sim_df, content_sim_df)
        save_cache(sim_data, CACHE_FILES["similarity"])
    else:
        item_sim_df, content_sim_df = sim_data
    
    print("  - ItemCF 相似度矩阵示例（前 3 部电影）：")
    print(item_sim_df.iloc[:3, :3])

    # ---------------------- 5. 模型评估（优先加载缓存）----------------------
    print("\n=== 正在评估/加载模型评估指标... ===")
    eval_data = load_cache(CACHE_FILES["evaluation"])
    
    if eval_data is None:
        itemcf_p, itemcf_r, itemcf_f1 = evaluate_model("itemcf", rating_matrix, item_sim_df, test=test, k=TOP_K, sample_users=100)
        content_p, content_r, content_f1 = evaluate_model("content", rating_matrix, content_sim_df, test=test, k=TOP_K, sample_users=100)
        eval_data = (itemcf_p, itemcf_r, itemcf_f1, content_p, content_r, content_f1)
        # 保存缓存
        save_cache(eval_data, CACHE_FILES["evaluation"])
    else:
        itemcf_p, itemcf_r, itemcf_f1, content_p, content_r, content_f1 = eval_data
    
    # 整理评估指标
    itemcf_metrics = [itemcf_p, itemcf_r, itemcf_f1]
    content_metrics = [content_p, content_r, content_f1]
    print(f"ItemCF: Precision @{TOP_K}={itemcf_p:.4f}, Recall @{TOP_K}={itemcf_r:.4f}, F1 @{TOP_K}={itemcf_f1:.4f}")
    print(f"内容推荐: Precision @{TOP_K}={content_p:.4f}, Recall @{TOP_K}={content_r:.4f}, F1 @{TOP_K}={content_f1:.4f}")

    # ---------------------- 6. 超参数调优（优先加载缓存）----------------------
    print("\n=== 正在调优/加载最优α参数... ===")
    alpha_data = load_cache(CACHE_FILES["alpha"])
    
    if alpha_data is None:
        best_alpha, best_f1, alpha_results = grid_search_alpha(
            rating_matrix, item_sim_df, content_sim_df, test, k=TOP_K, sample_users=100
        )
        alpha_data = (best_alpha, best_f1, alpha_results)
        save_cache(alpha_data, CACHE_FILES["alpha"])
    else:
        best_alpha, best_f1, alpha_results = alpha_data
    
    print(f"最优α: {best_alpha}, 最优F1 @{TOP_K}: {best_f1:.4f}")

    # ---------------------- 7. 生成全局推荐 ----------------------
    print(f"\n=== 正在生成全局Top-{TOP_K}电影推荐... ===")
    global_rec_result = global_movie_recommend(
        ratings_df=ratings,
        movies_df=movies,
        top_k=TOP_K,
        min_rating_count=MIN_RATING_COUNT
    )

    # ---------------------- 8. 保存推荐结果 ----------------------
    with open("reports/global_recommendations.txt", "w", encoding="utf-8") as f:
        f.write(f"电影推荐系统 - 全局Top-{TOP_K}热门电影榜单\n")
        f.write(f"筛选条件：评分人数≥{MIN_RATING_COUNT} | 综合得分=平均评分*0.7 + 归一化热度*1.5\n")
        f.write("-" * 80 + "\n")
        f.write(f"排名\t电影名称\t\t平均评分\t评分人数\t综合得分\n")
        f.write("-" * 80 + "\n")
        for idx, row in global_rec_result.iterrows():
            f.write(f"{idx+1}\t{row['title']:<20}\t{row['avg_rating']:.2f}\t\t{row['rating_count']}\t\t{row['global_score']:.2f}\n")

    # 控制台打印结果
    print("-" * 90)
    print(f"{'排名':<5}{'电影名称':<30}{'平均评分':<10}{'评分人数':<10}{'综合得分':<10}")
    print("-" * 90)
    for idx, row in global_rec_result.iterrows():
        print(f"{idx+1:<5}{row['title']:<30}{row['avg_rating']:<10.2f}{row['rating_count']:<10}{row['global_score']:<10.2f}")
    print("-" * 90)

    # 保存CSV格式
    global_rec_result.to_csv("reports/global_recommendations.csv", index=False, encoding="utf-8-sig")

    # ---------------------- 9. 生成可视化图表 ----------------------
    print("\n=== 正在生成可视化图表... ===")
    plot_data_distribution(ratings, "figures/01_data_distribution.png")
    plot_global_recommendation(global_rec_result, "figures/02_global_recommendation.png")
    plot_model_evaluation(itemcf_metrics, content_metrics, "figures/03_model_evaluation.png")
    plot_rating_vs_popularity(ratings, "figures/04_rating_vs_popularity.png")
    plot_alpha_tuning(alpha_results, "figures/05_alpha_tuning.png")

    # ---------------------- 10. 完成提示 ----------------------
    print(f"\n=== 全局推荐完成！所有结果已保存至：")
    print(f"  - reports/ （推荐结果文本/CSV）")
    print(f"  - figures/ （5张可视化图表）")
    print(f"  - cache/ （机器学习结果缓存，下次运行直接加载）")

if __name__ == "__main__":
    main()