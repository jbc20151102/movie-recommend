import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_matrices(rating_matrix, content_matrix):
    """
    计算ItemCF相似度矩阵和内容相似度矩阵（优化：添加数值稳定性处理）
    """
    # ---------------------- 1. ItemCF相似度矩阵 ----------------------
    # 标准化评分矩阵（减少用户评分偏差影响）
    rating_matrix_centered = rating_matrix.sub(rating_matrix.mean(axis=1), axis=0)
    rating_matrix_centered = rating_matrix_centered.fillna(0)
    
    item_sim = cosine_similarity(rating_matrix_centered.T)
    # 处理数值误差（限制在-1到1之间）
    item_sim = np.clip(item_sim, -1, 1)
    item_sim_df = pd.DataFrame(
        item_sim, 
        index=rating_matrix.columns, 
        columns=rating_matrix.columns
    )

    # ---------------------- 2. 内容相似度矩阵 ----------------------
    # 标准化内容特征矩阵
    content_matrix_centered = (content_matrix - content_matrix.mean()) / (content_matrix.std() + 1e-8)  # 避免除0
    content_sim = cosine_similarity(content_matrix_centered)
    content_sim = np.clip(content_sim, -1, 1)
    content_sim_df = pd.DataFrame(
        content_sim, 
        index=content_matrix.index, 
        columns=content_matrix.index
    )

    return item_sim_df, content_sim_df

if __name__ == "__main__":
    # 测试代码（需先运行前两个文件）
    from data_preprocessing import load_and_clean_data
    from feature_engineering import build_feature_matrices
    ratings, movies, train, test = load_and_clean_data()
    rating_matrix, content_matrix, genre_cols = build_feature_matrices(train, movies)
    item_sim_df, content_sim_df = compute_similarity_matrices(rating_matrix, content_matrix)
    print("ItemCF相似度矩阵示例：")
    print(item_sim_df.iloc[:3, :3])