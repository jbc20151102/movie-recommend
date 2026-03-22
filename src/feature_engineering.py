import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_feature_matrices(train, movies):
    """
    构建协同过滤评分矩阵和内容特征矩阵（新增年份多样性优化）
    :param train: 训练集评分数据
    :param movies: 电影信息数据
    :return: 评分矩阵、内容特征矩阵、电影类型列名列表
    """
    # ---------------------- 1. 协同过滤评分矩阵 ----------------------
    rating_matrix = train.pivot_table(
        index="user_id", 
        columns="movie_id", 
        values="rating", 
        fill_value=0
    )

    # ---------------------- 2. 内容特征矩阵 ----------------------
    genre_cols = [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    
    # 2.1 电影类型特征
    genre_features = movies[["movie_id"] + genre_cols].set_index("movie_id")
    
    # 2.2 上映年份特征（优化：增加年份离散化 + 归一化）
    scaler = StandardScaler()
    year_features = movies[["movie_id", "release_year"]].copy()
    # 新增：年份离散化（按年代分组）
    year_features["decade"] = (year_features["release_year"] // 10) * 10
    decade_dummies = pd.get_dummies(year_features["decade"], prefix="decade", drop_first=True)
    year_features = pd.concat([year_features, decade_dummies], axis=1)
    
    # 归一化年份
    year_features["release_year_scaled"] = scaler.fit_transform(year_features[["release_year"]])
    year_features = year_features.set_index("movie_id")[["release_year_scaled"] + list(decade_dummies.columns)]
    
    # 2.3 拼接特征，对齐电影顺序
    content_matrix = pd.concat([genre_features, year_features], axis=1)
    content_matrix = content_matrix.loc[rating_matrix.columns]  # 确保和评分矩阵的电影一致
    
    return rating_matrix, content_matrix, genre_cols