import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(data_dir="data/ml-100k/"):
    """
    加载并清洗MovieLens 100K数据
    :param data_dir: 数据集文件夹路径
    :return: 清洗后的评分数据、电影信息数据、训练集、测试集
    """
    # ---------------------- 1. 加载数据 ----------------------
    # 加载评分数据
    ratings = pd.read_csv(
        f"{data_dir}u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    # 加载电影信息数据（只取有用列）
    genre_cols = [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    movies = pd.read_csv(
        f"{data_dir}u.item",
        sep="|",
        encoding="latin-1",
        usecols=[0, 1, 2] + list(range(5, 24)),
        names=["movie_id", "title", "release_date", "unknown"] + genre_cols
    )
    movies = movies.drop("unknown", axis=1)  # 删除无用的unknown列

    # ---------------------- 2. 数据清洗 ----------------------
    # 2.1 评分数据清洗
    ratings = ratings.drop("timestamp", axis=1)
    ratings = ratings[(ratings["rating"] >= 1) & (ratings["rating"] <= 5)]
    
    # 2.2 电影数据清洗：提取上映年份
    movies["release_year"] = pd.to_datetime(movies["release_date"]).dt.year
    movies = movies.drop("release_date", axis=1)
    movies["release_year"] = movies["release_year"].fillna(movies["release_year"].median())
    
    # 2.3 过滤小众用户/电影
    user_count = ratings["user_id"].value_counts()
    movie_count = ratings["movie_id"].value_counts()
    ratings = ratings[ratings["user_id"].isin(user_count[user_count >= 5].index)]
    ratings = ratings[ratings["movie_id"].isin(movie_count[movie_count >= 10].index)]
    
    # 2.4 对齐电影ID
    common_movies = list(set(ratings["movie_id"]) & set(movies["movie_id"]))
    ratings = ratings[ratings["movie_id"].isin(common_movies)]
    movies = movies[movies["movie_id"].isin(common_movies)]

    # ---------------------- 3. 拆分训练集/测试集 ----------------------
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    
    return ratings, movies, train, test

if __name__ == "__main__":
    # 测试代码
    ratings, movies, train, test = load_and_clean_data()
    print(f"清洗后评分数据量：{len(ratings)}")
    print(f"训练集：{len(train)}，测试集：{len(test)}")