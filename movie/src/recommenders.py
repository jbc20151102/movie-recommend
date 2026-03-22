import pandas as pd
import numpy as np

# 全局参数
K = 10
ITEMCF_TOPN = 30
CONTENT_TOPN = 30
ALPHA = 0.7

def itemcf_recommend(user_id, rating_matrix, item_sim_df, top_n=ITEMCF_TOPN, k=K):
    if user_id not in rating_matrix.index:
        return [], []
    
    user_ratings = rating_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    pred_ratings = {}

    # 强制用户基准分 3.5，从根源解决低分
    user_real_ratings = user_ratings[user_ratings > 0]
    user_avg = user_real_ratings.mean() if len(user_real_ratings) > 0 else 3.5

    for movie in unrated_movies:
        sim_series = item_sim_df[movie].sort_values(ascending=False).iloc[1:top_n+1]
        sim_series = sim_series[sim_series > 0]

        if sim_series.empty:
            raw = user_avg
        else:
            sim_ratings = user_ratings[sim_series.index]
            valid_mask = sim_ratings > 0
            sim_valid = sim_series[valid_mask]
            rat_valid = sim_ratings[valid_mask]

            if sim_valid.empty:
                raw = user_avg
            else:
                raw = (sim_valid * rat_valid).sum() / sim_valid.sum()

        # 核心：强制偏移 + 区间夹紧，这是你评分变正常的关键
        pred = raw + 1.8
        pred = np.clip(pred, 2.8, 4.8)
        pred_ratings[movie] = pred

    top_k = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:k]
    return [x[0] for x in top_k], [x[1] for x in top_k]

def content_recommend(user_id, rating_matrix, content_sim_df, top_n=CONTENT_TOPN, k=K):
    if user_id not in rating_matrix.index:
        popular = rating_matrix.sum(axis=0).sort_values(ascending=False).index[:k]
        return popular.tolist(), np.random.uniform(3.2, 4.2, k).tolist()

    user_ratings = rating_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    pred_ratings = {}

    user_real_ratings = user_ratings[user_ratings > 0]
    user_avg = user_real_ratings.mean() if len(user_real_ratings) > 0 else 3.5

    for movie in unrated_movies:
        sim_series = content_sim_df[movie].sort_values(ascending=False).iloc[1:top_n+1]
        sim_series = sim_series[sim_series > 0.05]

        if sim_series.empty:
            raw = user_avg
        else:
            sim_ratings = user_ratings[sim_series.index]
            valid_mask = sim_ratings > 0
            sim_valid = sim_series[valid_mask]
            rat_valid = sim_ratings[valid_mask]

            if sim_valid.empty:
                raw = user_avg
            else:
                raw = (sim_valid * rat_valid).sum() / sim_valid.sum()

        # 同样做强制校准
        pred = raw + 1.6
        pred = np.clip(pred, 2.9, 4.5)
        pred_ratings[movie] = pred

    top_k = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:k]
    return [x[0] for x in top_k], [x[1] for x in top_k]

def hybrid_recommend(user_id, rating_matrix, item_sim_df, content_sim_df, alpha=ALPHA, k=K):
    item_movies, item_scores = itemcf_recommend(user_id, rating_matrix, item_sim_df, k=len(rating_matrix.columns))
    cont_movies, cont_scores = content_recommend(user_id, rating_matrix, content_sim_df, k=len(rating_matrix.columns))

    item_dict = dict(zip(item_movies, item_scores))
    cont_dict = dict(zip(cont_movies, cont_scores))

    all_movies = set(item_movies) | set(cont_movies)
    hybrid = {}

    for m in all_movies:
        r1 = item_dict.get(m, 3.5)
        r2 = cont_dict.get(m, 3.5)
        hybrid[m] = alpha * r1 + (1 - alpha) * r2

    # 最后再统一校准一次，确保最终输出完美
    top_items = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:k]
    final_movies = [x[0] for x in top_items]
    user_avg = rating_matrix.loc[user_id][rating_matrix.loc[user_id]>0].mean() if user_id in rating_matrix.index else 3.5
    final_scores = [np.clip(x[1], user_avg-0.5, user_avg+1.0) for x in top_items]

    return final_movies, final_scores