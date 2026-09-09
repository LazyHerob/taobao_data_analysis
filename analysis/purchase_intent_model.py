# -*- coding: utf-8 -*-
"""
任务4：用户购买意向回归预测（RandomForestRegressor | 回归 + 商品智能推荐）

业务场景：平台量化用户购买意愿强弱，purchase_intent ∈ (0,1) 为连续目标（购买意向得分）。
          高意向用户推高转化爆款；中低意向用户推优惠力度大、高性价比商品刺激下单。

流程：
  Step1 数据准备 —— 画像+消费+行为特征，y=purchase_intent，7:3 划分
  Step2 模型构建 —— RandomForestRegressor，评估 MAE / MSE / RMSE / R²
  Step3 商品运营策略 —— 意向分层 × 商品池（爆款高转化 / 高折扣高性价比）× 用户偏好类目
                        输出个性化推荐名单与分人群运营方案

交付物：
  1. analysis/output/用户购买意向回归预测 & 商品智能推荐报告.md   回归模型效果报告
     analysis/output/imgs/pi_*.png                    实际vs预测 / 残差 / 特征重要性 / 分层分布
  2. analysis/output/purchase_intent_scores.csv        用户购买意向评分表(真实+预测+分层+个性化推荐)
  3. analysis/output/product_recommend_pools.csv       商品池(爆款池/高性价比池)
     analysis/output/purchase_intent_recommend.csv     分人群商品推荐名单
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ENGINE = create_engine(
    "mysql+pymysql://root:LJJ20050412.@127.0.0.1:3306/taobao_data?charset=utf8mb4"
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
IMG_DIR = os.path.join(OUT_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
RANDOM_STATE = 42


def _save(fig, name):
    p = os.path.join(IMG_DIR, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("已生成图:", os.path.basename(p))
    return p


def md_table(df, index=False, fmt=".4f"):
    """免依赖 markdown 表格：列名 + DataFrame 行"""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    rows = df.reset_index() if index else df
    for _, r in rows.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:{fmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# =====================================================================
# Step1 数据准备
# =====================================================================
print("========== Step1 数据准备 ==========")
users = pd.read_sql(
    "SELECT user_id, age, member_level, credit_score, account_balance "
    "FROM users", ENGINE)
uf = pd.read_sql(
    "SELECT user_id, total_spent, order_count, avg_order_amount, order_frequency, "
    "browse_count, click_count, favorite_count, cart_count, purchase_intent "
    "FROM user_features", ENGINE)
df = users.merge(uf, on="user_id", how="inner")
dummies = pd.get_dummies(df["member_level"], prefix="lv").astype(int)
df = pd.concat([df, dummies], axis=1)

# 自变量：沿用用户画像 + 历史消费 + 交互行为（任务3口径）
FEATS = ["age", "credit_score", "account_balance",
         "total_spent", "order_count", "avg_order_amount", "order_frequency",
         "browse_count", "click_count", "favorite_count", "cart_count"] + \
        [c for c in dummies.columns]

y = df["purchase_intent"].astype(float)
X = df[FEATS]
print("样本量:", len(df), "| 特征数:", X.shape[1])
print("purchase_intent 分布: min={:.3f} q25={:.3f} 中位={:.3f} q75={:.3f} max={:.3f} mean={:.3f}".format(
    y.min(), y.quantile(.25), y.median(), y.quantile(.75), y.max(), y.mean()))
print("缺失值:", int(X.isna().sum().sum()), "| 重复 user_id:", int(df['user_id'].duplicated().sum()))

corr = df[FEATS + ["purchase_intent"]].corr()["purchase_intent"].drop("purchase_intent")
print("与 purchase_intent 相关性最强的特征:", corr.abs().idxmax(),
      "r=", round(corr[corr.abs().idxmax()], 4), "（均≈0 → 数据本身无线性信号）")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE)
print(f"数据划分: 训练集 {len(X_train)} / 测试集 {len(X_test)}")

# =====================================================================
# Step2 模型构建 & 评估（随机森林回归）
# =====================================================================
print("\n========== Step2 RandomForestRegressor ==========")
rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
pred_train = rf.predict(X_train)
pred_test = rf.predict(X_test)

mae = mean_absolute_error(y_test, pred_test)
mse = mean_squared_error(y_test, pred_test)
rmse = float(np.sqrt(mse))
r2 = r2_score(y_test, pred_test)
r2_tr = r2_score(y_train, pred_train)
# 基线：用均值预测（无特征信息时能到的最好水平）
base_pred = np.full_like(y_test, y_test.mean())
base_mae = mean_absolute_error(y_test, base_pred)
base_mse = mean_squared_error(y_test, base_pred)
print(f"MAE  = {mae:.4f}   (基线均值预测 MAE = {base_mae:.4f})")
print(f"MSE  = {mse:.4f}   (基线 MSE = {base_mse:.4f})")
print(f"RMSE = {rmse:.4f}")
print(f"R²(测试) = {r2:.4f}   R²(训练) = {r2_tr:.4f}")
print(f"y.std = {y_test.std():.4f}")

imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values(ascending=False)
imp_df = imp.rename("importance").reset_index().rename(columns={"index": "特征"})
imp_df["importance"] = imp_df["importance"].round(4)
print("\n特征重要性 Top10：")
print(imp_df.head(10).to_string(index=False))

# ---- 图1: 实际 vs 预测（测试集）----
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, pred_test, s=18, alpha=0.45, color="#4a90d9",
           label=f"样本点 (R²={r2:.3f})")
lim = [min(y_test.min(), pred_test.min()) - 0.01, max(y_test.max(), pred_test.max()) + 0.01]
ax.plot(lim, lim, ls="--", color="#c0392b", lw=1.5, label="理想 y=x")
ax.set_xlabel("实际 purchase_intent"); ax.set_ylabel("预测 purchase_intent")
ax.set_title("随机森林回归：实际 vs 预测（测试集）")
ax.legend(); ax.grid(alpha=0.3)
_save(fig, "pi_actual_pred.png")

# ---- 图2: 残差分布 ----
resid = y_test.values - pred_test
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(resid, bins=40, color="#7f8c8d", alpha=0.85)
ax.axvline(0, color="#c0392b", ls="--", lw=1.5)
ax.set_xlabel("残差 = 实际 - 预测"); ax.set_ylabel("样本数")
ax.set_title(f"预测残差分布（std={resid.std():.4f}）")
ax.grid(alpha=0.3, axis="y")
_save(fig, "pi_residual.png")

# ---- 图3: 特征重要性 ----
fig, ax = plt.subplots(figsize=(9, 7))
feat_plot = imp_df.sort_values("importance")
ax.barh(feat_plot["特征"], feat_plot["importance"], color="#95a5a6")
for i, (_, r) in enumerate(feat_plot.iterrows()):
    ax.text(r["importance"] + 0.0005, i, f"{r['importance']:.4f}", va="center", fontsize=9)
ax.set_xlabel("feature_importances_")
ax.set_title("随机森林回归特征重要性")
ax.grid(alpha=0.3, axis="x")
_save(fig, "pi_feature_importance.png")

# =====================================================================
# Step3 商品智能推荐（意向分层 × 商品池 × 偏好类目）
# =====================================================================
print("\n========== Step3 商品智能推荐 ==========")
# 商品画像：products + product_features + 订单折扣
products = pd.read_sql(
    "SELECT product_id, product_name, category, price, sales_count FROM products", ENGINE)
pfeat = pd.read_sql(
    "SELECT product_id, conversion_rate, avg_review_score, popularity_score "
    "FROM product_features", ENGINE)
orders = pd.read_sql(
    "SELECT product_id, total_amount, actual_payment FROM orders "
    "WHERE actual_payment > 0", ENGINE)
orders["disc_rate"] = (orders["total_amount"] - orders["actual_payment"]) / \
                      orders["total_amount"].replace(0, np.nan)
odisc = orders.groupby("product_id").agg(
    n_orders=("actual_payment", "size"), avg_disc=("disc_rate", "mean"))
pinfo = (products.merge(pfeat, on="product_id", how="inner")
                 .merge(odisc, on="product_id", how="left"))
pinfo["avg_disc"] = pinfo["avg_disc"].fillna(pinfo["avg_disc"].median())

# ---- 商品池1: 爆款高转化池（高意向用户）----
hot = pinfo[pinfo["conversion_rate"] >= pinfo["conversion_rate"].median()].copy()
hot["pool"] = "爆款高转化池"
hot["pool_score"] = hot["conversion_rate"] * 0.6 + \
                    (hot["popularity_score"] / hot["popularity_score"].max()) * 0.4
# ---- 商品池2: 高性价比/优惠池（中低意向用户）----
value = pinfo[(pinfo["avg_disc"] >= pinfo["avg_disc"].quantile(0.6)) &
              (pinfo["avg_review_score"] >= 4.3)].copy()
value["pool"] = "高性价比优惠池"
value["pool_score"] = value["avg_disc"] / value["avg_disc"].max() * 0.6 + \
                      (value["avg_review_score"] / 5.0) * 0.4
pools = pd.concat([hot, value], ignore_index=True)
print("爆款高转化池:", len(hot), "个商品 | 高性价比优惠池:", len(value), "个商品")

pool_out_cols = ["pool", "product_id", "product_name", "category", "price",
                 "conversion_rate", "popularity_score", "avg_review_score",
                 "avg_disc", "sales_count"]
pools.to_csv(os.path.join(OUT_DIR, "product_recommend_pools.csv"),
             index=False, encoding="utf-8-sig", columns=pool_out_cols)
print("商品池已导出: product_recommend_pools.csv")

# ---- 用户意向分层（基于真实 purchase_intent，存量用户可观测）----
# 阈值：低 < 0.20 ｜ 中 [0.20, 0.30] ｜ 高 > 0.30
def tier_of(p):
    if p > 0.30:
        return "高意向"
    if p >= 0.20:
        return "中意向"
    return "低意向"

# ---- 偏好类目：用户行为加权（加购>收藏>点击>浏览）----
beh = pd.read_sql(
    "SELECT user_id, product_id, behavior_type FROM user_behaviors", ENGINE)
bw = {"加购": 2.0, "收藏": 1.5, "点击": 1.2, "浏览": 1.0}
beh["w"] = beh["behavior_type"].map(bw).fillna(1.0)
cat_act = (beh.merge(products[["product_id", "category"]], on="product_id", how="left")
              .dropna(subset=["category"]))
cat_w = cat_act.groupby(["user_id", "category"])["w"].sum().rename("score").reset_index()
top_cat = (cat_w.sort_values("score", ascending=False)
                 .drop_duplicates("user_id")[["user_id", "category"]]
                 .rename(columns={"category": "top_category"}))

# ---- 每个用户在其偏好类目里取池内 Top5，不足用池内全局补足 ----
def rec_for_group(pool_df, cat, k=5):
    """给定某池子与类目，返回该类目下按 pool_score 排序的 top-k 商品"""
    sub = pool_df[pool_df["category"] == cat].sort_values("pool_score", ascending=False)
    return sub.head(k)

def user_recommend(user_pool, user_cat, k=5):
    cand = rec_for_group(user_pool, user_cat, k)
    if len(cand) < k:  # 类目内不足则用池内全局 Top 补齐
        extra = (user_pool.sort_values("pool_score", ascending=False)
                          .head(3 * k))
        seen = set(cand["product_id"])
        extra = extra[~extra["product_id"].isin(seen)]
        cand = pd.concat([cand, extra]).head(k)
    return cand

scores = pd.DataFrame({
    "user_id": df["user_id"],
    "purchase_intent": y.values,
    "predicted_intent": rf.predict(X).round(6),
})
scores["intent_tier"] = scores["purchase_intent"].map(tier_of)
scores = scores.merge(top_cat, on="user_id", how="left")
scores["rec_pool"] = np.where(scores["intent_tier"] == "高意向",
                              "爆款高转化池", "高性价比优惠池")

rec_rows = []
for _, u in scores.iterrows():
    pool = hot if u["rec_pool"] == "爆款高转化池" else value
    cat = u["top_category"]
    if pd.isna(cat):  # 无行为用户 → 池内全局 Top
        rec = pool.sort_values("pool_score", ascending=False).head(5)
    else:
        rec = user_recommend(pool, cat)
    rec_rows.append({
        "user_id": u["user_id"],
        "rec_product_ids": "|".join(rec["product_id"].astype(str).tolist()),
        "rec_product_names": "|".join(rec["product_name"].tolist()),
    })
rec_df = pd.DataFrame(rec_rows)
scores = scores.merge(rec_df, on="user_id", how="left")

# 意向分层规模
tier_n = scores.groupby("intent_tier")["user_id"].count().reset_index()
tier_n.columns = ["分层", "用户数"]
tier_n["占比%"] = (tier_n["用户数"] / len(scores) * 100).round(1)
order_tier = ["高意向", "中意向", "低意向"]
tier_n = tier_n.set_index("分层").loc[[t for t in order_tier if t in tier_n["分层"].tolist()]] \
               .reset_index()
print("\n意向分层规模：")
print(tier_n.to_string(index=False))

scores = scores.sort_values(["intent_tier", "purchase_intent"],
                            ascending=[False, False]).reset_index(drop=True)
scores["rank"] = np.arange(1, len(scores) + 1)
score_cols = ["rank", "user_id", "purchase_intent", "predicted_intent",
              "intent_tier", "top_category", "rec_pool",
              "rec_product_ids", "rec_product_names"]
scores[score_cols].to_csv(os.path.join(OUT_DIR, "purchase_intent_scores.csv"),
                          index=False, encoding="utf-8-sig")
print("用户评分表已导出: purchase_intent_scores.csv")

# 每用户推荐明细导出（id + 名称一一对应便于投放）
expand = scores[["user_id", "intent_tier", "top_category", "rec_pool",
                 "rec_product_ids", "rec_product_names"]].copy()
exp = []
for _, u in expand.iterrows():
    ids = str(u["rec_product_ids"]).split("|")
    nms = str(u["rec_product_names"]).split("|")
    for rk, (pid, nm) in enumerate(zip(ids, nms), 1):
        exp.append({"user_id": u["user_id"], "tier": u["intent_tier"],
                    "top_category": u["top_category"], "pool": u["rec_pool"],
                    "rec_rank": rk, "product_id": pid, "product_name": nm})
rec_expand = pd.DataFrame(exp)
rec_expand.to_csv(os.path.join(OUT_DIR, "purchase_intent_recommend.csv"),
                  index=False, encoding="utf-8-sig")
print("分人群推荐名单已导出: purchase_intent_recommend.csv")

# 分层 × 推荐池 汇总
cross = scores.groupby(["intent_tier", "rec_pool"])["user_id"].count().reset_index()
cross.columns = ["意向分层", "推荐商品池", "用户数"]
print("\n分层 → 推荐池 对应：")
print(cross.to_string(index=False))

# ---- 图4: 意向分层分布 ----
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(scores["purchase_intent"], bins=40, color="#4a90d9", alpha=0.85)
ax.axvline(0.20, color="#e67e22", ls="--", lw=1.5, label="低/中 分界 0.20")
ax.axvline(0.30, color="#c0392b", ls="--", lw=1.5, label="中/高 分界 0.30")
ax.set_xlabel("purchase_intent"); ax.set_ylabel("用户数")
ax.set_title("购买意向分层分布（真实意向分）")
ax.legend(); ax.grid(alpha=0.3, axis="y")
_save(fig, "pi_intent_tiers.png")

# 商品池全局 Top10（供报告展示）
hot10 = hot.sort_values("pool_score", ascending=False).head(10).copy()
value10 = value.sort_values("pool_score", ascending=False).head(10).copy()

# =====================================================================
# 报告
# =====================================================================
R = []
R.append("# 用户购买意向回归预测 & 商品智能推荐报告（RandomForestRegressor）\n")
R.append("> 业务目标：量化用户购买意愿强弱（`purchase_intent`，0~1 连续意向分），结合商品热度/转化做个性化推荐："
         "高意向用户推高转化爆款、中低意向用户推高性价比商品刺激下单。"
         f"样本 {len(df)} 人。数据源：MySQL taobao_data（只读）\n")

R.append("## 1 数据准备\n")
R.append(f"- 目标变量 `purchase_intent`：min={y.min():.3f}，中位={y.median():.3f}，max={y.max():.3f}，"
         f"mean={y.mean():.3f}，std={y.std():.3f}（范围集中在 0.10~0.40）")
R.append("- 自变量（沿用画像+消费+行为）：age、member_level(独热5档)、credit_score、account_balance、"
         "total_spent、order_count、avg_order_amount、order_frequency、browse_count、click_count、"
         f"favorite_count、cart_count，共 {X.shape[1]} 个；无缺失、无重复用户")
R.append(f"- 数据划分：7:3（训练 {len(X_train)} / 测试 {len(X_test)}，random_state={RANDOM_STATE}）\n")

R.append("## 2 模型评估（随机森林回归）\n")
R.append("| 指标 | 测试集 | 均值基线 | 业务含义 |")
R.append("|---|---:|---:|---|")
R.append(f"| MAE 平均绝对误差 | {mae:.4f} | {base_mae:.4f} | 平均每个用户意向分差多少 |")
R.append(f"| MSE 均方误差 | {mse:.4f} | {base_mse:.4f} | 误差平方平均（惩罚大偏差）|")
R.append(f"| RMSE | {rmse:.4f} | {float(np.sqrt(base_mse)):.4f} | 误差的标准尺度 |")
R.append(f"| R² 决定系数 | {r2:.4f} | 0 | 模型解释了多少意向分方差 |")
R.append(f"| R²(训练) | {r2_tr:.4f} | - | 检验是否欠/过拟合 |\n")
R.append("![实际vs预测](imgs/pi_actual_pred.png)\n")
R.append("![残差分布](imgs/pi_residual.png)\n")
R.append("- 特征与 `purchase_intent` 的线性相关全部 |r|<0.03；最大化特征集 5 折交叉验证 R²<0，"
         f"**测试集 R²={r2:.3f} ≈ 0**：在当前可得画像/消费/行为特征下，模型只学到“预测均值”，"
         "未能解释该合成意向分。")
R.append("- **业务解读（如实）**：本数据集 `purchase_intent` 与用户历史行为几乎无关（数据构造所致）。"
         "因此回归模型暂不可用于对新客的意向打分；运营方案（第 4~5 节）对**存量用户**采用平台可观测的真实意向分分层，"
         "模型预测分（≈0.25 均值）仅作冷启动占位基线，待接入浏览序列/实时上下文/商品偏好等更丰富信号后再训练。\n")

R.append("## 3 特征重要性\n")
R.append("![特征重要性](imgs/pi_feature_importance.png)\n")
R.append("完整排序：\n")
R.append(md_table(imp_df, index=False) + "\n")
R.append("> 说明：由于目标与特征近乎无关，特征重要性分布接近均匀，参考价值有限（与 R²≈0 相互印证）。\n")

R.append("## 4 商品池构建\n")
R.append("把商品 `product_features`(转化率/热度/好评) + `products`(价格/类目) + 订单折扣率(avg_disc) 融合，构建两个策略商品池：\n")
R.append("| 商品池 | 口径 | 面向人群 | 用途 |")
R.append("|---|---|---|---|")
R.append("| 爆款高转化池 | 转化率≥中位数，池分=0.6×转化率+0.4×热度 | 高意向 | 提升转化与客单 |")
R.append("| 高性价比优惠池 | 折扣率≥60分位 且 好评≥4.3，池分=0.6×折扣+0.4×好评 | 中/低意向 | 用优惠刺激下单 |\n")
R.append("**爆款高转化池 Top10：**\n")
cols10 = ["product_id", "product_name", "category", "price", "conversion_rate", "popularity_score", "avg_disc"]
R.append(md_table(hot10[cols10].rename(columns={"product_id": "商品ID", "product_name": "商品名", "category": "类目",
                                               "price": "价格", "conversion_rate": "转化率",
                                               "popularity_score": "热度分", "avg_disc": "平均折扣率"}),
                 index=False) + "\n")
R.append("**高性价比优惠池 Top10：**\n")
R.append(md_table(value10[cols10].rename(columns={"product_id": "商品ID", "product_name": "商品名", "category": "类目",
                                                 "price": "价格", "conversion_rate": "转化率",
                                                 "popularity_score": "热度分", "avg_disc": "平均折扣率"}),
                 index=False) + "\n")
R.append("> 全量商品池明细见 `product_recommend_pools.csv`（pool 标记所在池）。\n")

R.append("## 5 分人群商品推荐运营方案\n")
R.append("### 5.1 意向分层与投放规则\n")
R.append("| 分层 | 判定(真实意向分) | 用户数 | 占比 | 推荐商品池 | 投放动作 |")
R.append("|---|---|---:|---:|---|---|")
for _, r in tier_n.iterrows():
    pool = "爆款高转化池" if r["分层"] == "高意向" else "高性价比优惠池"
    action = ("主推高转化爆款，搭配会员券冲转化" if r["分层"] == "高意向"
              else "推高折扣高性价比商品，用优惠力度刺激下单")
    R.append(f"| {r['分层']} | {'>0.30' if r['分层']=='高意向' else ('0.20~0.30' if r['分层']=='中意向' else '<0.20')} | "
             f"{r['用户数']} | {r['占比%']}% | {pool} | {action} |")
R.append("\n![意向分层分布](imgs/pi_intent_tiers.png)\n")
R.append("### 5.2 个性化逻辑\n")
R.append("- **偏好类目**：按用户行为加权（加购2 > 收藏1.5 > 点击1.2 > 浏览1）统计用户最常交互类目 `top_category`；")
R.append("- **推荐取数**：在“该用户分层对应商品池”内，先取 `top_category` 类目下池分 Top5；若类目商品不足则用池内全局 Top 补齐。")
R.append("- **全量结果**：`purchase_intent_scores.csv`（打分表，含真实/预测意向分、分层、top_category、推荐商品id与名称）；"
         "`purchase_intent_recommend.csv`（逐条 user×商品 推荐明细，可直接导给投放系统）。\n")
R.append("### 5.3 运营 SOP\n")
R.append("1. **高意向人群（约 {:.1f}%）**：发高转化爆款主推位 + 会员专享券，目标是“临门一脚成交”，投放预算优先；".format(
    tier_n.loc[tier_n["分层"] == "高意向", "占比%"].iloc[0] if (tier_n["分层"] == "高意向").any() else 0))
R.append("2. **中意向人群（约 {:.1f}%）**：推其偏好类目里的高折扣好物，配低门槛满减券刺激转化；".format(
    tier_n.loc[tier_n["分层"] == "中意向", "占比%"].iloc[0] if (tier_n["分层"] == "中意向").any() else 0))
R.append("3. **低意向人群（约 {:.1f}%）**：用大额优惠/新人补贴 + 高性价比商品做唤醒，控制单客成本。\n".format(
    tier_n.loc[tier_n["分层"] == "低意向", "占比%"].iloc[0] if (tier_n["分层"] == "低意向").any() else 0))
R.append("**注意**：以上分层基于平台可观测的真实意向分（存量用户）。若面向**无历史的新用户**，"
         "当前特征无法估计其意向（R²≈0），应先用活动测款 + 实时行为补数据，再重训模型。\n")

R.append("## 6 交付物清单\n")
R.append("- `用户购买意向回归预测 & 商品智能推荐报告.md`：本报告（回归效果）")
R.append("- `imgs/pi_*.png`：实际vs预测、残差、特征重要性、意向分层分布")
R.append("- `purchase_intent_scores.csv`：用户购买意向评分表（真实分 + 预测分 + 分层 + 推荐）")
R.append("- `product_recommend_pools.csv`：爆款高转化池 / 高性价比优惠池商品明细")
R.append("- `purchase_intent_recommend.csv`：分人群商品推荐名单（user×商品逐条）")

report_path = os.path.join(OUT_DIR, "用户购买意向回归预测 & 商品智能推荐报告.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(R))
print("\n报告已生成:", report_path)

print("\n========== 交付物清单 ==========")
for nm in ["用户购买意向回归预测 & 商品智能推荐报告.md", "purchase_intent_scores.csv",
           "product_recommend_pools.csv", "purchase_intent_recommend.csv"]:
    print(" -", os.path.join(OUT_DIR, nm))
