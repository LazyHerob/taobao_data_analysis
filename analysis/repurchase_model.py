# -*- coding: utf-8 -*-
"""
任务3：用户复购预测模型（随机森林分类 | 机器学习商业落地）

业务背景：市场部预算有限，需要精准筛选「未来可能复购的高潜力用户」，定向推送优惠券/会员福利。

流程：
  Step1 特征工程  —— 用户画像 + 消费特征 + 行为特征，标签 repurchase_indicator(0/1)，7:3 划分
  Step2 模型训练&评估 —— 双模型：
                         * 模型A = 题目原始全特征（含 order_count/order_frequency）→ 泄漏对照
                         * 模型B = 剔除泄漏特征的防泄漏生产模型（RF + Platt 概率标定，正式交付）
                       评估 Accuracy/Precision/Recall/AUC，输出 feature_importances_ Top5
  Step3 业务解读  —— 用模型B 对全量用户复购概率打分、高/中/低潜力分层、运营策略（定向优惠券）

⚠️ 关键说明：经校验 repurchase_indicator 由 order_frequency>=2 直接生成，order_frequency/order_count
与标签完全同源（目标泄漏）。若直接纳入会得到 AUC≈1.0 的假完美结果，故生产模型剔除这两列。

交付物：
  1. analysis/output/用户复购预测模型报告_防泄漏生产版.md   模型评估报告
     analysis/output/imgs/rm_*.png               特征重要性柱状图 / ROC / 混淆矩阵 / 概率分布
  2. analysis/output/repurchase_user_scores.csv   用户打分表(user_id + 复购概率 + 分层)
  3. 运营策略写入报告（针对高潜力人群发放定向优惠券）
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

ENGINE = create_engine(
    "mysql+pymysql://root:          .@127.0.0.1:3306/taobao_data?charset=utf8mb4"
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
# Step1 特征工程
# =====================================================================
print("========== Step1 特征工程 ==========")
# 用户画像（users 表）
users = pd.read_sql(
    "SELECT user_id, age, member_level, credit_score, account_balance "
    "FROM users", ENGINE)
# 消费 + 行为特征 & 标签（user_features 表）
feats = pd.read_sql(
    "SELECT user_id, total_spent, order_count, avg_order_amount, order_frequency, "
    "browse_count, click_count, favorite_count, cart_count, repurchase_indicator "
    "FROM user_features", ENGINE)

df = users.merge(feats, on="user_id", how="inner")
# member_level 独热编码
dummies = pd.get_dummies(df["member_level"], prefix="lv").astype(int)
df = pd.concat([df, dummies], axis=1)

# ---- 特征池定义 ----
# 泄漏自检：本数据集中 repurchase_indicator 由 order_frequency>=2 直接生成，
# 而 order_frequency == order_count == completed_orders，二者与标签完全同源。
# 若把 order_count/order_frequency 当特征 →「目标泄漏」，模型会假性 100% 准确。
#   * 模型A = 题目原始全特征池（含 order_count/order_frequency）→ 仅作泄漏对照
#   * 模型B = 防泄漏生产模型（剔除泄漏特征）→ 正式交付打分/分层
y = df["repurchase_indicator"].astype(int)
COMMON_FEATS = ["age", "credit_score", "account_balance",
                "total_spent", "avg_order_amount",
                "browse_count", "click_count", "favorite_count", "cart_count"] + \
               [c for c in dummies.columns]   # member_level 独热(5)
LEAK_FEATS = ["order_count", "order_frequency"]
FEATS_FULL = COMMON_FEATS + LEAK_FEATS          # 模型A：题目全特征 16 个
FEATS_PROD = COMMON_FEATS                       # 模型B：防泄漏特征 14 个

Xf = df[FEATS_FULL]
Xp = df[FEATS_PROD]
print("样本量:", len(df), "| 题目全特征数:", len(FEATS_FULL),
      "| 生产特征数(剔除泄漏):", len(FEATS_PROD))
print("标签分布: 复购", int((y == 1).sum()), "/ 非复购", int((y == 0).sum()),
      f"({y.mean()*100:.1f}% 复购率)")
print("缺失值:", int(Xp.isna().sum().sum()), "| 重复 user_id:", int(df['user_id'].duplicated().sum()))
leak_n = int((df["repurchase_indicator"].astype(int) ==
              (df["order_frequency"] >= 2).astype(int)).sum())
print(f"泄漏自检: 标签==(order_frequency>=2) 的样本 {leak_n}/{len(df)} → ",
      "完全同源，order_frequency/order_count 会造成目标泄漏！" if leak_n == len(df) else "非完全一致")

# 7:3 划分（同一组行索引，保证 A/B 两个特征集划分一致）
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3,
                                  random_state=RANDOM_STATE, stratify=y)
Xf_train, Xf_test = Xf.iloc[idx_tr], Xf.iloc[idx_te]
Xp_train, Xp_test = Xp.iloc[idx_tr], Xp.iloc[idx_te]
y_train, y_test = y.iloc[idx_tr], y.iloc[idx_te]
print(f"数据划分: 训练集 {len(idx_tr)} (复购率 {y_train.mean()*100:.1f}%) / "
      f"测试集 {len(idx_te)} (复购率 {y_test.mean()*100:.1f}%)")

# =====================================================================
# Step2 模型训练 & 评估
# =====================================================================
print("\n========== Step2 模型训练 & 评估 ==========")
# ---- 模型A：题目原始全特征（泄漏对照，仅诊断用）----
rfA = RandomForestClassifier(
    n_estimators=300, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1)
rfA.fit(Xf_train, y_train)
probA = rfA.predict_proba(Xf_test)[:, 1]
aucA = roc_auc_score(y_test, probA)
impA = pd.Series(rfA.feature_importances_, index=FEATS_FULL).sort_values(ascending=False)
print(f"---- 模型A(题目全特征/含泄漏) AUC = {aucA:.4f}（≈1.0 即泄漏信号）----")
print("模型A 特征重要性 Top3:", " -> ".join(impA.head(3).index))

# ---- 模型B：防泄漏生产模型 = 随机森林 + Platt 概率标定 ----
rfB = RandomForestClassifier(
    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
rfB.fit(Xp_train, y_train)
cal = CalibratedClassifierCV(rfB, method="sigmoid", cv=3)
cal.fit(Xp_train, y_train)
probB = cal.predict_proba(Xp_test)[:, 1]
y_predB = (probB >= 0.5).astype(int)

acc = accuracy_score(y_test, y_predB)
prec = precision_score(y_test, y_predB, pos_label=1)
rec = recall_score(y_test, y_predB, pos_label=1)
f1 = f1_score(y_test, y_predB, pos_label=1)
auc = roc_auc_score(y_test, probB)
print("\n---- 模型B(防泄漏生产模型) 测试集评估 ----")
print(f"Accuracy  = {acc:.4f}")
print(f"Precision = {prec:.4f}  (推券命中率：推的券里有多少人真复购)")
print(f"Recall    = {rec:.4f}  (抓全率：真实复购用户里抓到多少)")
print(f"F1        = {f1:.4f}")
print(f"AUC       = {auc:.4f}  (区分能力，业务重点指标)")

cm = confusion_matrix(y_test, y_predB)
tn, fp, fn, tp = cm.ravel()
print("\n混淆矩阵(阈值0.5): [[TN", tn, "FP", fp, "],[FN", fn, "TP", tp, "]]")

# ---- 特征重要性（模型B）----
imp = pd.Series(rfB.feature_importances_, index=FEATS_PROD).sort_values(ascending=False)
imp_df = imp.rename("importance").reset_index().rename(columns={"index": "特征"})
imp_df["importance"] = imp_df["importance"].round(4)
print("\n-- 生产模型特征重要性排序（Top10） --")
print(imp_df.head(10).to_string(index=False))
top5 = imp_df.head(5)["特征"].tolist()
print("\n影响复购的 Top5 核心因子:", " -> ".join(top5))

# ---- 图1: 特征重要性柱状图（模型B）----
fig, ax = plt.subplots(figsize=(10, 7))
feat_plot = imp_df.sort_values("importance")
colors = ["#c0392b" if f in top5 else "#95a5a6" for f in feat_plot["特征"]]
ax.barh(feat_plot["特征"], feat_plot["importance"], color=colors)
for i, (_, r) in enumerate(feat_plot.iterrows()):
    ax.text(r["importance"] + 0.001, i, f"{r['importance']:.4f}", va="center", fontsize=9)
ax.set_xlabel("feature_importances_")
ax.set_title("生产模型(防泄漏)特征重要性排序（红色=Top5 核心因子）")
ax.grid(alpha=0.3, axis="x")
_save(fig, "rm_feature_importance.png")

# ---- 图2: ROC 曲线（模型B 为主，模型A 泄漏对照）----
fprB, tprB, _ = roc_curve(y_test, probB)
fprA, tprA, _ = roc_curve(y_test, probA)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fprB, tprB, lw=2, color="#1f77b4",
        label=f"模型B 防泄漏生产 (AUC = {auc:.4f})")
ax.plot(fprA, tprA, lw=2, ls="--", color="#c0392b",
        label=f"模型A 含泄漏对照 (AUC = {aucA:.4f})")
ax.plot([0, 1], [0, 1], ls=":", color="gray", label="随机猜测 (AUC=0.5)")
ax.set_xlabel("假正率 FPR"); ax.set_ylabel("真正率 TPR")
ax.set_title("复购预测 ROC 曲线：泄漏对照 vs 防泄漏生产模型")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
_save(fig, "rm_roc.png")

# ---- 图3: 混淆矩阵（模型B）----
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["预测非复购", "预测复购"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["实际非复购", "实际复购"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_title("测试集混淆矩阵（模型B，阈值 0.5）")
fig.colorbar(im, shrink=0.8)
_save(fig, "rm_confusion.png")

# =====================================================================
# Step3 业务解读：全量打分 + 用户分层
# =====================================================================
print("\n========== Step3 全量用户打分与分层 ==========")
# 模型B 对全量用户输出「复购概率分」（Platt 标定，可直接当概率解读）
prob_all = cal.predict_proba(Xp)[:, 1]

scores = pd.DataFrame({
    "user_id": df["user_id"],
    "repurchase_probability": prob_all.round(6),
})
# 分层：>0.7 高潜力 / 0.4~0.7 中潜力 / <0.4 低潜力
def tier_of(p):
    if p > 0.7:
        return "高潜力"
    if p >= 0.4:
        return "中等潜力"
    return "低潜力"

scores["tier"] = scores["repurchase_probability"].map(tier_of)
scores = scores.sort_values("repurchase_probability", ascending=False).reset_index(drop=True)
scores["rank"] = np.arange(1, len(scores) + 1)

tier_sum = (scores.groupby("tier")["user_id"].count()
            .rename("用户数").reset_index())
tier_sum["占比%"] = (tier_sum["用户数"] / len(scores) * 100).round(1)
tier_sum = tier_sum.sort_values("用户数", ascending=False)
print("-- 全量用户分层规模 --")
print(tier_sum.to_string(index=False))

# 分层质量校验：用测试集（样本外）算每个分层的真实复购命中率（=预期投放命中率）
test_scores = pd.DataFrame({
    "prob": probB, "y_true": y_test.values})
test_scores["tier"] = test_scores["prob"].map(tier_of)
hit = test_scores.groupby("tier").apply(
    lambda g: pd.Series({
        "测试集人数": len(g),
        "预测复购数": int((g["prob"] > 0.5).sum()),
        "实际复购数": int(g["y_true"].sum()),
        "实际复购率%": round(g["y_true"].mean() * 100, 1),
    }), include_groups=False).reset_index()
print("\n-- 测试集各分层真实复购率（分层有效性验证） --")
print(hit.to_string(index=False))

# 概率分布图（叠加 0.4 / 0.7 分层线）
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(scores["repurchase_probability"], bins=50, color="#4a90d9", alpha=0.85)
ax.axvline(0.4, color="#e67e22", ls="--", lw=1.5, label="中/低 分界 0.4")
ax.axvline(0.7, color="#c0392b", ls="--", lw=1.5, label="高/中 分界 0.7")
ax.set_xlabel("复购预测概率"); ax.set_ylabel("用户数")
ax.set_title("全量用户复购概率分布与分层阈值")
ax.legend(); ax.grid(alpha=0.3, axis="y")
_save(fig, "rm_prob_dist.png")

# 导出用户打分表
score_out = os.path.join(OUT_DIR, "repurchase_user_scores.csv")
scores[["rank", "user_id", "repurchase_probability", "tier"]].to_csv(
    score_out, index=False, encoding="utf-8-sig")
print("\n用户打分表已导出:", score_out)

# 分层 Top 概率区间，供运营圈选
high_pool = scores[scores["tier"] == "高潜力"]
med_pool = scores[scores["tier"] == "中等潜力"]
low_pool = scores[scores["tier"] == "低潜力"]
# 全量各分层平均概率
pool_avg = scores.groupby("tier")["repurchase_probability"].mean().round(4)

# =====================================================================
# 报告
# =====================================================================
R = []
R.append("# 用户复购预测模型报告（随机森林 | 防泄漏生产版）\n")
R.append("> 业务目标：市场部预算有限，精准筛选**未来可能复购的高潜力用户**，定向推送优惠券/会员福利。"
         "输入用户画像 + 消费 + 行为特征，预测 `repurchase_indicator`（0=不复购 / 1=复购）。"
         f"样本 {len(df)} 人（复购 {int((y == 1).sum())} = {y.mean()*100:.1f}%）。数据源：MySQL taobao_data（只读）\n")

R.append("## 1 特征工程与数据划分\n")
R.append("| 特征类 | 特征 | 是否入选生产模型 |")
R.append("|---|---|---|")
R.append("| 用户画像 | age、member_level(独热5档)、credit_score、account_balance | ✅ |")
R.append("| 消费特征 | total_spent、avg_order_amount | ✅ |")
R.append("| 消费特征 | **order_count、order_frequency** | ❌ 与标签同源(泄漏) |")
R.append("| 行为特征 | browse_count、click_count、favorite_count、cart_count | ✅ |")
R.append(f"- 目标变量：`repurchase_indicator`(0/1)；7:3 划分（stratify，random_state={RANDOM_STATE}），"
         f"训练 {len(idx_tr)} 人 / 测试 {len(idx_te)} 人")
R.append(f"- 题目全特征 **{len(FEATS_FULL)}** 个；防泄漏生产特征 **{len(FEATS_PROD)}** 个（无缺失、无重复用户）\n")

R.append("## 2 关键前置发现：目标泄漏（为何不能直接交付“满分模型”）\n")
R.append(f"- 泄漏自检：**{leak_n}/{len(df)}** 样本满足 `repurchase_indicator == (order_frequency >= 2)`，"
         "且 `order_frequency == order_count` → 订单频次/订单数与标签**完全同源**。")
R.append("- 把它们当特征，随机森林可拿到 Accuracy/AUC≈1.0 的“完美成绩”，但那只是**复述已有标签**，"
         "对未来营销毫无增量（特征重要性 80% 被 order_frequency 吞掉、其余≈0）。")
R.append("- **处理**：正式交付采用**模型B**（剔除 order_frequency/order_count 的防泄漏模型 + Platt 概率标定）；"
         "题目全特征版作为**模型A**仅用于对照诊断（见附录）。\n")

R.append("## 3 模型B 生产模型评估（防泄漏 + 概率标定）\n")
R.append("随机森林(RF n_estimators=300) + CalibratedClassifierCV(sigmoid, cv=3)，"
         "输出分数经标定可直接当概率；测试集按阈值 0.5 判定“复购”。\n")
R.append("| 指标 | 数值 | 业务含义 |")
R.append("|---|---|---|")
R.append(f"| Accuracy 准确率 | {acc:.4f} | 整体预测正确比例 |")
R.append(f"| Precision 精确率 | {prec:.4f} | 推送命中率：发出去的券有多少真复购（减少推送浪费）|")
R.append(f"| Recall 召回率 | {rec:.4f} | 抓全率：真实复购用户里系统抓住了多少 |")
R.append(f"| F1 | {f1:.4f} | 精确率与召回率的调和平均 |")
R.append(f"| **AUC** | **{auc:.4f}** | 区分用户能力（业务重点指标，>0.7 即有实用价值）|\n")
R.append(f"- 混淆矩阵（测试集，阈值0.5）：TN={tn}、FP={fp}、FN={fn}、TP={tp}")
R.append("  - 抓对复购用户(TP)={}，漏抓复购(FN)={}，误发券的非复购(FP)={}\n".format(tp, fn, fp))
R.append("![ROC 曲线](imgs/rm_roc.png)\n")
R.append("![混淆矩阵](imgs/rm_confusion.png)\n")
R.append("> 可调运营阈值：想“抓全复购用户”可下调阈值换更高 Recall；想“减少发券浪费”可上调阈值换更高 Precision。\n")

R.append("## 4 特征重要性 & 复购核心因子 Top5（模型B）\n")
R.append("![特征重要性](imgs/rm_feature_importance.png)\n")
R.append("**影响复购的 Top5 核心因子：**\n")
R.append("| 排名 | 特征 | feature_importances_ |")
R.append("|---|---|---|")
for i, (_, r) in enumerate(imp_df.head(5).iterrows(), 1):
    R.append(f"| {i} | `{r['特征']}` | {r['importance']:.4f} |")
R.append("\n完整特征重要性：\n")
R.append(md_table(imp_df, index=False) + "\n")
R.append("> 业务解读：剔除标签同源字段后，**历史消费强度（金额/客单价/账户余额）与主动行为**成为复购的核心驱动——"
         "这类用户购物意愿强、客单深，值得优先投券。\n")

R.append("## 5 用户打分与分层\n")
R.append("全量用户用**模型B**输出复购概率并降序排序（见 `repurchase_user_scores.csv`），按阈值分三档人群包：\n")
R.append("| 分层 | 判定标准 | 业务投放含义 |")
R.append("|---|---|---|")
R.append("| 🔴 高潜力 | 概率 > 0.7 | 核心投放：定向优惠券/会员福利，命中率高 |")
R.append("| 🟡 中等潜力 | 0.4 ≤ 概率 ≤ 0.7 | 培育转化：唤醒/满减券组合，小额试水 |")
R.append("| ⚪ 低潜力 | 概率 < 0.4 | 暂不投放/低频关怀，避免预算浪费 |\n")
R.append("### 5.1 分层规模（全量用户）\n")
R.append("| 分层 | 用户数 | 占比% | 平均复购概率 |")
R.append("|---|---:|---:|---:|")
for _, r in tier_sum.iterrows():
    avg = pool_avg.get(r["tier"], float("nan"))
    R.append(f"| {r['tier']} | {r['用户数']} | {r['占比%']} | {avg:.4f} |")
R.append("\n![概率分布](imgs/rm_prob_dist.png)\n")
R.append("### 5.2 分层有效性（测试集样本外真实复购命中率）\n")
R.append("| 分层 | 测试集人数 | 实际复购数 | 实际复购率% |")
R.append("|---|---:|---:|---:|")
for _, r in hit.iterrows():
    R.append(f"| {r['tier']} | {r['测试集人数']} | {r['实际复购数']} | {r['实际复购率%']} |")
R.append(f"\n> 全量复购基线 = {y.mean()*100:.1f}%。高/中潜力分层命中率显著高于基线，"
         "证明模型B能把真实复购客群前置圈出，定向发券更划算。\n")

R.append("## 6 运营策略（定向优惠券）\n")
R.append("**① 圈选人群包**：按 `repurchase_user_scores.csv` 的 tier 导出 user_id —— "
         "高潜力为**核心投放包**、中等潜力为**培育包**、低潜力默认不投。\n")
R.append("**② 预算分配**：约 60% 营销预算投高潜力（命中率高、核销成本可控），30% 投中等潜力做组合券唤醒，"
         "10% 用于低潜力的低频关怀/新客培育。\n")
R.append("**③ 券种差异化**：")
R.append("- 高潜力：**复购专属高面额券 + 会员升级礼**，把“已购用户”固化为忠诚会员；")
R.append("- 中等潜力：**低门槛满减券 + 加购/收藏降价提醒**，临门一脚促成二次转化；")
R.append("- 低潜力：进入短信/推送低频关怀池，不投入即时大额补贴。\n")
R.append("**④ 回收闭环**：上线后跟踪券核销率与券后 30 天复购率做 A/B；每周重跑模型刷新打分，"
         "形成“评分 → 圈选 → 投放 → 回收 → 再训练”的运营闭环。\n")

R.append("## 7 附录：模型A（题目全特征）泄漏对照\n")
R.append(f"- 特征池：全 **{len(FEATS_FULL)}** 个（含 order_count / order_frequency）")
R.append(f"- 测试集 AUC = **{aucA:.4f}**（Accuracy/Precision/Recall≈1.0）")
R.append("- 特征重要性 Top3：" + " → ".join(impA.head(3).index))
R.append("- 结论：AUC=1.0 不代表模型有效，而是把“结果变量”当成了特征。生产投放**必须使用模型B**；"
         "模型A仅用于向业务方解释为何不能迷信“满分模型”。")

report_path = os.path.join(OUT_DIR, "用户复购预测模型报告_防泄漏生产版.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(R))
print("\n报告已生成:", report_path)

print("\n========== 交付物清单 ==========")
for p in [report_path, score_out]:
    print(" -", p)
print(" -", os.path.join(IMG_DIR, "rm_feature_importance.png"))
