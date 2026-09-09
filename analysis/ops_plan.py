# -*- coding: utf-8 -*-
"""
任务5：综合业务落地 —— 会员精细化运营策略方案（顶层业务输出）

把任务1~4 的分析结论沉淀为 4 套可落地运营策略：
  ① 高价值会员留存      ② 沉睡低复购用户唤醒（优惠券触达）
  ③ 滞销商品清库存      ④ 流量转化提升（流失最高环节优化）

用户分群：RFM 优化 —— M=消费等级(自带) + F/R=复购预测分(任务3模型分, 样本外AUC≈0.75) + 订单频次

交付物：
  1. analysis/output/会员精细化运营与增长策略方案.md                    顶层业务方案（主交付）
  2. analysis/output/imgs/op_segments.png           用户分层画像图
     analysis/output/imgs/op_funnel_bar.png         转化漏斗流失环节图
  3. analysis/output/ops_user_segments.csv          全量用户分层+建议动作（含券参数）
  4. analysis/output/ops_cart_recovery.csv          加购未支付/未下单 限时券召回名单
  5. analysis/output/ops_slow_bundles.csv           滞销×爆款捆绑促销清单
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

ENGINE = create_engine(
    "mysql+pymysql://root:          @127.0.0.1:3306/taobao_data?charset=utf8mb4"
)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
IMG_DIR = os.path.join(OUT_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, name):
    p = os.path.join(IMG_DIR, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("已生成图:", os.path.basename(p))


def md_table(df, index=False, fmt=".4f"):
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
# 1. 用户资产盘点与分群（RFM 优化：消费等级 M + 复购预测分 F/R）
# =====================================================================
print("========== Step1 用户资产盘点 ==========")
users = pd.read_sql("SELECT user_id, member_level, age, gender, province FROM users", ENGINE)
uf = pd.read_sql(
    "SELECT user_id, consumption_level, total_spent, order_count, order_frequency, "
    "completed_orders, repurchase_indicator, browse_count, click_count, "
    "favorite_count, cart_count FROM user_features", ENGINE)
scores = pd.read_csv(os.path.join(OUT_DIR, "repurchase_user_scores.csv"),
                     usecols=["user_id", "repurchase_probability"])
d = users.merge(uf, on="user_id").merge(scores, on="user_id")

# 分群规则：价值=高消费&复购分≥0.5；潜力=复购分≥0.4且非价值；流失沉睡=复购分<0.4
V = (d["consumption_level"] == "高") & (d["repurchase_probability"] >= 0.5)
P = (d["repurchase_probability"] >= 0.4) & ~V
S = d["repurchase_probability"] < 0.4
d["segment"] = np.where(V, "价值用户", np.where(P, "潜力用户", "流失沉睡"))

seg_stat = d.groupby("segment").apply(lambda g: pd.Series({
    "人数": len(g),
    "占比%": round(len(g) / len(d) * 100, 1),
    "人均消费(元)": round(g["total_spent"].mean(), 0),
    "平均订单频次": round(g["order_frequency"].mean(), 2),
    "复购率%": round(g["repurchase_indicator"].mean() * 100, 1),
    "平均复购预测分": round(g["repurchase_probability"].mean(), 3),
}), include_groups=False).reindex(["价值用户", "潜力用户", "流失沉睡"])
print(seg_stat.round(2).to_string())

# 沉睡细分：0单 / 1单（首购未复购）
d["freq_g"] = np.where(d["order_frequency"] >= 2, "复购≥2次",
                       np.where(d["order_frequency"] >= 1, "仅1单", "从未下单"))
sub_dorm = d.loc[S, "freq_g"].value_counts()
print("沉睡人群内部:\n", sub_dorm.to_string())

# 价值用户会员等级分布（留存权益依据）
val_mem = (d.loc[V, "member_level"].value_counts(normalize=True) * 100).round(1)
print("价值用户会员等级占比:\n", val_mem.to_string())

# 会员等级价值（营收贡献，有效成交口径）
orders = pd.read_sql(
    "SELECT user_id, actual_payment FROM orders "
    "WHERE order_status IN ('已付款','已发货','已收货','已完成')", ENGINE)
m = orders.merge(users[["user_id", "member_level"]], on="user_id")
mem_users = m.drop_duplicates(["member_level", "user_id"]).groupby("member_level").size()
mem_rev = m.groupby("member_level")["actual_payment"].sum()
mem_val = pd.concat([mem_users.rename("users_n"), mem_rev.rename("rev")], axis=1).reset_index()
mem_val["rev_share%"] = (mem_val["rev"] / mem_val["rev"].sum() * 100).round(1)
mem_val = mem_val.sort_values("rev", ascending=False)
print(mem_val.to_string(index=False))

# ---- 图1: 分层画像 ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
names = ["价值用户", "潜力用户", "流失沉睡"]
sizes = seg_stat["人数"].values
spend = seg_stat["人均消费(元)"].values
bars = a1.bar(names, sizes, color=["#c0392b", "#e67e22", "#95a5a6"], width=0.55)
for b, s in zip(bars, seg_stat["占比%"].values):
    a1.text(b.get_x() + b.get_width() / 2, b.get_height() + 60,
            f"{s:.1f}%", ha="center", fontsize=11)
a1.set_title("用户分群规模（共 5000 人）"); a1.set_ylabel("用户数"); a1.grid(alpha=0.3, axis="y")
bars2 = a2.bar(names, spend, color=["#c0392b", "#e67e22", "#95a5a6"], width=0.55)
for b, s in zip(bars2, spend):
    a2.text(b.get_x() + b.get_width() / 2, b.get_height() + 400,
            f"{s:,.0f}", ha="center", fontsize=11)
a2.set_title("人均累计消费（元）"); a2.set_ylabel("total_spent(元)"); a2.grid(alpha=0.3, axis="y")
fig.suptitle("会员分群画像（M=消费等级 + 复购预测分）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
_save(fig, "op_segments.png")

# =====================================================================
# 2. 转化漏斗（策略四：流失环节）
# =====================================================================
print("\n========== Step2 转化漏斗 ==========")
beh = pd.read_sql("SELECT user_id, behavior_type FROM user_behaviors", ENGINE)
stg = {s: set(beh.loc[beh["behavior_type"] == s, "user_id"]) for s in ["浏览", "点击", "收藏", "加购"]}
n_browse, n_click, n_fav, n_cart = (len(stg["浏览"]), len(stg["点击"]),
                                    len(stg["收藏"]), len(stg["加购"]))
o = pd.read_sql("SELECT user_id, order_status FROM orders", ENGINE)
paid_st = set(o.loc[o["order_status"].isin(["已付款", "已发货", "已收货", "已完成"]), "user_id"])
rcv_st = set(o.loc[o["order_status"].isin(["已收货", "已完成"]), "user_id"])
any_ord = set(o["user_id"])
n_order, n_paid, n_rcv = len(any_ord), len(paid_st), len(rcv_st)

cart_no_order = len(stg["加购"] - any_ord)     # 加购但从未下单
cart_no_pay = len(stg["加购"] - paid_st)       # 加购但从未支付
favcart_no_pay = len((stg["收藏"] | stg["加购"]) - paid_st)  # 收藏/加购 但未支付
browse_no_click = len(stg["浏览"] - stg["点击"])            # 浏览未点击
click_no_cart = len(stg["点击"] - stg["加购"])              # 点击未加购
intent_no_pay = len((stg["浏览"] | stg["点击"]) - paid_st)  # 有点击/浏览但从未支付
print(f"漏斗: 浏览{n_browse} 点击{n_click} 收藏{n_fav} 加购{n_cart} | "
      f"下单{n_order} 支付{n_paid} 收货{n_rcv}")
print(f"加购未下单 {cart_no_order} | 加购未支付 {cart_no_pay} | 收藏/加购未支付 {favcart_no_pay} | "
      f"浏览未点击 {browse_no_click} | 点击未加购 {click_no_cart} | 浏览/点击未支付 {intent_no_pay}")

# 流失环比（下游履约链：下单→支付→收货，事件链可环比）
order_events = pd.read_sql(
    "SELECT COUNT(*) n FROM orders WHERE order_status IN ('已付款','已发货','已收货','已完成')", ENGINE).iloc[0, 0]
# 用户数口径流失率
drop_cart2pay = (1 - (n_cart - cart_no_pay) / n_cart) * 100 if n_cart else 0
drop_pay = (1 - n_paid / n_order) * 100

# ---- 图2: 漏斗流失环节 ----
stages = ["浏览", "点击", "收藏", "加购", "下单", "支付", "收货完成"]
counts = [n_browse, n_click, n_fav, n_cart, n_order, n_paid, n_rcv]
colors = ["#8fb3d9", "#4a90d9", "#1f77b4", "#f39c12", "#e67e22", "#27ae60", "#16a085"]
fig, ax = plt.subplots(figsize=(9, 6))
y = np.arange(len(stages))[::-1]
ax.barh(y, counts, color=colors)
for yi, c, s in zip(y, counts, stages):
    ax.text(c + 40, yi, f"{c:,}", va="center", fontsize=10)
ax.set_yticks(y); ax.set_yticklabels(stages)
ax.set_xlabel("触达/成交用户数（去重）")
ax.set_title("用户生命周期漏斗（流失环节定位：加购→支付 存在临门流失）")
ax.grid(alpha=0.3, axis="x")
_save(fig, "op_funnel_bar.png")

# =====================================================================
# 3. 商品 ABC 分层 & 滞销清库存（策略三）
# =====================================================================
print("\n========== Step3 商品ABC与滞销 ==========")
pf = pd.read_sql(
    "SELECT product_id, total_revenue, completed_count, conversion_rate, "
    "popularity_score, avg_review_score FROM product_features", ENGINE)
pr = pd.read_sql("SELECT product_id, product_name, category, price FROM products", ENGINE)
g = pf.merge(pr, on="product_id")
g = g.sort_values("total_revenue", ascending=False).reset_index(drop=True)
g["rev_cum"] = g["total_revenue"].cumsum() / g["total_revenue"].sum()
# ABC：A 累计0~80%，B 80~95%，C 95~100%
def abc_of(r):
    return "A" if r <= 0.80 else ("B" if r <= 0.95 else "C")
g["abc"] = g["rev_cum"].map(abc_of)
abc_stat = g.groupby("abc").agg(
    商品数=("product_id", "size"),
    营收占比=("total_revenue", lambda s: round(s.sum() / g["total_revenue"].sum() * 100, 1)),
    平均popularity=("popularity_score", "mean")).reindex(["A", "B", "C"])
print(abc_stat.round(2).to_string())

# 滞销候选：C 类内 completed_count 低（真实动销差）
slow = g[(g["abc"] == "C") & (g["completed_count"] <= 1)].copy()
print("滞销候选(C类 且 完成单≤1):", len(slow))
# 引流爆款：每类目 revenue 最高的 A 类商品
cat_top = g[g["abc"] == "A"].sort_values("total_revenue", ascending=False) \
    .drop_duplicates("category")[["category", "product_id", "product_name", "total_revenue"]]
# 每类目最滞销代表（同 A 类爆款搭配）
slow_rep = slow.sort_values("total_revenue").drop_duplicates("category", keep="first") \
    .merge(cat_top, on="category", suffixes=("_slow", "_hot"))
slow_rep = slow_rep.rename(columns={
    "product_id_slow": "滞销商品ID", "product_name_slow": "滞销商品",
    "total_revenue_slow": "滞销营收", "product_id_hot": "引流爆款ID",
    "product_name_hot": "引流爆款", "total_revenue_hot": "爆款营收"})
slow_rep = slow_rep[["category", "引流爆款", "滞销商品", "爆款营收", "滞销营收"]]
print("可捆绑类目组合:", len(slow_rep))

# =====================================================================
# 4. 落地名单导出
# =====================================================================
# 4.1 全量分群 + 建议动作（策略1/2）
def coupon_of(r):
    if r["segment"] == "价值用户":
        return "会员专属权益(不主发券)"
    if r["segment"] == "潜力用户":
        return "小额升档券：满200-30，30天有效"
    if r["freq_g"] == "从未下单":
        return "首单唤醒大额券：满100-50，7天有效"
    return "复购唤醒券：满300-80，14天有效"  # 流失沉睡-仅1单
d["建议动作"] = d.apply(coupon_of, axis=1)
seg_out = d[["user_id", "member_level", "consumption_level", "segment", "freq_g",
             "total_spent", "order_frequency", "repurchase_probability", "建议动作"]]
seg_out = seg_out.sort_values(["segment", "repurchase_probability"],
                              ascending=[True, False]).reset_index(drop=True)
seg_out.to_csv(os.path.join(OUT_DIR, "ops_user_segments.csv"), index=False, encoding="utf-8-sig")
print("全量分群已导出: ops_user_segments.csv")

# 4.2 加购未支付/未下单 召回名单（策略4）
cart_meta = users.merge(uf, on="user_id")
cart_users = d[d["user_id"].isin(stg["加购"] - paid_st)].copy()
cart_users["是否从未下单"] = cart_users["user_id"].isin(any_ord).map(lambda x: "否(有单未支付)" if x else "是(从未下单)")
cart_out = cart_users[["user_id", "member_level", "total_spent", "order_frequency",
                       "repurchase_probability", "是否从未下单"]]
cart_out = cart_out.sort_values(["是否从未下单", "total_spent"], ascending=[True, False])
cart_out.to_csv(os.path.join(OUT_DIR, "ops_cart_recovery.csv"), index=False, encoding="utf-8-sig")
print("加购召回名单已导出: ops_cart_recovery.csv  (", len(cart_out), "人 )")

# 4.3 滞销×爆款捆绑清单（策略3）
slow_bundle = slow_rep.copy()
slow_bundle["玩法"] = "同品类满额立减/第二件半价/加9.9元换购"
slow_bundle.to_csv(os.path.join(OUT_DIR, "ops_slow_bundles.csv"), index=False, encoding="utf-8-sig")
print("滞销捆绑清单已导出: ops_slow_bundles.csv  (", len(slow_bundle), "组 )")

# =====================================================================
# 5. 生成顶层方案 会员精细化运营与增长策略方案.md
# =====================================================================
R = []
R.append("# 会员精细化运营与增长策略方案（综合落地）\n")
R.append("> **定位**：基于任务1~4 的完整分析（健康度大盘、复购影响因素、复购预测模型、购买意向与商品推荐），"
         "产出 4 套可直接执行的运营策略。数据口径与明细见附录。")
R.append(f"> **数据底座**：用户 {len(d)}｜有效成交订单 {order_events:,} 笔｜商品 {len(g)} 款｜全量复购率 34.6%。\n")

R.append("## 0 执行摘要（TL;DR）\n")
R.append(f"- **用户分层**（RFM 优化：消费等级 M + 复购预测分 F/R）：价值用户 {int(V.sum())} 人（高消费+高复购分，人均消费 "
         f"{seg_stat.loc['价值用户','人均消费(元)']:,.0f} 元、复购率 {seg_stat.loc['价值用户','复购率%']}%）；"
         f"潜力用户 {int(P.sum())} 人；流失沉睡 {int(S.sum())} 人（{seg_stat.loc['流失沉睡','占比%']}%，其中从未下单 "
         f"{int(sub_dorm.get('从未下单', 0))} 人、仅 1 单 {int(sub_dorm.get('仅1单', 0))} 人）。")
R.append("- **四套策略**：①高价值会员留存（专属权益防流失）；②沉睡/低复购唤醒（分层优惠券，名单已导出）；"
         "③滞销商品清库存（C 类×同品类爆款捆绑促销）；④流量转化提升（加购未支付/未下单 限时券 + 上游意向召回）。")
R.append("- **优先级建议**：先跑 ④（临门一脚，见效最快、成本最低）与 ②（沉睡池最大，唤醒即增量）→ 再上 ③（清库存回笼资金）→ ① 常态化维护。\n")

R.append("## 1 用户资产盘点与分群\n")
R.append("### 1.1 分群口径（RFM 优化）\n")
R.append("| 维度 | 数据来源 | 说明 |")
R.append("|---|---|---|")
R.append("| M 消费金额 | 自带 `consumption_level`（高 1589 / 中 1890 / 低 1521）| 帕累托：31.8% 高消费用户贡献 71.3% 营收 |")
R.append("| F 频次 | `order_frequency`（0单 1469 / 1单 1803 / ≥2次复购 1728）| 复购=至少完成 2 次购买 |")
R.append("| R/F 预测 | 任务3 复购预测分（防泄漏模型，样本外 AUC≈0.75）| 前瞻性“还会不会复购” |")
R.append("| 沉睡标记 | 0 单或仅 1 单 且 预测分低 | 需唤醒 |\n")
R.append("### 1.2 三群画像\n")
R.append("| 分层 | 判定规则 | 人数 | 占比 | 人均消费 | 平均频次 | 复购率 | 复购预测分 |")
R.append("|---|---|---:|---:|---:|---:|---:|---:|")
for seg in ["价值用户", "潜力用户", "流失沉睡"]:
    r = seg_stat.loc[seg]
    R.append(f"| {seg} | " + {
        "价值用户": "消费等级=高 & 复购分≥0.5",
        "潜力用户": "复购分≥0.4 且非价值（多为中消费）",
        "流失沉睡": "复购分<0.4（含 0 单/仅 1 单）"}[seg] +
        f" | {int(r['人数'])} | {r['占比%']}% | {r['人均消费(元)']:,.0f} | "
        f"{r['平均订单频次']:.2f} | {r['复购率%']}% | {r['平均复购预测分']:.3f} |")
R.append("\n![分层画像](imgs/op_segments.png)\n")
R.append(f"- 价值用户（{int(V.sum())} 人）会员结构：钻石 {val_mem.get('钻石会员', 0):.0f}% / 金牌 {val_mem.get('金牌会员', 0):.0f}% / "
         f"银牌 {val_mem.get('银牌会员', 0):.0f}% / 铜牌 {val_mem.get('铜牌会员', 0):.0f}% / 普通 {val_mem.get('普通会员', 0):.0f}%。")
R.append("- 会员营收贡献（有效成交）：\n")
mem_md = mem_val.copy()
mem_md.columns = ["会员等级", "消费人数", "营收(元)", "营收占比%"]
R.append(md_table(mem_md[["会员等级", "消费人数", "营收占比%"]], fmt=".2f") + "\n")
R.append("![会员营收结构](imgs/member_pie.png)\n")

R.append("## 2 策略一：高价值会员留存（维护存量基本盘）\n")
R.append(f"**人群**：价值用户 {int(V.sum())} 人（贡献高营收的头部，人均消费 {seg_stat.loc['价值用户','人均消费(元)']:,.0f} 元、"
         f"复购率 {seg_stat.loc['价值用户','复购率%']}%）。**目标**：降流失、提年消费、做口碑。\n")
R.append("| 动作 | 机制 | 承接权益 |")
R.append("|---|---|---|")
R.append("| 专属身份 | 钻石/金牌高价值直通，免运费、专属客服、新品优先购 | 提升离店成本 |")
R.append("| 付费锁客 | 年卡/季卡（含 0 门槛包邮+专属折扣），用“已付会员费”锚定消费 | 提高跨品类复购 |")
R.append("| 关怀触发 | 生日礼券、30 天未消费自动送关怀券（额度高于沉睡档以免被低估）| 防止高价值掉档 |")
R.append("| 价值上探 | 高价值×高意向：按任务4 商品池推其偏好类目的**爆款高转化品** | 客单提升 |")
R.append("| 客户成功 | 大额异常（仅退款/客诉）48h 专人跟进 | 保口碑 |")
R.append("**KPI**：会员流失率≤3%/季；高价值人均年消费 +15%；付费会员续费率≥60%。\n")

R.append("## 3 策略二：沉睡低复购用户唤醒（优惠券触达）\n")
R.append(f"**人群**：流失沉睡 {int(S.sum())} 人（占比 {seg_stat.loc['流失沉睡','占比%']}%）+ 潜力用户 {int(P.sum())} 人。"
         f"沉睡内部：从未下单 {int(sub_dorm.get('从未下单', 0))}、仅 1 单 {int(sub_dorm.get('仅1单', 0))}（有信任基础，唤醒性价比更高）。\n")
R.append("| 子人群 | 触达券 | 有效期/渠道 | 目标 |")
R.append("|---|---|---|---|")
R.append(f"| 从未下单（{int(sub_dorm.get('从未下单', 0))} 人）| 首单大额券 满100-50 | 7 天｜App push+短信 | 破首单 |")
R.append(f"| 仅1单未复购（{int(sub_dorm.get('仅1单', 0))} 人）| 复购唤醒券 满300-80 | 14 天｜短信+站内信 | 促成第2单 |")
R.append(f"| 潜力用户（{int(P.sum())} 人，中等消费）| 小额升档券 满200-30 | 30 天｜App | 复购并提客单 |")
R.append("- 投放节奏：分 3 批每批间隔 3 天，避免疲劳；券后埋点归因，看核销率与券后复购率。")
R.append("- 名单已导出 `ops_user_segments.csv`（含 user_id/分层/券建议），可直接圈选推送。\n")
R.append("**示例测算（假设标注，非实证）**：若唤醒券核销率 10~15%、增量复购率 5~8%，沉睡 3150 人约可带来 150~250 个增量第 1/2 单；"
         "按客单价 {:.0f} 元估算撬动营收数十万元量级，ROI 需以实际核销回收为准。\n".format(d.loc[S, "total_spent"].mean()))

R.append("## 4 策略三：滞销商品清库存\n")
R.append("### 4.1 商品 ABC 分层（营收口径）\n")
abc_md = abc_stat.reset_index()
abc_md.columns = ["类别", "商品数", "营收占比%", "平均popularity"]
R.append(md_table(abc_md, fmt=".2f") + "\n")
R.append("![帕累托](imgs/abc_pareto.png)\n")
R.append(f"- **爆款(A 类 {int((g['abc']=='A').sum())} 款，80% 营收)**：持续投流量、保排名、做组合引流款；")
R.append(f"- **长尾(B 类 {int((g['abc']=='B').sum())} 款)**：按需补货，配合券促销做“腰部提量”；")
R.append(f"- **滞销(C 类 {int((g['abc']=='C').sum())} 款，5% 营收)**：清库存主战场。\n")
R.append("### 4.2 清库存玩法（滞销×爆款捆绑）\n")
R.append(f"- 候选池：C 类且真实完成单≤1 的商品（{len(slow)} 款）——动销差、占仓储、压资金。")
R.append("- **捆绑促销**：同品类爆款带滞销——\u201c买爆款 +9.9 元换购\u201d/\u201c爆款+滞销组合满额立减\u201d，用爆款流量清长尾；")
R.append("- **独立促销**：滞销区满减+大额券叠加清库，价格带参考其类目折扣上限；")
R.append("- **搭配销售**：与高意向用户（任务4）偏好类目的价值池/高性价比池联动推荐。\n")
R.append("捆绑组合样例（每类目 1 组，全量见 `ops_slow_bundles.csv`）：\n")
bundle_md = slow_rep.head(15).copy()
bundle_md.columns = ["类目", "引流爆款", "滞销品", "爆款营收(元)", "滞销营收(元)"]
R.append(md_table(bundle_md, fmt=".2f") + "\n")
R.append("**KPI**：季度 C 类动销率 +10pct、清库资金回笼率、滞销占用降低。\n")

R.append("## 5 策略四：流量转化提升（流失环节优化）\n")
R.append("### 5.1 环节流失测算（用户去重口径）\n")
R.append("| 环节 | 用户数 | 说明 |")
R.append("|---|---:|---|")
R.append(f"| 浏览 | {n_browse:,} | 认知层 |")
R.append(f"| 点击 | {n_click:,} | 点击率 {n_click/n_browse*100:.1f}%（流失 {browse_no_click:,} 人未点击）|")
R.append(f"| 收藏/加购 | {n_fav:,} / {n_cart:,} | 高意向信号 |")
R.append(f"| 下单 | {n_order:,} | 其中未支付 {n_order - n_paid:,} 人 |")
R.append(f"| 支付 | {n_paid:,} | 下单→支付 {n_paid/n_order*100:.1f}% |")
R.append(f"| 收货完成 | {n_rcv:,} | 履约层 |")
R.append("\n![漏斗](imgs/op_funnel_bar.png)\n")
R.append("**流失定位**：最大的人群级漏点在**“看→加购”上游（浏览未加购为主）**与**“加购→支付”临门流失**；"
         f"其中 加购未下单 {cart_no_order} 人、加购未支付 {cart_no_pay} 人（已并入 {favcart_no_pay} 人“收藏/加购但从未支付”高意向池）、"
         f"下单未支付 {n_order-n_paid} 人。加购到下单链条是**转化意愿最强、召回成本最低**的环节。\n")
R.append("### 5.2 动作清单\n")
R.append("| 优先级 | 人群 | 动作 | 预期 |")
R.append("|---|---|---|---|")
R.append(f"| P0 | 加购/收藏未支付 {favcart_no_pay} 人 | **限时优惠券**（如 满200-60/免邮，12~24h 有效期）+ 库存紧张提醒 | 临门转化，见效最快 |")
R.append(f"| P0 | 加购但从未下单 {cart_no_order} 人 | 限时券+组合购（绑定其浏览爆款）| 破单 |")
R.append(f"| P1 | 下单未支付 {n_order-n_paid} 人 | 自动催付短信（30/120 分钟两波）| 挽回支付 |")
R.append(f"| P2 | 浏览未点击/点击未加购 {click_no_cart:,} 人 | 内容种草+价格锚点+推荐位优化（兴趣→加购）| 扩大上层池子 |")
R.append("- 加购未支付/未下单召回名单已导出 `ops_cart_recovery.csv`，可直接圈选发放限时券。\n")
R.append("**KPI**：加购未支付转化率提升至 15%+；整体 浏览→成交 转化率环比提升 1~2pct。\n")

R.append("## 6 排期、预算与风险\n")
R.append("| 阶段 | 内容 | 周期 | 负责 |")
R.append("|---|---|---|---|")
R.append("| Phase 1 | 策略④ 加购召回 + 策略② 沉睡券 | 第 1-2 周 | 运营+增长 |")
R.append("| Phase 2 | 策略③ 滞销捆绑清库 | 第 2-4 周 | 采销+类目运营 |")
R.append("| Phase 3 | 策略① 高价值会员体系常态化 | 持续 | 会员运营 |")
R.append("\n**预算假设**：券面额按子人群分档；建议首期预算先投 ④+②（召回ROI 最高），回收数据后再滚动放大 ③。"
         "**风险**：①过度折扣伤毛利 → 设券池上限与毛利率红线；②唤醒疲劳 → 控制频次与沉默期；"
         "③数据口径变化 → 统一以有效成交口径复盘。\n")

R.append("## 7 数据依据与交付物\n")
R.append("- 引用分析：`report.md`(经营大盘/漏斗/ABC)、`repurchase_report.md`(复购驱动)、"
         "`用户复购预测模型报告_防泄漏生产版.md`(复购预测模型)、`用户购买意向回归预测 & 商品智能推荐报告.md`(意向+商品池)")
R.append("- 本方案新增交付：`会员精细化运营与增长策略方案.md`(本文件)、`op_segments.png`、`op_funnel_bar.png`、"
         "`ops_user_segments.csv`、`ops_cart_recovery.csv`、`ops_slow_bundles.csv`")

report_path = os.path.join(OUT_DIR, "会员精细化运营与增长策略方案.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(R))
print("\n顶层运营方案已生成:", report_path)

print("\n========== 交付物清单 ==========")
for nm in ["会员精细化运营与增长策略方案.md", "ops_user_segments.csv", "ops_cart_recovery.csv", "ops_slow_bundles.csv"]:
    print(" -", os.path.join(OUT_DIR, nm))
print(" -", os.path.join(IMG_DIR, "op_segments.png"))
print(" -", os.path.join(IMG_DIR, "op_funnel_bar.png"))
