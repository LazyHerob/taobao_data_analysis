# -*- coding: utf-8 -*-
"""
电商经营健康度诊断报告（只读分析）
数据源：MySQL taobao_data 库（与 CSV/SQL 数据一致，只 SELECT 不改动）

口径（已与用户确认）：
- 有效成交订单 = order_status IN ('已付款','已发货','已收货','已完成')
- GMV = 有效订单 total_amount 合计；实际营收 = 有效订单 actual_payment 合计
- 漏斗：创建订单=全部订单；支付=已付款及以上；发货=已发货及以上；收货=已收货/已完成
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

# ---------- 配置 ----------
ENGINE = create_engine(
    "mysql+pymysql://root:          @127.0.0.1:3306/taobao_data?charset=utf8mb4"
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
IMG_DIR = os.path.join(OUT_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 状态口径
VALID = ["已付款", "已发货", "已收货", "已完成"]
SHIPPED = ["已发货", "已收货", "已完成"]
RECEIVED = ["已收货", "已完成"]
VALID_IN = "('" + "','".join(VALID) + "')"

# ---------- 读数据（内存内分析） ----------
orders = pd.read_sql(
    "SELECT order_id,user_id,product_id,quantity,order_date,order_status,"
    "total_amount,discount,actual_payment FROM orders", ENGINE)
users = pd.read_sql("SELECT user_id,province,member_level FROM users", ENGINE)
ufeats = pd.read_sql("SELECT user_id,consumption_level,total_spent FROM user_features", ENGINE)
pf = pd.read_sql(
    "SELECT product_id,total_revenue,popularity_score FROM product_features", ENGINE)
prods = pd.read_sql(
    "SELECT product_id,product_name,category,brand,price,sales_count FROM products", ENGINE)
beh = pd.read_sql(
    "SELECT user_id,behavior_type FROM user_behaviors", ENGINE)

# ---------- 清洗检查（只报告，不改源） ----------
print("========== Step1 数据质量检查 ==========")
print("缺失值：orders 金额列空:", int(orders[["total_amount", "actual_payment"]].isna().sum().sum()),
      "| 金额负值:", int((orders["total_amount"] < 0).sum() | (orders["actual_payment"] < 0).sum()),
      "| 数量<=0:", int((orders["quantity"] <= 0).sum()))
print("users 缺失:", int(users.isna().sum().sum()),
      "| user_features 缺失:", int(ufeats.isna().sum().sum()),
      "| product_features 缺失:", int(pf.isna().sum().sum()))
print("订单时间范围:", orders["order_date"].min(), "~", orders["order_date"].max())

valid_orders = orders[orders["order_status"].isin(VALID)].copy()
print("总订单:", len(orders), "| 有效成交订单:", len(valid_orders))

# ---------- Step2 大盘指标 ----------
gmv = float(valid_orders["total_amount"].sum())
rev = float(valid_orders["actual_payment"].sum())
pay_user = valid_orders["user_id"].nunique()
recv_user = orders[orders["order_status"].isin(RECEIVED)]["user_id"].nunique()
order_user = orders["user_id"].nunique()
browse_user = beh[beh["behavior_type"] == "浏览"]["user_id"].nunique()
# 成交订单数(最终收货) / 创建订单数
recv_orders = int((orders["order_status"].isin(RECEIVED)).sum())
# 行为支付转化率（浏览→有效成交用户）
print("\n========== Step2 核心经营大盘 ==========")
print(f"创建订单数: {len(orders)}   有效成交订单数(已付款+): {len(valid_orders)}")
print(f"最终收货完成订单数: {recv_orders}")
print(f"GMV(有效订单 total_amount): {gmv:,.2f}")
print(f"实际营收(有效订单 actual_payment): {rev:,.2f}")
print(f"折扣让利总额: {float(valid_orders['discount'].sum()):,.2f}")
print(f"去重用户: 浏览{browse_user} / 创建订单{order_user} / 有效支付{pay_user} / 收货{recv_user}")
print(f"下单转化率(收货完成/全部创建): {recv_orders / len(orders):.2%}")
print(f"支付转化率(有效支付/全部创建): {len(valid_orders) / len(orders):.2%}")
print(f"行为支付转化率(有效支付用户/浏览用户): {pay_user / browse_user:.2%}")
print(f"客单价-实收口径(实际营收/有效支付用户): {rev / pay_user:,.2f}")
print(f"客单价-GMV口径(GMV/有效支付用户): {gmv / pay_user:,.2f}")

# 月趋势
trend = valid_orders.copy()
trend["ym"] = pd.to_datetime(trend["order_date"]).dt.to_period("M").astype(str)
mtrend = trend.groupby("ym").agg(gmv=("total_amount", "sum"), rev=("actual_payment", "sum")).sort_index()

# ---------- Step3 用户分层 ----------
print("\n========== Step3 用户分层 ==========")
o_u = valid_orders.merge(users, on="user_id", how="left")
prov = o_u.groupby("province").agg(gmv=("total_amount", "sum"), rev=("actual_payment", "sum"),
                                   users=("user_id", "nunique")).sort_values("rev", ascending=False)
print("\n-- 地域营收 Top5 --")
print(prov.head(5).round(2).to_string())
mem = o_u.groupby("member_level").agg(rev=("actual_payment", "sum"), gmv=("total_amount", "sum"),
                                      orders=("order_id", "count"),
                                      users=("user_id", "nunique")).sort_values("rev", ascending=False)
print("\n-- 会员价值分层 --")
print(mem.round(2).to_string())
# 消费等级占比(人数) 与营收
cu = o_u.merge(ufeats[["user_id", "consumption_level"]], on="user_id", how="left")
cons_lvl = cu.groupby("consumption_level").agg(rev=("actual_payment", "sum"),
                                               users=("user_id", "nunique")).sort_values("rev", ascending=False)
cons_cnt = ufeats["consumption_level"].value_counts()
print("\n-- 消费等级：人数占比 / 营收 --")
for lv in ["高", "中", "低"]:
    if lv in cons_lvl.index:
        print(f"{lv}: 人数{cons_cnt.get(lv,0)} ({cons_cnt.get(lv,0)/len(ufeats):.1%}) | "
              f"营收{cons_lvl.loc[lv,'rev']:,.0f} ({cons_lvl.loc[lv,'rev']/cons_lvl['rev'].sum():.1%})")

# ---------- Step4 商品ABC ----------
print("\n========== Step4 商品 ABC 分层 ==========")
p_rev = valid_orders.groupby("product_id")["actual_payment"].sum().sort_values(ascending=False)
total_rev_all = float(p_rev.sum())
cum = p_rev.cumsum() / total_rev_all
df_p = pd.DataFrame({"rev": p_rev, "cum": cum})
df_p["cls"] = "C"
df_p.loc[df_p["cum"] <= 0.80, "cls"] = "A"
df_p.loc[(df_p["cum"] > 0.80) & (df_p["cum"] <= 0.95), "cls"] = "B"
cls_stat = df_p.groupby("cls").agg(rev=("rev", "sum"), items=("rev", "size")).reindex(["A", "B", "C"])
cls_stat["rev_share"] = cls_stat["rev"] / total_rev_all
cls_stat["item_share"] = cls_stat["items"] / len(df_p)
print(cls_stat.round(4).to_string())
# 热度匹配
p_info = df_p.merge(pf, on="product_id", how="left").merge(
    prods[["product_id", "product_name"]], on="product_id", how="left")
heat_by_cls = p_info.groupby("cls")["popularity_score"].mean()
print("\n-- 各类平均 popularity_score（爆款匹配度） --")
print(heat_by_cls.round(3).to_string())
never_sold = len(prods) - p_rev.shape[0]
print(f"\n从未有有效成交的商品数: {never_sold}")
print("\n-- 爆款 Top10 --")
print(p_info[p_info["cls"] == "A"].head(10)[["product_id", "product_name", "rev", "popularity_score"]]
      .round(2).to_string(index=False))
print("\n-- 滞销样例 Top20(最低营收/含未售) --")
print(p_info[p_info["cls"] == "C"].sort_values("rev").head(20)[["product_id", "product_name", "rev", "popularity_score"]]
      .round(2).to_string(index=False))

# ---------- Step5 漏斗（用户数） ----------
print("\n========== Step5 生命周期漏斗 ==========")
def users_of(table, cond):
    q = f"SELECT COUNT(DISTINCT user_id) AS u FROM {table}"
    if cond:
        q += f" WHERE {cond}"
    return int(pd.read_sql(q, ENGINE)["u"].iloc[0])

browse = beh[beh["behavior_type"] == "浏览"]["user_id"].nunique()
click = beh[beh["behavior_type"] == "点击"]["user_id"].nunique()
cart = beh[beh["behavior_type"] == "加购"]["user_id"].nunique()
favor = beh[beh["behavior_type"] == "收藏"]["user_id"].nunique()
order_all = orders["user_id"].nunique()
paid = orders[orders["order_status"].isin(VALID)]["user_id"].nunique()
ship = orders[orders["order_status"].isin(SHIPPED)]["user_id"].nunique()
recv = orders[orders["order_status"].isin(RECEIVED)]["user_id"].nunique()
print("上游行为意向漏斗: 浏览", browse, "-> 点击", click, "-> 加购", cart)
print("下游订单履约漏斗: 创建", order_all, "-> 支付", paid, "-> 发货", ship, "-> 收货", recv)
# 行为用户中实际下单/支付重叠
b_users = set(beh.loc[beh["behavior_type"] == "浏览", "user_id"])
o_users = set(orders["user_id"])
print("浏览用户中最终下单人数:", len(b_users & o_users),
      f"({len(b_users & o_users)/browse:.1%})")
print("点击流失率(1-点击/浏览):", 1 - click / browse)
print("点击->加购(非强制):", cart / click)
print("下单->支付流失率:", 1 - paid / order_all)
print("支付->发货流失率:", 1 - ship / paid)
print("发货->收货流失率:", 1 - recv / ship)

# ---------- 画图 ----------
def _save(fig, name):
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("已生成图:", path)

# 图1 上游意向漏斗
fig, ax = plt.subplots(figsize=(6, 5))
vals = [browse, click, cart]
labels = [f"浏览\n{browse}", f"点击\n{click}", f"加购\n{cart}"]
for i, v in enumerate(vals):
    w = v / vals[0]
    ax.barh(i, w, color=plt.cm.Reds(0.85 - 0.25 * i), height=0.55)
    ax.text(w / 2, i, labels[i], ha="center", va="center", fontsize=12, color="white")
ax.invert_yaxis(); ax.set_xlim(0, 1.12); ax.axis("off")
ax.set_title("上游行为意向漏斗（去重用户）", fontsize=13)
_save(fig, "funnel_up.png")

# 图2 下游订单履约漏斗
fig, ax = plt.subplots(figsize=(6, 5))
vals = [order_all, paid, ship, recv]
labels = [f"创建订单\n{order_all}", f"支付\n{paid}", f"发货\n{ship}", f"收货完成\n{recv}"]
for i, v in enumerate(vals):
    w = v / vals[0]
    ax.barh(i, w, color=plt.cm.Blues(0.9 - 0.2 * i), height=0.55)
    ax.text(w / 2, i, labels[i], ha="center", va="center", fontsize=12, color="white")
ax.invert_yaxis(); ax.set_xlim(0, 1.1); ax.axis("off")
ax.set_title("下游订单履约漏斗（去重用户）", fontsize=13)
_save(fig, "funnel_down.png")

# 图3 营收月度趋势
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(mtrend.index, mtrend["gmv"] / 1e4, marker="o", label="GMV(万元)", color="#d62728")
ax.plot(mtrend.index, mtrend["rev"] / 1e4, marker="s", label="实际营收(万元)", color="#1f77b4")
for x, y in zip(mtrend.index, mtrend["rev"] / 1e4):
    ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.set_title("有效成交订单：月度 GMV 与实际营收趋势")
ax.legend(); ax.grid(alpha=0.3)
plt.xticks(rotation=45)
_save(fig, "revenue_trend.png")

# 图4 省份营收热力条形图（Top15）
top_prov = prov.head(15).sort_values("rev")
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.YlOrRd(top_prov["rev"] / top_prov["rev"].max())
ax.barh(top_prov.index, top_prov["rev"] / 1e4, color=colors)
for i, (idx, row) in enumerate(top_prov.iterrows()):
    ax.text(row["rev"] / 1e4 + 0.3, i, f"{row['rev']/1e4:,.0f}", va="center", fontsize=9)
ax.set_xlabel("实际营收(万元)"); ax.set_title("省份实际营收热力分布 Top15")
_save(fig, "province_heat.png")

# 图5 会员营收饼图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
mem_sorted = mem.sort_values("rev")
ax1.pie(mem_sorted["rev"], labels=[f"{i}\n{mem_sorted.loc[i,'rev']/mem_sorted['rev'].sum():.1%}"
                                   for i in mem_sorted.index], autopct="", startangle=90,
        colors=plt.cm.Set3(range(len(mem_sorted))))
ax1.set_title("会员营收占比(实际营收)")
ax2.bar(mem_sorted.index, mem_sorted["rev"] / 1e4, color=plt.cm.Set3(range(len(mem_sorted))))
for i, (idx, row) in enumerate(mem_sorted.iterrows()):
    ax2.text(i, row["rev"] / 1e4 + 0.2, f"{row['rev']/1e4:,.0f}", ha="center", fontsize=9)
ax2.set_title("会员营收(万元)与订单量"); ax2.set_ylabel("实际营收(万元)")
ax2b = ax2.twinx(); ax2b.plot(range(len(mem_sorted)), mem_sorted["orders"], "o--", color="gray")
ax2b.set_ylabel("订单量")
_save(fig, "member_pie.png")

# 图6 商品ABC帕累托
fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})
ax = axes[0]
x = range(1, len(df_p) + 1)
ax.plot(x, df_p["cum"].values * 100, color="#1f77b4")
ax.axhline(80, ls="--", color="green", alpha=0.7); ax.axhline(95, ls="--", color="orange", alpha=0.7)
nA = int((df_p["cum"] <= 0.8).sum()); nB = int(((df_p["cum"] > 0.8) & (df_p["cum"] <= 0.95)).sum())
ax.annotate(f"A 截止 {nA}品", (nA, 80), xytext=(nA * 0.4, 84), fontsize=9, color="green")
ax.annotate(f"B 截止 {nA+nB}品", (nA + nB, 95), xytext=(nA + nB, 96), fontsize=9, color="orange")
ax.set_xlabel("商品(按营收降序)"); ax.set_ylabel("累计营收占比(%)")
ax.set_title("商品 ABC 帕累托曲线"); ax.grid(alpha=0.3); ax.set_xlim(0, len(df_p))
ax2 = axes[1]
heat = heat_by_cls.reindex(["A", "B", "C"])
ax2.bar(heat.index, heat.values, color=["#d62728", "#ff7f0e", "#7f7f7f"])
ax2.set_title("各类平均 popularity_score"); ax2.set_ylim(0, 1.2)
_save(fig, "abc_pareto.png")

# ---------- 输出 markdown 报告 ----------
lines = []
lines.append("# 电商经营健康度诊断报告")
lines.append("\n> 分析周期：2025-09 ~ 2026-03（约 6 个月）｜数据源：MySQL taobao_data（只读）｜口径：有效成交订单=已付款/已发货/已收货/已完成\n")
lines.append("## 1 核心经营大盘\n")
lines.append(f"- GMV（有效订单总交易额）：**{gmv:,.2f}** 元")
lines.append(f"- 实际营收（实际到账）：**{rev:,.2f}** 元（折扣让利合计 {float(valid_orders['discount'].sum()):,.2f} 元）")
lines.append(f"- 创建订单 {len(orders)} 笔 / 有效成交 {len(valid_orders)} 笔 / 最终收货完成 {recv_orders} 笔")
lines.append(f"- 下单→最终收货转化率：{recv_orders/len(orders):.2%}；支付转化率：{len(valid_orders)/len(orders):.2%}")
lines.append(f"- 行为支付转化率（有效支付用户/浏览用户）：{pay_user/browse_user:.2%}")
lines.append(f"- 客单价（实收/有效支付用户）：{rev/pay_user:,.2f} 元")
lines.append("\n![营收趋势](imgs/revenue_trend.png)\n")
lines.append("## 2 用户分层\n")
lines.append("### 地域营收 Top5\n")
lines.append(prov.head(5).round(2).to_markdown() + "\n")
lines.append("![省份热力](imgs/province_heat.png)\n")
lines.append("### 会员价值分层\n")
lines.append(mem.round(2).to_markdown() + "\n")
lines.append("![会员营收](imgs/member_pie.png)\n")
lines.append("### 消费等级占比\n")
for lv in ["高", "中", "低"]:
    if lv in cons_lvl.index:
        lines.append(f"- {lv}消费：人数 {cons_cnt.get(lv,0)}（{cons_cnt.get(lv,0)/len(ufeats):.1%}），贡献营收 "
                     f"{cons_lvl.loc[lv,'rev']:,.0f}（{cons_lvl.loc[lv,'rev']/cons_lvl['rev'].sum():.1%}）")
lines.append("\n## 3 商品 ABC 分层\n")
lines.append("| 类别 | 商品数(占比) | 营收(占比) | 平均popularity |")
lines.append("|---|---|---|---|")
for cls in ["A", "B", "C"]:
    lines.append(f"| {cls} | {int(cls_stat.loc[cls,'items'])} ({cls_stat.loc[cls,'item_share']:.1%}) | "
                 f"{cls_stat.loc[cls,'rev']:,.0f} ({cls_stat.loc[cls,'rev_share']:.1%}) | "
                 f"{heat_by_cls.loc[cls]:.3f} |")
lines.append("\n![帕累托](imgs/abc_pareto.png)\n")
lines.append("## 4 订单生命周期漏斗\n")
lines.append("![上游意向漏斗](imgs/funnel_up.png)\n")
lines.append("![下游履约漏斗](imgs/funnel_down.png)\n")
lines.append(f"- 上游：浏览 {browse} → 点击 {click}（点击率 {click/browse:.1%}）→ 加购 {cart}")
lines.append(f"- 下游：创建 {order_all} → 支付 {paid} → 发货 {ship} → 收货完成 {recv}")
lines.append(f"- 浏览用户中最终下单 {len(b_users & o_users)} 人（{len(b_users & o_users)/browse:.1%}）")
lines.append("\n## 5 爆款清单（A类 Top10）\n")
lines.append(p_info[p_info["cls"] == "A"].head(10)[["product_id", "product_name", "rev", "popularity_score"]]
             .round(2).to_markdown(index=False) + "\n")
lines.append("## 6 滞销清单（C类尾部样例）\n")
lines.append(f"> 从未有有效成交的商品 **{never_sold}** 款（共 {len(prods)} 款商品）。\n")
lines.append(p_info[p_info["cls"] == "C"].sort_values("rev").head(20)[["product_id", "product_name", "rev", "popularity_score"]]
             .round(2).to_markdown(index=False) + "\n")

report_path = os.path.join(OUT_DIR, "report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n报告已生成:", report_path)
