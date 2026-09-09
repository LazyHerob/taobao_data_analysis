# -*- coding: utf-8 -*-
"""补充只读探查：确认漏斗各阶段去重用户数与时间窗交集，避免漏斗非单调。"""
import pandas as pd
from sqlalchemy import create_engine

ENGINE = create_engine(
    "mysql+pymysql://root:         @127.0.0.1:3306/taobao_data?charset=utf8mb4"
)

# 行为日志时间窗
beh_win = pd.read_sql(
    "SELECT MIN(behavior_time) AS mn, MAX(behavior_time) AS mx FROM user_behaviors", ENGINE
)
bmin, bmax = beh_win["mn"].iloc[0], beh_win["mx"].iloc[0]
print("行为时间窗:", bmin, "~", bmax)

# 各阶段去重用户数（全量）
stages = {
    "浏览": ("user_behaviors", "behavior_type='浏览'", None),
    "点击": ("user_behaviors", "behavior_type='点击'", None),
    "加购": ("user_behaviors", "behavior_type='加购'", None),
    "收藏": ("user_behaviors", "behavior_type='收藏'", None),
    "创建订单(全订单)": ("orders", None, None),
    "支付(已付款及以上)": ("orders", "order_status IN ('已付款','已发货','已收货','已完成')", None),
    "发货(已发货及以上)": ("orders", "order_status IN ('已发货','已收货','已完成')", None),
    "收货完成(已收货/已完成)": ("orders", "order_status IN ('已收货','已完成')", None),
}
print("\n===== 各阶段去重用户数（全量）=====")
for name, (tbl, cond, _) in stages.items():
    where = f" WHERE {cond}" if cond else ""
    n = int(pd.read_sql(
        f"SELECT COUNT(DISTINCT user_id) AS u FROM `{tbl}`{where}", ENGINE)["u"].iloc[0])
    print(f"{name:22s} {n}")

# 限制在行为时间窗内的订单各阶段去重用户数（与行为可比）
print("\n===== 限制订单在行为时间窗内 的各阶段去重用户数 =====")
wmin = str(bmin).replace(" ", "T").split("T")[0]
wmax = str(bmax).replace(" ", "T").split("T")[0]
for name, cond in [
    ("创建订单(窗内)", None),
    ("支付(窗内)", "order_status IN ('已付款','已发货','已收货','已完成')"),
    ("发货(窗内)", "order_status IN ('已发货','已收货','已完成')"),
    ("收货完成(窗内)", "order_status IN ('已收货','已完成')"),
]:
    where = f" WHERE order_date BETWEEN '{wmin}' AND '{wmax}'"
    if cond:
        where += f" AND {cond}"
    n = int(pd.read_sql(
        f"SELECT COUNT(DISTINCT user_id) AS u FROM orders{where}", ENGINE)["u"].iloc[0])
    print(f"{name:22s} {n}")

# 关键：浏览用户里有多少最终下单/支付
print("\n===== 行为用户中进入订单漏斗的情况 =====")
sql = """
SELECT
  COUNT(DISTINCT b.user_id) AS browse_users,
  COUNT(DISTINCT CASE WHEN o.order_id IS NOT NULL THEN b.user_id END) AS browse_then_order,
  COUNT(DISTINCT CASE WHEN o.order_status IN ('已付款','已发货','已收货','已完成')
                 THEN b.user_id END) AS browse_then_paid
FROM (SELECT DISTINCT user_id FROM user_behaviors WHERE behavior_type='浏览') b
LEFT JOIN orders o ON o.user_id = b.user_id
"""
print(pd.read_sql(sql, ENGINE).to_string(index=False))

# 检查每个商品在 orders 中累计营收与 product_features.total_revenue 是否对得上（决定ABC数据源）
print("\n===== product_features.total_revenue 合计 vs orders 有效订单合计 =====")
print(pd.read_sql("SELECT SUM(total_revenue) AS pf FROM product_features", ENGINE).to_string(index=False))
print(pd.read_sql(
    "SELECT SUM(total_amount) AS gmv, SUM(actual_payment) AS pay "
    "FROM orders WHERE order_status IN ('已付款','已发货','已收货','已完成')", ENGINE).to_string(index=False))
