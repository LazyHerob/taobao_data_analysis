 # -*- coding: utf-8 -*-
"""只读元数据探查：确认表行数、关键分类字段取值，不做任何写入。"""
import pandas as pd
from sqlalchemy import create_engine

ENGINE = create_engine(
    "mysql+pymysql://root:          @127.0.0.1:3306/taobao_data?charset=utf8mb4"
)

TABLES = [
    "users",
    "products",
    "user_features",
    "product_features",
    "user_behaviors",
    "orders",
]

print("===== 各表行数 =====")
for t in TABLES:
    n = int(pd.read_sql(f"SELECT COUNT(*) AS c FROM `{t}`", ENGINE)["c"].iloc[0])
    print(f"{t:18s} {n}")

print("\n===== orders.order_status 取值分布 =====")
print(pd.read_sql("SELECT order_status, COUNT(*) AS cnt FROM orders GROUP BY order_status ORDER BY cnt DESC", ENGINE).to_string(index=False))

print("\n===== orders 金额负值/异常检查（分状态）=====")
print(pd.read_sql(
    "SELECT order_status, COUNT(*) AS cnt, SUM(total_amount < 0) AS neg_total, "
    "SUM(actual_payment < 0) AS neg_pay, SUM(quantity <= 0) AS bad_qty "
    "FROM orders GROUP BY order_status", ENGINE).to_string(index=False))

print("\n===== user_behaviors.behavior_type 取值分布 =====")
print(pd.read_sql(
    "SELECT behavior_type, COUNT(*) AS cnt, COUNT(DISTINCT user_id) AS users "
    "FROM user_behaviors GROUP BY behavior_type ORDER BY cnt DESC", ENGINE).to_string(index=False))

print("\n===== users.member_level 取值分布 =====")
print(pd.read_sql("SELECT member_level, COUNT(*) AS cnt FROM users GROUP BY member_level ORDER BY cnt DESC", ENGINE).to_string(index=False))

print("\n===== users.province 取值（Top20）=====")
print(pd.read_sql(
    "SELECT province, COUNT(*) AS cnt FROM users GROUP BY province "
    "ORDER BY cnt DESC LIMIT 20", ENGINE).to_string(index=False))

print("\n===== user_features.consumption_level 取值分布 =====")
print(pd.read_sql(
    "SELECT consumption_level, COUNT(*) AS cnt FROM user_features "
    "GROUP BY consumption_level ORDER BY cnt DESC", ENGINE).to_string(index=False))

print("\n===== product_features.popularity_score 基本统计 =====")
print(pd.read_sql(
    "SELECT COUNT(*) AS cnt, MIN(popularity_score) AS mn, MAX(popularity_score) AS mx, "
    "AVG(popularity_score) AS avg_s FROM product_features", ENGINE).to_string(index=False))

print("\n===== 时间范围检查 =====")
print(pd.read_sql(
    "SELECT MIN(order_date) AS mn, MAX(order_date) AS mx, COUNT(*) AS n FROM orders", ENGINE).to_string(index=False))
print(pd.read_sql(
    "SELECT MIN(behavior_time) AS mn, MAX(behavior_time) AS mx FROM user_behaviors", ENGINE).to_string(index=False))

print("\n===== orders 数量与行级完整性抽样检查（前3行）=====")
print(pd.read_sql("SELECT * FROM orders ORDER BY order_id LIMIT 3", ENGINE).to_string(index=False))
