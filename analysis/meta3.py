# -*- coding: utf-8 -*-
"""只读探查：复购任务目标分布与关键特征取值。"""
import pandas as pd
from sqlalchemy import create_engine

ENGINE = create_engine(
    "mysql+pymysql://root:          @127.0.0.1:3306/taobao_data?charset=utf8mb4"
)

print("===== 目标变量 repurchase_indicator 分布 =====")
print(pd.read_sql(
    "SELECT repurchase_indicator, COUNT(*) AS cnt FROM user_features "
    "GROUP BY repurchase_indicator", ENGINE).to_string(index=False))

print("\n===== user_features 列 =====")
print(list(pd.read_sql("SELECT * FROM user_features LIMIT 1", ENGINE).columns))

print("\n===== users.age 统计 / gender 分布 =====")
print(pd.read_sql("SELECT MIN(age) AS mn, MAX(age) AS mx, AVG(age) AS avg FROM users", ENGINE).to_string(index=False))
print(pd.read_sql("SELECT gender, COUNT(*) AS cnt FROM users GROUP BY gender", ENGINE).to_string(index=False))
print(pd.read_sql("SELECT credit_score IS NULL AS null_credit, COUNT(*) AS cnt FROM users GROUP BY credit_score IS NULL", ENGINE).to_string(index=False))

print("\n===== user_features 消费/行为列统计 =====")
print(pd.read_sql(
    "SELECT MIN(total_spent) AS mn_s, MAX(total_spent) AS mx_s, "
    "MIN(avg_order_amount) AS mn_a, MAX(avg_order_amount) AS mx_a, "
    "MIN(order_frequency) AS mn_f, MAX(order_frequency) AS mx_f, "
    "MIN(browse_count) AS mn_b, MAX(browse_count) AS mx_b, "
    "MIN(cart_count) AS mn_c, MAX(cart_count) AS mx_c "
    "FROM user_features", ENGINE).to_string(index=False))

print("\n===== user_behaviors 人均浏览时长可算性 =====")
print(pd.read_sql(
    "SELECT behavior_type, COUNT(*) AS n, AVG(duration_seconds) AS avg_dur, "
    "SUM(duration_seconds IS NULL) AS null_dur "
    "FROM user_behaviors GROUP BY behavior_type", ENGINE).to_string(index=False))

print("\n===== user_behaviors 每用户行为覆盖（与 user_features 对齐） =====")
print(pd.read_sql(
    "SELECT COUNT(DISTINCT user_id) AS beh_users FROM user_behaviors", ENGINE).to_string(index=False))

print("\n===== 复购组/非复购组的消费与会员交叉（快速预检） =====")
print(pd.read_sql(
    "SELECT uf.repurchase_indicator, COUNT(*) AS n, AVG(uf.total_spent) AS avg_spent, "
    "AVG(uf.order_frequency) AS avg_freq "
    "FROM user_features uf GROUP BY uf.repurchase_indicator", ENGINE).to_string(index=False))
