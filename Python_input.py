import os

import pandas as pd
from sqlalchemy import create_engine, inspect, text

# ============ 数据库配置区，自行修改 ============
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PWD = "          "
MYSQL_DB = "taobao_data"

# 你的建表 SQL 文件路径（列名以它为准）
SQL_CODE_PATH = r"D:\Users\new\PycharmProjects\taobao_data_analysis\sql代码"
# CSV 数据目录
BASE_PATH = r"D:\Users\new\PycharmProjects\taobao_data_analysis\taobao_data"

# True  = 先执行 sql代码（会 DROP 并重建全部表，清掉旧数据，含上次半导入的 orders）再导入
# False = 跳过建表，直接向现有表追加（前提：你已手动执行过建表 SQL）
EXECUTE_SQL_FIRST = False

# 导入顺序：外键父表(users/products) 必须排在子表之前，否则会报外键 1452
IMPORT_ORDER = [
    ("users.csv", "users"),
    ("products.csv", "products"),
    ("user_features.csv", "user_features"),       # 外键 -> users
    ("product_features.csv", "product_features"), # 外键 -> products
    ("user_behaviors.csv", "user_behaviors"),
    ("orders.csv", "orders"),
]

# CSV 中文列名 -> SQL 英文字段（仅 product_features 存在这 4 处差异，其余文件列名一致）
COLUMN_RENAME = {
    "加购_count": "cart_count",
    "收藏_count": "collect_count",
    "浏览_count": "browse_count",
    "点击_count": "click_count",
}
# ==============================================


def make_engine(with_db: bool):
    """with_db=True 连接目标库；False 连接服务器（用于先建库）。"""
    db_part = f"/{MYSQL_DB}" if with_db else ""
    conn_url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PWD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}{db_part}?charset=utf8mb4"
    )
    return create_engine(conn_url)


def run_sql_schema():
    """在服务器连接上按顺序执行 sql代码 全部语句（含建库、USE、删表重建）。"""
    engine = make_engine(with_db=False)
    with open(SQL_CODE_PATH, encoding="utf-8") as f:
        text_content = f.read()
    # 去掉 -- 注释行
    lines = [ln for ln in text_content.splitlines() if not ln.strip().startswith("--")]
    statements = [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
    # 同一连接内顺序执行，USE 会作用于后续 CREATE TABLE
    with engine.begin() as conn:
        for stmt in statements:
            print("执行 SQL:", stmt.splitlines()[0][:60])
            conn.execute(text(stmt))
    engine.dispose()


def validate_columns(engine, table_name, df_columns):
    """导入前校验：CSV(已重命名) 的列必须都能在目标表中找到，避免再次报 1054。"""
    try:
        table_cols = {c["name"] for c in inspect(engine).get_columns(table_name)}
    except Exception:
        raise RuntimeError(
            f"表 {table_name} 不存在，请先执行建表 SQL，或把 EXECUTE_SQL_FIRST 设为 True"
        )
    missing = [c for c in df_columns if c not in table_cols]
    if missing:
        raise ValueError(f"表 {table_name} 缺少列 {missing}，请检查列名映射")
    return True


def import_csv(engine, file_name, table_name):
    full_file = os.path.join(BASE_PATH, file_name)
    print(f"正在导入：{file_name} -> 表 {table_name}")
    df = pd.read_csv(full_file, encoding="utf-8")
    df = df.rename(columns=COLUMN_RENAME)  # 中文列名 -> SQL 英文字段
    validate_columns(engine, table_name, df.columns.tolist())
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="append",
        index=False,
        chunksize=2000,
    )
    print(f"{file_name} 导入成功！\n")


if __name__ == "__main__":
    if EXECUTE_SQL_FIRST:
        # 注意：这会清空重建全部表，含之前半导入的 orders，确认后再运行本脚本
        run_sql_schema()
    engine = make_engine(with_db=True)
    for csv_file, table in IMPORT_ORDER:
        import_csv(engine, csv_file, table)
    print("全部 6 个 CSV 文件导入 MySQL 完成！")

