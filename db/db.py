import sqlite3
import json
from pathlib import Path

DB_PATH = Path("/home/ubuntu/project/IMU-DL-Terrain-Classification/results/experiments.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            exp_name    TEXT,
            model       TEXT,
            input_mode  TEXT,
            feat_combo  TEXT,
            kfold       INTEGER,
            acc         REAL,
            macro_f1    REAL,
            per_class_recall TEXT,
            folds       TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_result(result: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO results
        (timestamp, exp_name, model, input_mode, feat_combo, kfold, acc, macro_f1, per_class_recall, folds)
        VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.get("exp_name"),
        result.get("model"),
        result.get("input_mode"),
        json.dumps(result.get("feat_combo", [])),
        result.get("kfold", 5),
        result.get("acc"),
        result.get("macro_f1"),
        json.dumps(result.get("per_class_recall", {})),
        json.dumps(result.get("folds", [])),
    ))
    conn.commit()
    conn.close()

def query_best(metric="macro_f1", top_n=10):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(f"""
        SELECT exp_name, model, input_mode, feat_combo, acc, macro_f1
        FROM results
        ORDER BY {metric} DESC
        LIMIT {top_n}
    """).fetchall()
    conn.close()
    return rows
