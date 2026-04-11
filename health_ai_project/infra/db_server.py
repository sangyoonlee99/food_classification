# infra/db_server.py
import os
import oracledb
from contextlib import contextmanager
import streamlit as st

LIB_DIR = os.getenv("ORACLE_CLIENT_LIB_DIR", r"C:\util\instantclient-basic-windows.x64-23.26.0.0.0\instantclient_23_0")
DSN     = os.getenv("ORACLE_DSN", "localhost:1521/ai3db")
USER    = os.getenv("ORACLE_USER", "health_user")
PW      = os.getenv("ORACLE_PASSWORD", "pass")

oracledb.init_oracle_client(lib_dir=LIB_DIR)

@st.cache_resource(show_spinner=False)
def _get_cached_conn():
    conn = oracledb.connect(user=USER, password=PW, dsn=DSN)
    conn.autocommit = False
    return conn

@contextmanager
def get_db_conn():
    conn = _get_cached_conn()
    try:
        yield conn
    finally:
        pass