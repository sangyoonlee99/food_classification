import oracledb

def bind_raw16(v):
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)) and len(v) == 16:
        return oracledb.Binary(v)
    raise ValueError("user_id must be RAW(16)")
