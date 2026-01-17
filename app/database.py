import sqlite3, os, json
BASE = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE, '..', 'applicant_data_pro.db')

def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def create_db():
    conn = _get_conn()
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS applicants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        dob TEXT,
        address TEXT,
        phone TEXT,
        email TEXT,
        loan_amnt REAL,
        int_rate REAL,
        fico_low INTEGER,
        annual_inc REAL,
        guarantor_name TEXT,
        guarantor_relation TEXT,
        guarantor_phone TEXT,
        guarantor_address TEXT,
        insurance_opted INTEGER,
        insurance_premium REAL,
        documents TEXT,
        assistant_notes TEXT,
        base_prob REAL,
        adjusted_prob REAL,
        timestamp TEXT DEFAULT (datetime('now'))
    )
    ''')
    conn.commit()
    conn.close()

def insert_applicant(record):
    conn = _get_conn()
    c = conn.cursor()
    c.execute('''
    INSERT INTO applicants (
        name, dob, address, phone, email,
        loan_amnt, int_rate, fico_low, annual_inc,
        guarantor_name, guarantor_relation, guarantor_phone, guarantor_address,
        insurance_opted, insurance_premium, documents, assistant_notes,
        base_prob, adjusted_prob
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        record.get('name'), record.get('dob'), record.get('address'), record.get('phone'), record.get('email'),
        record.get('loan_amnt'), record.get('int_rate'), record.get('fico_low'), record.get('annual_inc'),
        record.get('guarantor_name'), record.get('guarantor_relation'), record.get('guarantor_phone'), record.get('guarantor_address'),
        1 if record.get('insurance_opted') else 0, record.get('insurance_premium'),
        record.get('documents_json'), record.get('assistant_notes'),
        record.get('base_prob'), record.get('adjusted_prob')
    ))
    conn.commit()
    conn.close()

def fetch_all_applicants():
    conn = _get_conn()
    df = None
    try:
        import pandas as pd
        df = pd.read_sql_query("SELECT * FROM applicants ORDER BY id DESC", conn)
    except Exception:
        df = None
    conn.close()
    return df