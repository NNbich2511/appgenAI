import pyodbc

conn_str = (
    'DRIVER={ODBC Driver 18 for SQL Server};'
    'SERVER=localhost;'
    'DATABASE=master;'
    'Trusted_Connection=yes;'
    'TrustServerCertificate=yes;'
)

try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    print("Kết nối thành công!")
except Exception as e:
    print("Lỗi kết nối:", e)

# Kết nối
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Truy vấn tất cả dữ liệu từ bảng Users
cursor.execute("SELECT username, password, role, created_at, last_login FROM Users")

# Lấy tất cả kết quả
rows = cursor.fetchall()

# In dữ liệu
for row in rows:
    print(f"Username: {row.username},password: {row.password}, Role: {row.role}, Created: {row.created_at}, Last login: {row.last_login}")