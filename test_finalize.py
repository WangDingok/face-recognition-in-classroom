#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from core.attendance_db import AttendanceDatabase
from datetime import datetime as dt

# Load existing database
db = AttendanceDatabase()
session_id = db.session_id

print(f"📊 Testing finalize_duration for session: {session_id}")
print()

# Get all records before finalize
import sqlite3
conn = sqlite3.connect(db.db_path)
cursor = conn.cursor()
cursor.execute("SELECT id, student_name, first_seen_time, last_seen_time, duration_seconds FROM attendance ORDER BY id LIMIT 3")
print("Before finalize_duration:")
for row in cursor.fetchall():
    print(f"  {row}")
conn.close()

# Call finalize_duration
print("\n⏳ Calling finalize_duration()...")
updated = db.finalize_duration(session_id)
print(f"✅ Updated {updated} records")

# Check again
conn = sqlite3.connect(db.db_path)
cursor = conn.cursor()
cursor.execute("SELECT id, student_name, first_seen_time, last_seen_time, duration_seconds FROM attendance ORDER BY id LIMIT 3")
print("\nAfter finalize_duration:")
for row in cursor.fetchall():
    print(f"  {row}")
conn.close()
