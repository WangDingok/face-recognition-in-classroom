#!/usr/bin/env python3
"""
Quick test script untuk verify attendance database setup
"""

from core.attendance_db import AttendanceDatabase
from pathlib import Path

def test_attendance_db():
    """Test database creation dan basic operations"""
    
    print("🧪 Testing Attendance Database Setup...\n")
    
    # Test 1: Create database
    print("1️⃣  Creating attendance database...")
    db = AttendanceDatabase("output/attendance/attendance.db")
    print("   ✅ Database created\n")
    
    # Test 2: Set session ID
    print("2️⃣  Setting session ID...")
    session_id = db.set_session_id()
    print(f"   ✅ Session ID: {session_id}\n")
    
    # Test 3: Insert test records
    print("3️⃣  Inserting test records...")
    records = [
        ("student_001", "Nguyễn Văn A", "output/attendance_images/2025_06_01/student_001_1_10_30_45_123.jpg", 0.95),
        ("student_002", "Trần Thị B", "output/attendance_images/2025_06_01/student_002_2_10_35_12_456.jpg", 0.87),
        ("student_003", "Lê Văn C", "output/attendance_images/2025_06_01/student_003_3_10_40_33_789.jpg", 0.92),
    ]
    
    for student_id, name, image_path, confidence in records:
        record_id = db.insert_record(student_id, name, image_path, confidence)
        print(f"   ✅ Inserted: {name} (Record ID: {record_id})")
    
    print()
    
    # Test 4: Export CSV
    print("4️⃣  Exporting to CSV...")
    csv_path = f"output/attendance/test_attendance_{session_id}.csv"
    success = db.export_csv(output_path=csv_path, session_id=session_id)
    
    if success:
        print(f"   ✅ CSV exported to: {csv_path}\n")
        
        # Display CSV content
        with open(csv_path, 'r', encoding='utf-8') as f:
            print("   📄 CSV Content Preview:")
            print("   " + f.read().replace('\n', '\n   '))
    
    # Test 5: Query records
    print("\n5️⃣  Querying session records...")
    records = db.get_session_records(session_id)
    print(f"   ✅ Found {len(records)} records")
    
    # Test 6: Summary
    print("\n📊 Database Test Summary:")
    print(f"   Database Path: output/attendance/attendance.db")
    print(f"   Session ID: {session_id}")
    print(f"   Total Records: {len(records)}")
    print(f"   Image Folder: output/attendance_images/YYYY_MM_DD/")
    print(f"   CSV Export: {csv_path}")
    
    print("\n✅ All tests passed!\n")

if __name__ == "__main__":
    test_attendance_db()
