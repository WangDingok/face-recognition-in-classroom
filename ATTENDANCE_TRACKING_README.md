## 🎓 Face Recognition Attendance Tracking System

> Hệ thống tự động ghi nhận thời gian, danh tính học sinh, và ảnh khuôn mặt khi nhận diện

---

## 📋 Tính năng chính

✅ **Ghi thời gian tự động** - Timestamp `datetime.now()` khi detect lần đầu  
✅ **Database SQLite** - Lưu: session_id, timestamp, student_id, name, image_path, confidence  
✅ **Lưu ảnh crop** - Face crop images trong `output/attendance_images/YYYY_MM_DD/`  
✅ **Export CSV** - Tự động export sau video xong → `output/attendance/attendance_SESSION_ID.csv`  
✅ **Session tracking** - Phân biệt các buổi học bằng `session_id` (YYYY_MM_DD_HH_MM_SS)  
✅ **Deduplicate** - Mỗi học sinh chỉ được ghi nhận 1 lần per video  

---

## 📁 Cấu trúc folder

```
face_recognition_all/
├── main.py                          # ✅ Modified - thêm attendance tracking
├── core/
│   ├── attendance_db.py             # ✅ NEW - SQLite database module
│   ├── detector.py
│   ├── embedder.py
│   ├── recognizer.py
│   └── ...
├── test_attendance_db.py            # Test script
└── output/
    ├── attendance/
    │   ├── attendance.db            # SQLite database
    │   └── attendance_2026_06_01_12_03_11.csv  # Export CSV per session
    └── attendance_images/
        └── 2026_06_01/
            ├── student_001_1_10_30_45_123.jpg
            ├── student_002_2_10_35_12_456.jpg
            └── ...
```

---

## 🚀 Cách sử dụng

### 1. **Chạy video như bình thường**

```python
# main.py - không cần thay đổi gì
python main.py
```

### 2. **Kết quả sau video xong**

Hệ thống sẽ tự động:
- 💾 Lưu ảnh crop → `output/attendance_images/YYYY_MM_DD/`
- 📊 Lưu database → `output/attendance/attendance.db`
- 📄 Export CSV → `output/attendance/attendance_SESSION_ID.csv`

**Console output example:**
```
✅ Recorded: Nguyễn Văn A (ID: student_001) at 12:03:45
✅ Recorded: Trần Thị B (ID: student_002) at 12:05:12
✅ Recorded: Lê Văn C (ID: student_003) at 12:07:33

📊 Attendance Summary:
   Session: 2026_06_01_12_03_11
   CSV File: output/attendance/attendance_2026_06_01_12_03_11.csv
   Total Records: 3
   Unique Students: 3
```

### 3. **Mở CSV trong Excel**

```
Session ID          Timestamp              Student ID    Student Name      Image Path                    Confidence
2026_06_01_12_03_11 2026-06-01 12:03:45    student_001   Nguyễn Văn A      output/.../student_001_...    0.95
2026_06_01_12_03_11 2026-06-01 12:05:12    student_002   Trần Thị B        output/.../student_002_...    0.87
2026_06_01_12_03_11 2026-06-01 12:07:33    student_003   Lê Văn C          output/.../student_003_...    0.92
```

---

## 🛠️ API - Attendance Database Module

### **Import**
```python
from core.attendance_db import AttendanceDatabase

# Initialize (creates database if not exists)
db = AttendanceDatabase("output/attendance/attendance.db")

# Auto-generate session ID
db.set_session_id()  # Returns: "2026_06_01_12_03_11"
```

### **Main Methods**

#### 1. Insert Record
```python
record_id = db.insert_record(
    student_id="student_001",
    student_name="Nguyễn Văn A",
    image_path="output/attendance_images/2026_06_01/student_001_1_10_30_45_123.jpg",
    confidence=0.95
)
```

#### 2. Export CSV
```python
# Export specific session
db.export_csv(
    output_path="output/attendance/attendance_2026_06_01_12_03_11.csv",
    session_id="2026_06_01_12_03_11"
)

# Export all records
db.export_csv()
```

#### 3. Query Records
```python
# Get all records for session
records = db.get_session_records("2026_06_01_12_03_11")

# Get records for specific student
records = db.get_student_records("student_001", session_id="2026_06_01_12_03_11")
```

---

## 📊 Database Schema

```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,              -- Buổi học ID (YYYY_MM_DD_HH_MM_SS)
    timestamp TEXT NOT NULL,                -- Thời gian detect (YYYY-MM-DD HH:MM:SS.mmm)
    student_id TEXT NOT NULL,               -- ID từ recognizer label
    student_name TEXT NOT NULL,             -- Tên học sinh
    image_path TEXT,                        -- Đường dẫn ảnh crop
    confidence REAL,                        -- Confidence score (0.0-1.0)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_session_id ON attendance(session_id);
CREATE INDEX idx_student_id ON attendance(student_id);
```

---

## 🔍 Workflow Chi tiết

### **During Video Processing**

```
Frame 1: Detect face #1 (track_id=1)
  ↓
Frame 5: Recognize face #1 → label=student_001 (confidence=0.95)
  ↓
Check: student_001 not in prev_recorded_labels?
  ↓
  YES → Save:
    - Face crop → output/attendance_images/2026_06_01/student_001_1_10_30_45_123.jpg
    - DB record → INSERT (session_id, timestamp, student_001, Nguyễn Văn A, image_path, 0.95)
    - Add to prev_recorded_labels {student_001}

Frame 10: Detect same face #1 again (still track_id=1)
  ↓
Check: student_001 already in prev_recorded_labels?
  ↓
  YES → Skip (không lưu lại)
```

### **After Video Ends**

```
cap.release() → Export CSV:

output/attendance/attendance_2026_06_01_12_03_11.csv
```

---

## ⚙️ Configuration

### **Thay đổi database path** (nếu cần)
```python
processor = FaceRecognitionProcessor(
    ...existing params...,
    attendance_db_path="custom/path/attendance.db"  # Default: "output/attendance/attendance.db"
)
```

### **Thay đổi image quality** (trong `_save_face_image()`)
```python
# Hiện tại: 90% quality
cv2.imwrite(str(image_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

# Thay 90 thành số khác (0-100) để adjust quality
```

---

## ❌ Troubleshooting

### **Ảnh không được save**
- Check: Folder `output/attendance_images/` có quyền write?
- Check: Disk space đủ?

### **Database error**
- Check: File `output/attendance/attendance.db` không bị lock
- Solution: Delete file → sẽ tạo lại mới

### **CSV không export**
- Check: Folder `output/attendance/` tồn tại?
- Check: confidence >= sim_threshold (default 0.6)?

---

## 🧪 Test

**Run test script:**
```bash
python test_attendance_db.py
```

**Expected output:**
```
🧪 Testing Attendance Database Setup...

1️⃣  Creating attendance database...
   ✅ Database created

2️⃣  Setting session ID...
   ✅ Session ID: 2026_06_01_12_03_11

3️⃣  Inserting test records...
   ✅ Inserted: Nguyễn Văn A (Record ID: 1)
   ✅ Inserted: Trần Thị B (Record ID: 2)
   ✅ Inserted: Lê Văn C (Record ID: 3)

4️⃣  Exporting to CSV...
   ✅ CSV exported to: output/attendance/test_attendance_2026_06_01_12_03_11.csv

5️⃣  Querying session records...
   ✅ Found 3 records

✅ All tests passed!
```

---

## 📝 Summary

| Tính năng | Giải pháp | Lưu trữ |
|----------|---------|--------|
| **Thời gian** | `datetime.now()` on first detect | database: timestamp |
| **Danh tính** | Nhận diện label từ recognizer | database: student_id, student_name |
| **Ảnh** | Face crop JPEG 90% quality | folder: `output/attendance_images/YYYY_MM_DD/` |
| **Deduplicate** | `seen_labels` set per video | memory |
| **Session** | Auto-generate YYYY_MM_DD_HH_MM_SS | database: session_id |
| **Export** | CSV after video done | file: `attendance_SESSION_ID.csv` |

---

## 🎯 Next Steps (Optional)

1. **Multi-session handling**: Thêm UI để select session, xem records từ buổi học cũ
2. **Analytics**: Tạo dashboard để visualize attendance statistics
3. **Sync to server**: Upload database/CSV to cloud storage (Google Drive, etc.)
4. **Backup**: Auto-backup database every N videos
5. **Re-identification**: Track cùng 1 người qua nhiều video bằng embedding similarity

---

**Implementation Date:** June 1, 2026  
**Status:** ✅ Ready to Use
