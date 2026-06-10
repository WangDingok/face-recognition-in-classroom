import sqlite3
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


class AttendanceDatabase:
    """SQLite database untuk tracking kehadiran siswa dengan timestamp dan gambar"""
    
    def __init__(self, db_path: str = "output/attendance/attendance.db"):
        """
        Initialize attendance database
        
        Args:
            db_path: Path ke database file (akan create folder otomatis jika belum ada)
        """
        self.db_path = db_path
        self.session_id = None
        
        # Create folder jika belum ada
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.create_table()
    
    def create_table(self):
        """Create attendance table jika belum ada"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            first_seen_time TEXT NOT NULL,
            last_seen_time TEXT,
            duration_seconds INTEGER DEFAULT 0,
            student_id TEXT NOT NULL,
            student_name TEXT NOT NULL,
            image_path TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create index untuk query cepat
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_session_id ON attendance(session_id)
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_student_id ON attendance(student_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def set_session_id(self, session_id: Optional[str] = None):
        """
        Set session ID untuk tracking buổi học
        
        Args:
            session_id: Custom session ID, nếu None sẽ auto-generate
        """
        if session_id is None:
            # Auto-generate: YYYY_MM_DD_HH_MM_SS
            self.session_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        else:
            self.session_id = session_id
        
        return self.session_id
    
    def insert_record(self, 
                     student_id: str, 
                     student_name: str,
                     image_path: Optional[str] = None,
                     confidence: float = 0.0,
                     first_seen_time: Optional[str] = None) -> int:
        """
        Insert attendance record ke database
        
        Args:
            student_id: ID của học sinh (từ label resolver)
            student_name: Tên học sinh
            image_path: Path ke ảnh crop khuôn mặt
            confidence: Confidence score từ recognizer
            first_seen_time: Custom timestamp, nếu None sẽ dùng datetime.now()
        
        Returns:
            Record ID nếu successful, -1 nếu failed
        """
        if self.session_id is None:
            self.set_session_id()
        
        if first_seen_time is None:
            first_seen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO attendance 
            (session_id, first_seen_time, last_seen_time, student_id, student_name, image_path, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (self.session_id, first_seen_time, first_seen_time, student_id, student_name, image_path, confidence))
            
            conn.commit()
            record_id = cursor.lastrowid
            conn.close()
            
            return record_id
        
        except Exception as e:
            print(f"Error inserting record: {e}")
            return -1
    
    def update_last_seen_time(self, session_id: str, student_id: str, last_seen_time: Optional[str] = None) -> bool:
        """
        Update last_seen_time untuk student (call setiap frame student masih terlihat)
        
        Args:
            session_id: Session ID
            student_id: Student ID
            last_seen_time: Timestamp, nếu None sẽ dùng datetime.now()
        
        Returns:
            True jika successful
        """
        if last_seen_time is None:
            last_seen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE attendance
            SET last_seen_time = ?
            WHERE session_id = ? AND student_id = ?
            ''', (last_seen_time, session_id, student_id))
            
            conn.commit()
            conn.close()
            return cursor.rowcount > 0
        
        except Exception as e:
            print(f"Error updating last_seen_time: {e}")
            return False
    
    def finalize_duration(self, session_id: Optional[str] = None) -> int:
        """
        Calculate duration_seconds untuk semua records berdasarkan first_seen_time vs last_seen_time
        
        Args:
            session_id: Session ID, nếu None sẽ process semua records
        
        Returns:
            Number of records updated
        """
        try:
            from datetime import datetime as dt
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all records
            if session_id:
                cursor.execute('''
                SELECT id, first_seen_time, last_seen_time
                FROM attendance
                WHERE session_id = ?
                ''', (session_id,))
            else:
                cursor.execute('''
                SELECT id, first_seen_time, last_seen_time
                FROM attendance
                ''')
            
            rows = cursor.fetchall()
            updated = 0
            
            for record_id, first_time_str, last_time_str in rows:
                if first_time_str and last_time_str:
                    try:
                        first_time = dt.strptime(first_time_str, "%Y-%m-%d %H:%M:%S.%f")
                        last_time = dt.strptime(last_time_str, "%Y-%m-%d %H:%M:%S.%f")
                        duration_seconds = int((last_time - first_time).total_seconds())
                        
                        cursor.execute('''
                        UPDATE attendance
                        SET duration_seconds = ?
                        WHERE id = ?
                        ''', (duration_seconds, record_id))
                        
                        updated += 1
                    except Exception as e:
                        print(f"Error parsing times for record {record_id}: {e}")
            
            conn.commit()
            conn.close()
            return updated
        
        except Exception as e:
            print(f"Error finalizing duration: {e}")
            return 0
    
    def export_csv(self, output_path: Optional[str] = None, session_id: Optional[str] = None) -> bool:
        """
        Export attendance records ke CSV file
        
        Args:
            output_path: Path ke output CSV file. 
                        Nếu None: output/attendance/attendance_{session_id}.csv
            session_id: Filter export ke session cụ thể. 
                       Nếu None: export semua records
        
        Returns:
            True jika successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query records
            if session_id:
                cursor.execute('''
                SELECT session_id, first_seen_time, last_seen_time, duration_seconds, student_id, student_name, image_path, confidence
                FROM attendance
                WHERE session_id = ?
                ORDER BY first_seen_time
                ''', (session_id,))
            else:
                cursor.execute('''
                SELECT session_id, first_seen_time, last_seen_time, duration_seconds, student_id, student_name, image_path, confidence
                FROM attendance
                ORDER BY session_id, first_seen_time
                ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            # Determine output path
            if output_path is None:
                session_filter = session_id if session_id else "all"
                output_path = f"output/attendance/attendance_{session_filter}.csv"
            
            # Create folder jika belum ada
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Session ID', 'Time In', 'Time Out', 'Duration (seconds)', 'Duration (MM:SS)', 'Student ID', 'Student Name', 'Image Path', 'Confidence'])
                
                # Format rows dengan duration
                for row in rows:
                    session, first_time, last_time, duration_sec, student_id, name, img_path, conf = row
                    minutes = duration_sec // 60 if duration_sec else 0
                    seconds = duration_sec % 60 if duration_sec else 0
                    duration_formatted = f"{minutes}:{seconds:02d}"
                    writer.writerow([session, first_time, last_time, duration_sec, duration_formatted, student_id, name, img_path, conf])
            
            print(f"CSV exported: {output_path}")
            return True
        
        except Exception as e:
            print(f"Error exporting CSV: {e}")
            return False
    
    def get_session_records(self, session_id: str) -> list:
        """Get semua records untuk session tertentu"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT session_id, first_seen_time, last_seen_time, duration_seconds, student_id, student_name, image_path, confidence
            FROM attendance
            WHERE session_id = ?
            ORDER BY first_seen_time
            ''', (session_id,))
            
            rows = cursor.fetchall()
            conn.close()
            return rows
        
        except Exception as e:
            print(f"Error getting records: {e}")
            return []
    
    def get_student_records(self, student_id: str, session_id: Optional[str] = None) -> list:
        """Get records untuk student tertentu"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute('''
                SELECT session_id, first_seen_time, last_seen_time, duration_seconds, student_id, student_name, image_path, confidence
                FROM attendance
                WHERE student_id = ? AND session_id = ?
                ORDER BY first_seen_time
                ''', (student_id, session_id))
            else:
                cursor.execute('''
                SELECT session_id, first_seen_time, last_seen_time, duration_seconds, student_id, student_name, image_path, confidence
                FROM attendance
                WHERE student_id = ?
                ORDER BY first_seen_time
                ''', (student_id,))
            
            rows = cursor.fetchall()
            conn.close()
            return rows
        
        except Exception as e:
            print(f"Error getting student records: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        # SQLite auto-closes, tapi method ini untuk konsistensi
        pass
