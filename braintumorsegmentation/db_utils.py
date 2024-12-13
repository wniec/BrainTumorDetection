import sqlite3

class db_conn():
    def __init__(self):
        self.con = sqlite3.connect("patients.db")
        self.cur = self.con.cursor()
        # res = cur.execute("SELECT name FROM sqlite_master")
        # name = res.fetchall()
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS Patient (
            patient_id INTEGER PRIMARY KEY ,
            last_name varchar(255) NOT NULL,
            first_name varchar(255),
            birth_date DATE
        );
                    """)
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS Image (
            image_id INTEGER PRIMARY KEY,
            patient_id int NOT NULL,
            scan_date date,
            completed BOOL DEFAULT FALSE,
            danger FLOAT,
            FOREIGN KEY (Patient_id) REFERENCES Patient(Patient_id)
        );
                    """)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
    def close(self):
        self.con.close()
        
    def add_pacient(self, lastName, firstName=None, birth=None):
        res = self.cur.execute("""
              INSERT INTO Patient (patient_id, last_name, first_name, birth_date)
              VALUES(?,?,?,?)
                               """, (None, lastName, firstName, birth))
        self.con.commit()
        return self.cur.rowcount
        
    def add_imaging(self,pacient_id, danger, scan_date=None):
        res = self.cur.execute("""
              INSERT INTO Image (image_id, patient_id, scan_date, danger)
              VALUES(?,?,?,?)
                               """, (None, pacient_id, scan_date, danger))
        self.con.commit()
        return self.cur.rowcount
    
    #no lepiej by po image_id
    def set_pacient_completed(self, pacient_id, status=True):
        res = self.cur.execute("""
              UPDATE Image 
              SET completed = ?
              WHERE Patient_id = ?
                               """, (status, pacient_id))
        
    def get_top_not_completed_pacients_ordered(self):
        res = self.cur.execute("""
              SELECT Patient.patient_id, Patient.last_name, Patient.first_name, Image.danger
              FROM Image INNER JOIN Patient ON Patient.Patient_id = Image.Patient_id
              WHERE Image.Completed = FALSE
              ORDER BY Image.danger DESC
              LIMIT 5
                               """)
        return res.fetchall()
    
    # def get_all(self):
    #     res = self.cur.execute("""
    #           SELECT * FROM Image
    #                            """)
    #     return res.fetchall()
    
    # def get_cmp(self):
    #     res = self.cur.execute("""
    #           SELECT completed FROM Image
    #                            """)
    #     return res.fetchall()
    