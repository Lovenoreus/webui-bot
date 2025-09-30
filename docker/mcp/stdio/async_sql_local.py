from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import uvicorn
from typing import List, Dict, Any
import aiosqlite
import json
import os


def load_config():
    """Load configuration from config.json if it exists"""
    config_path = "config.json"
    default_config = {
        "database_path": "healthcare_database.db",
        "docker_database_path": "/app/database_data/healthcare_database.db"
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config
        except Exception as e:
            return default_config
    return default_config


def get_database_path():
    """Determine the correct database path based on environment"""
    config = load_config()

    # Check if we're running in Docker by looking for docker-specific paths
    if os.path.exists("/app/database_data"):
        return config.get("docker_database_path", "/app/database_data/healthcare_database.db")
    else:
        return config.get("database_path", "healthcare_database.db")


class AsyncSQLiteServer:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or get_database_path()
        self.connection_semaphore = asyncio.Semaphore(20)
        print(f"Database path: {self.db_path}")
        print(f"Directory exists: {os.path.exists(os.path.dirname(self.db_path))}")
        print(f"Database file exists: {os.path.exists(self.db_path)}")

    async def initialize_database(self):
        """Initialize database with healthcare tables and sample data"""
        print(f"Initializing healthcare database at: {self.db_path}")

        # Ensure directory exists for docker path
        try:
            db_dir = os.path.dirname(self.db_path)
            print(f"Creating database directory: {db_dir}")
            os.makedirs(db_dir, exist_ok=True)
            print(f"Directory created successfully. Exists: {os.path.exists(db_dir)}")
        except Exception as e:
            print(f"Error creating directory: {e}")

        async with self.connection_semaphore:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA synchronous=NORMAL")
                await db.execute("PRAGMA cache_size=10000")
                await db.execute("PRAGMA temp_store=memory")
                await db.execute("PRAGMA foreign_keys=ON")

                # Drop any existing tables for clean slate
                drop_tables = [
                    "DROP TABLE IF EXISTS InsuranceCoverage",
                    "DROP TABLE IF EXISTS FacilityServices",
                    "DROP TABLE IF EXISTS MedicalInventory",
                    "DROP TABLE IF EXISTS LabTestReferenceRanges",
                    "DROP TABLE IF EXISTS InsuranceProviders",
                    "DROP TABLE IF EXISTS MedicalServicesCatalog",
                    "DROP TABLE IF EXISTS HealthcareFacilities"
                ]

                for drop_sql in drop_tables:
                    await db.execute(drop_sql)

                # Create healthcare tables
                create_tables = [
                    """CREATE TABLE HealthcareFacilities (
                        FacilityID INTEGER PRIMARY KEY AUTOINCREMENT,
                        Name VARCHAR(100) NOT NULL,
                        Type VARCHAR(50) NOT NULL,
                        Address TEXT,
                        City VARCHAR(50),
                        State VARCHAR(50),
                        Country VARCHAR(50),
                        LicenseNumber VARCHAR(50),
                        AccreditationStatus VARCHAR(50),
                        OperationalSince DATE,
                        IsActive BOOLEAN DEFAULT TRUE,
                        CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ModifiedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )""",

                    """CREATE TABLE MedicalServicesCatalog (
                        ServiceID INTEGER PRIMARY KEY AUTOINCREMENT,
                        ServiceName VARCHAR(100) NOT NULL,
                        ServiceCode VARCHAR(50) UNIQUE NOT NULL,
                        Department VARCHAR(50),
                        Description TEXT,
                        BasePrice DECIMAL(10,2),
                        RequiresAppointment BOOLEAN DEFAULT TRUE,
                        IsActive BOOLEAN DEFAULT TRUE,
                        CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ModifiedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )""",

                    """CREATE TABLE LabTestReferenceRanges (
                        RangeID INTEGER PRIMARY KEY AUTOINCREMENT,
                        TestName VARCHAR(100) NOT NULL,
                        ServiceID INTEGER,
                        Unit VARCHAR(20),
                        Gender VARCHAR(10),
                        AgeMin INTEGER,
                        AgeMax INTEGER,
                        MinValue DECIMAL(10,2),
                        MaxValue DECIMAL(10,2),
                        Notes TEXT,
                        CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ModifiedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (ServiceID) REFERENCES MedicalServicesCatalog(ServiceID)
                    )""",

                    """CREATE TABLE MedicalInventory (
                        InventoryID INTEGER PRIMARY KEY AUTOINCREMENT,
                        ItemName VARCHAR(100) NOT NULL,
                        Category VARCHAR(50),
                        Quantity INTEGER DEFAULT 0,
                        Unit VARCHAR(20),
                        FacilityID INTEGER NOT NULL,
                        ReorderThreshold INTEGER DEFAULT 10,
                        ExpiryDate DATE,
                        LastUpdated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (FacilityID) REFERENCES HealthcareFacilities(FacilityID)
                    )""",

                    """CREATE TABLE InsuranceProviders (
                        ProviderID INTEGER PRIMARY KEY AUTOINCREMENT,
                        ProviderName VARCHAR(100) NOT NULL,
                        ContactEmail VARCHAR(100),
                        ContactPhone VARCHAR(20),
                        Address TEXT,
                        ServicesCovered TEXT,
                        Country VARCHAR(50),
                        ContractStart DATE,
                        ContractEnd DATE,
                        IsActive BOOLEAN DEFAULT TRUE,
                        CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ModifiedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )""",

                    """CREATE TABLE FacilityServices (
                        FacilityServiceID INTEGER PRIMARY KEY AUTOINCREMENT,
                        FacilityID INTEGER NOT NULL,
                        ServiceID INTEGER NOT NULL,
                        IsAvailable BOOLEAN DEFAULT TRUE,
                        EffectiveDate DATE DEFAULT CURRENT_DATE,
                        CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (FacilityID) REFERENCES HealthcareFacilities(FacilityID),
                        FOREIGN KEY (ServiceID) REFERENCES MedicalServicesCatalog(ServiceID),
                        UNIQUE(FacilityID, ServiceID)
                    )""",

                    """CREATE TABLE InsuranceCoverage (
                        CoverageID INTEGER PRIMARY KEY AUTOINCREMENT,
                        ProviderID INTEGER NOT NULL,
                        ServiceID INTEGER NOT NULL,
                        CoveragePercentage DECIMAL(5,2) DEFAULT 0.00,
                        Deductible DECIMAL(10,2) DEFAULT 0.00,
                        IsActive BOOLEAN DEFAULT TRUE,
                        CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ModifiedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (ProviderID) REFERENCES InsuranceProviders(ProviderID),
                        FOREIGN KEY (ServiceID) REFERENCES MedicalServicesCatalog(ServiceID),
                        UNIQUE(ProviderID, ServiceID)
                    )"""
                ]

                for create_sql in create_tables:
                    await db.execute(create_sql)

                # Create indexes for better performance
                indexes = [
                    "CREATE INDEX idx_facilities_type ON HealthcareFacilities(Type)",
                    "CREATE INDEX idx_services_code ON MedicalServicesCatalog(ServiceCode)",
                    "CREATE INDEX idx_inventory_facility ON MedicalInventory(FacilityID)",
                    "CREATE INDEX idx_inventory_expiry ON MedicalInventory(ExpiryDate)",
                    "CREATE INDEX idx_lab_ranges_test ON LabTestReferenceRanges(TestName)"
                ]

                for index_sql in indexes:
                    await db.execute(index_sql)

                # Insert sample data
                print("Starting to insert healthcare sample data...")
                await self.insert_sample_data(db)
                print("Healthcare sample data insertion completed!")

                await db.commit()
                print("Healthcare database initialized successfully!")

                # Verify data was inserted
                cursor = await db.execute("SELECT COUNT(*) FROM HealthcareFacilities")
                count = await cursor.fetchone()
                print(f"Total healthcare facilities inserted: {count[0] if count else 0}")

    async def insert_sample_data(self, db):
        """Insert comprehensive healthcare sample data"""

        # Insert Healthcare Facilities
        facilities_data = [
            ('Metro General Hospital', 'Hospital', '123 Main St', 'New York', 'NY', 'USA', 'LIC-H001',
             'Joint Commission Accredited', '2010-05-15', True),
            ('Westside Medical Center', 'Medical Center', '456 Oak Ave', 'Los Angeles', 'CA', 'USA', 'LIC-MC002',
             'AAAHC Accredited', '2015-03-22', True),
            ('Downtown Primary Clinic', 'Clinic', '789 Pine Rd', 'Chicago', 'IL', 'USA', 'LIC-C003', 'NCQA Certified',
             '2018-07-10', True),
            ('Riverside Laboratory Services', 'Laboratory', '321 River St', 'Miami', 'FL', 'USA', 'LIC-L004',
             'CLIA Certified', '2012-11-08', True),
            ('Central Imaging Center', 'Imaging Center', '654 Center Blvd', 'Houston', 'TX', 'USA', 'LIC-I005',
             'ACR Accredited', '2016-09-30', True),
            ('Northside Urgent Care', 'Urgent Care', '987 North Ave', 'Phoenix', 'AZ', 'USA', 'LIC-UC006',
             'AAAHC Accredited', '2019-01-12', True),
            ('Eastside Specialty Clinic', 'Specialty Clinic', '147 East St', 'Dallas', 'TX', 'USA', 'LIC-SC007',
             'Joint Commission Accredited', '2017-08-25', True)
        ]
        await db.executemany(
            """INSERT INTO HealthcareFacilities 
               (Name, Type, Address, City, State, Country, LicenseNumber, AccreditationStatus, OperationalSince, IsActive) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            facilities_data
        )

        # Insert Medical Services Catalog
        services_data = [
            ('Blood Chemistry Panel', 'LAB001', 'Laboratory',
             'Comprehensive metabolic panel including glucose, electrolytes, kidney function', 85.00, True, True),
            ('Complete Blood Count', 'LAB002', 'Laboratory', 'CBC with differential and platelet count', 45.00, True,
             True),
            ('Lipid Panel', 'LAB003', 'Laboratory', 'Total cholesterol, HDL, LDL, triglycerides', 65.00, True, True),
            ('Thyroid Function Test', 'LAB004', 'Laboratory', 'TSH, T3, T4 levels', 95.00, True, True),
            ('Chest X-Ray', 'RAD001', 'Radiology', 'Posterior-anterior and lateral chest radiograph', 120.00, True,
             True),
            ('MRI Brain', 'RAD002', 'Radiology', 'Magnetic resonance imaging of brain with contrast', 1200.00, True,
             True),
            ('CT Scan Abdomen', 'RAD003', 'Radiology', 'Computed tomography of abdomen and pelvis', 800.00, True, True),
            ('Mammogram', 'RAD004', 'Radiology', 'Digital screening mammography bilateral', 180.00, True, True),
            ('Ultrasound Abdomen', 'RAD005', 'Radiology', 'Abdominal ultrasound examination', 250.00, True, True),
            ('Annual Physical Exam', 'PREV001', 'Primary Care', 'Comprehensive annual wellness examination', 200.00,
             True, True),
            ('Vaccination Service', 'PREV002', 'Primary Care', 'Immunization administration', 25.00, True, True),
            ('Echocardiogram', 'CARD001', 'Cardiology', 'Transthoracic echocardiogram with Doppler', 350.00, True,
             True),
            ('Stress Test', 'CARD002', 'Cardiology', 'Exercise stress test with EKG monitoring', 400.00, True, True),
            ('Colonoscopy', 'GI001', 'Gastroenterology', 'Diagnostic colonoscopy with biopsy capability', 800.00, True,
             True),
            ('Endoscopy', 'GI002', 'Gastroenterology', 'Upper endoscopy examination', 600.00, True, True)
        ]
        await db.executemany(
            """INSERT INTO MedicalServicesCatalog 
               (ServiceName, ServiceCode, Department, Description, BasePrice, RequiresAppointment, IsActive) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            services_data
        )

        # Insert Lab Test Reference Ranges
        lab_ranges_data = [
            ('Glucose', 1, 'mg/dL', 'All', 18, 99, 70.0, 99.0, 'Fasting glucose normal range'),
            ('Glucose', 1, 'mg/dL', 'All', 65, 99, 70.0, 126.0, 'Fasting glucose normal range for elderly'),
            ('Hemoglobin', 2, 'g/dL', 'Male', 18, 99, 13.8, 17.2, 'Normal range for adult males'),
            ('Hemoglobin', 2, 'g/dL', 'Female', 18, 99, 12.1, 15.1, 'Normal range for adult females'),
            ('Total Cholesterol', 3, 'mg/dL', 'All', 18, 99, 0.0, 200.0, 'Desirable level'),
            ('HDL Cholesterol', 3, 'mg/dL', 'Male', 18, 99, 40.0, 999.0, 'Good HDL for males'),
            ('HDL Cholesterol', 3, 'mg/dL', 'Female', 18, 99, 50.0, 999.0, 'Good HDL for females'),
            ('LDL Cholesterol', 3, 'mg/dL', 'All', 18, 99, 0.0, 100.0, 'Optimal LDL level'),
            ('TSH', 4, 'mIU/L', 'All', 18, 99, 0.4, 4.0, 'Normal thyroid stimulating hormone'),
            ('Platelet Count', 2, 'K/uL', 'All', 18, 99, 150.0, 450.0, 'Normal platelet range')
        ]
        await db.executemany(
            """INSERT INTO LabTestReferenceRanges 
               (TestName, ServiceID, Unit, Gender, AgeMin, AgeMax, MinValue, MaxValue, Notes) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            lab_ranges_data
        )

        # Insert Medical Inventory
        inventory_data = [
            ('Disposable Gloves', 'PPE', 5000, 'boxes', 1, 500, '2025-12-31'),
            ('N95 Masks', 'PPE', 2000, 'pieces', 1, 200, '2026-06-30'),
            ('Surgical Masks', 'PPE', 10000, 'pieces', 1, 1000, '2026-03-15'),
            ('Blood Collection Tubes', 'Laboratory Supplies', 1500, 'units', 4, 100, '2025-09-30'),
            ('Contrast Dye', 'Imaging Supplies', 50, 'vials', 5, 10, '2025-11-20'),
            ('Stethoscopes', 'Medical Equipment', 25, 'units', 2, 5, None),
            ('Blood Pressure Monitors', 'Medical Equipment', 15, 'units', 3, 3, None),
            ('Ultrasound Gel', 'Imaging Supplies', 200, 'bottles', 5, 20, '2026-01-15'),
            ('Syringes (10ml)', 'Medical Supplies', 3000, 'units', 1, 300, '2027-05-10'),
            ('Gauze Pads', 'Medical Supplies', 800, 'packages', 2, 50, '2028-02-28'),
            ('Thermometer Covers', 'Medical Supplies', 5000, 'units', 3, 500, '2026-08-12'),
            ('X-Ray Film', 'Imaging Supplies', 100, 'sheets', 5, 20, '2025-10-31')
        ]
        await db.executemany(
            """INSERT INTO MedicalInventory 
               (ItemName, Category, Quantity, Unit, FacilityID, ReorderThreshold, ExpiryDate) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            inventory_data
        )

        # Insert Insurance Providers
        insurance_data = [
            ('Blue Cross Blue Shield', 'info@bcbs.com', '1-800-BCBS-123', '100 Insurance Way, Chicago, IL',
             'Primary Care,Laboratory,Radiology,Cardiology', 'USA', '2023-01-01', '2025-12-31', True),
            ('Aetna Healthcare', 'contact@aetna.com', '1-800-AETNA-01', '200 Health Ave, Hartford, CT',
             'Primary Care,Specialty Care,Laboratory', 'USA', '2023-06-01', '2026-05-31', True),
            ('Cigna Health', 'support@cigna.com', '1-800-CIGNA-24', '300 Wellness Blvd, Philadelphia, PA',
             'Primary Care,Radiology,Gastroenterology', 'USA', '2023-03-01', '2025-02-28', True),
            ('United HealthCare', 'help@uhc.com', '1-800-UHC-CARE', '400 Coverage St, Minneapolis, MN',
             'Primary Care,Laboratory,Cardiology,Radiology', 'USA', '2022-01-01', '2024-12-31', True),
            ('Kaiser Permanente', 'member@kp.org', '1-800-KAISER-1', '500 Integrated Dr, Oakland, CA',
             'Primary Care,Laboratory,Radiology,Specialty Care', 'USA', '2023-01-01', '2026-12-31', True)
        ]
        await db.executemany(
            """INSERT INTO InsuranceProviders 
               (ProviderName, ContactEmail, ContactPhone, Address, ServicesCovered, Country, ContractStart, ContractEnd, IsActive) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            insurance_data
        )

        # Insert Facility Services (which services are available at which facilities)
        facility_services_data = [
            # Metro General Hospital (1) - Full service hospital
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13),
            (1, 14), (1, 15),
            # Westside Medical Center (2) - Medical center
            (2, 1), (2, 2), (2, 3), (2, 5), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12),
            # Downtown Primary Clinic (3) - Primary care focus
            (3, 10), (3, 11), (3, 1), (3, 2), (3, 3),
            # Riverside Laboratory Services (4) - Lab only
            (4, 1), (4, 2), (4, 3), (4, 4),
            # Central Imaging Center (5) - Imaging only
            (5, 5), (5, 6), (5, 7), (5, 8), (5, 9),
            # Northside Urgent Care (6) - Urgent care services
            (6, 1), (6, 2), (6, 5), (6, 10), (6, 11),
            # Eastside Specialty Clinic (7) - Specialty services
            (7, 12), (7, 13), (7, 14), (7, 15)
        ]
        await db.executemany(
            """INSERT INTO FacilityServices (FacilityID, ServiceID) VALUES (?, ?)""",
            facility_services_data
        )

        # Insert Insurance Coverage
        coverage_data = [
            # Blue Cross Blue Shield coverage
            (1, 1, 80.00, 50.00), (1, 2, 80.00, 25.00), (1, 5, 70.00, 100.00), (1, 10, 90.00, 20.00),
            (1, 12, 75.00, 200.00),
            # Aetna Healthcare coverage
            (2, 1, 75.00, 75.00), (2, 2, 75.00, 30.00), (2, 10, 85.00, 25.00), (2, 14, 60.00, 500.00),
            # Cigna Health coverage
            (3, 1, 70.00, 60.00), (3, 5, 65.00, 150.00), (3, 10, 80.00, 30.00), (3, 14, 55.00, 400.00),
            # United HealthCare coverage
            (4, 1, 85.00, 40.00), (4, 2, 85.00, 20.00), (4, 5, 75.00, 75.00), (4, 12, 80.00, 150.00),
            # Kaiser Permanente coverage
            (5, 1, 90.00, 20.00), (5, 2, 90.00, 15.00), (5, 5, 85.00, 50.00), (5, 10, 95.00, 10.00),
            (5, 12, 85.00, 100.00)
        ]
        await db.executemany(
            """INSERT INTO InsuranceCoverage (ProviderID, ServiceID, CoveragePercentage, Deductible) VALUES (?, ?, ?, ?)""",
            coverage_data
        )

    async def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute any SQL query and return results"""
        async with self.connection_semaphore:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys=ON")
                db.row_factory = aiosqlite.Row
                async with db.execute(query) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]

    async def execute_query_stream(self, query: str):
        """Execute SQL query and yield results one row at a time"""
        async with self.connection_semaphore:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys=ON")
                db.row_factory = aiosqlite.Row
                async with db.execute(query) as cursor:
                    async for row in cursor:
                        yield dict(row)


# FastAPI setup
app = FastAPI(title="Healthcare Database API", description="AsyncSQLite Database Server for Healthcare Management")
db_server = AsyncSQLiteServer()


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
async def startup_event():
    print("Starting healthcare database initialization...")
    await db_server.initialize_database()
    print("Healthcare database initialization completed!")

    # Test if data was inserted
    try:
        test_result = await db_server.execute_query("SELECT COUNT(*) as count FROM HealthcareFacilities")
        print(f"Healthcare facilities count after initialization: {test_result}")

        # Show sample of what's in the database
        services_result = await db_server.execute_query("SELECT COUNT(*) as count FROM MedicalServicesCatalog")
        print(f"Medical services count: {services_result}")

        inventory_result = await db_server.execute_query("SELECT COUNT(*) as count FROM MedicalInventory")
        print(f"Inventory items count: {inventory_result}")

    except Exception as e:
        print(f"Error checking database counts: {e}")


@app.post("/query")
async def execute_query(request: QueryRequest):
    """Execute SQL query and return results"""
    try:
        results = await db_server.execute_query(request.query)
        return {"success": True, "data": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/query_stream")
async def execute_query_stream(request: QueryRequest):
    """Stream SQL query results one row at a time"""

    async def generate_results():
        try:
            yield json.dumps({"type": "start", "query": request.query}) + "\n"

            count = 0
            async for row in db_server.execute_query_stream(request.query):
                count += 1
                result = {
                    "type": "row",
                    "data": row,
                    "index": count
                }
                yield json.dumps(result) + "\n"

            yield json.dumps({"type": "complete", "total_rows": count}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"

    return StreamingResponse(
        generate_results(),
        media_type="application/x-ndjson"
    )


@app.get("/health")
async def health_check():
    try:
        # Test if database is accessible and has data
        test_query = "SELECT COUNT(*) as facility_count FROM HealthcareFacilities"
        result = await db_server.execute_query(test_query)
        facility_count = result[0]['facility_count'] if result else 0

        # Get additional counts for health check
        services_result = await db_server.execute_query("SELECT COUNT(*) as service_count FROM MedicalServicesCatalog")
        service_count = services_result[0]['service_count'] if services_result else 0

        return {
            "status": "healthy",
            "database_path": db_server.db_path,
            "facility_count": facility_count,
            "service_count": service_count,
            "tables_initialized": facility_count > 0 and service_count > 0
        }
    except Exception as e:
        return {
            "status": "error",
            "database_path": db_server.db_path,
            "error": str(e)
        }


# Additional healthcare-specific endpoints
@app.get("/facilities")
async def get_facilities():
    """Get all healthcare facilities"""
    try:
        query = """
        SELECT FacilityID, Name, Type, City, State, AccreditationStatus, IsActive 
        FROM HealthcareFacilities 
        WHERE IsActive = 1
        ORDER BY Name
        """
        results = await db_server.execute_query(query)
        return {"success": True, "data": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/services")
async def get_services():
    """Get all medical services"""
    try:
        query = """
        SELECT ServiceID, ServiceName, ServiceCode, Department, BasePrice, IsActive 
        FROM MedicalServicesCatalog 
        WHERE IsActive = 1
        ORDER BY Department, ServiceName
        """
        results = await db_server.execute_query(query)
        return {"success": True, "data": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/inventory/low-stock")
async def get_low_stock_items():
    """Get inventory items that are below reorder threshold"""
    try:
        query = """
        SELECT i.ItemName, i.Category, i.Quantity, i.ReorderThreshold, 
               f.Name as FacilityName, i.ExpiryDate
        FROM MedicalInventory i
        JOIN HealthcareFacilities f ON i.FacilityID = f.FacilityID
        WHERE i.Quantity <= i.ReorderThreshold
        ORDER BY i.Quantity ASC
        """
        results = await db_server.execute_query(query)
        return {"success": True, "data": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8762)