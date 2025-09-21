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
        "database_path": "company_database.db",
        "docker_database_path": "/app/database_data/company_database.db"
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config
        except Exception as e:
            # print((f"Error loading config.json: {e}")
            return default_config
    return default_config


def get_database_path():
    """Determine the correct database path based on environment"""
    config = load_config()

    # Check if we're running in Docker by looking for docker-specific paths
    if os.path.exists("/app/database_data"):
        return config.get("docker_database_path", "/app/database_data/company_database.db")
    else:
        return config.get("database_path", "company_database.db")


class AsyncSQLiteServer:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or get_database_path()
        self.connection_semaphore = asyncio.Semaphore(20)
        print(f"Database path: {self.db_path}")
        print(f"Directory exists: {os.path.exists(os.path.dirname(self.db_path))}")
        print(f"Database file exists: {os.path.exists(self.db_path)}")

    async def initialize_database(self):
        """Initialize database with tables and sample data"""
        print(f"Initializing database at: {self.db_path}")
        
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

                # Drop existing tables if they exist
                drop_tables = [
                    "DROP TABLE IF EXISTS User_Teams",
                    "DROP TABLE IF EXISTS User_Roles",
                    "DROP TABLE IF EXISTS Role_Permissions",
                    "DROP TABLE IF EXISTS Permissions",
                    "DROP TABLE IF EXISTS Teams",
                    "DROP TABLE IF EXISTS Roles",
                    "DROP TABLE IF EXISTS Users",
                    "DROP TABLE IF EXISTS Departments"
                ]

                for drop_sql in drop_tables:
                    await db.execute(drop_sql)

                # Create tables
                create_tables = [
                    """CREATE TABLE IF NOT EXISTS Departments (
                        department_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        department_name VARCHAR(100),
                        manager_id INTEGER NULL
                    )""",

                    """CREATE TABLE IF NOT EXISTS Users (
                        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name VARCHAR(100),
                        email VARCHAR(100),
                        department_id INTEGER,
                        hire_date DATE,
                        FOREIGN KEY (department_id) REFERENCES Departments(department_id)
                    )""",

                    """CREATE TABLE IF NOT EXISTS Roles (
                        role_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        role_name VARCHAR(100),
                        description TEXT
                    )""",

                    """CREATE TABLE IF NOT EXISTS Permissions (
                        permission_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        permission_name VARCHAR(100),
                        description TEXT
                    )""",

                    """CREATE TABLE IF NOT EXISTS Role_Permissions (
                        role_id INTEGER,
                        permission_id INTEGER,
                        PRIMARY KEY (role_id, permission_id),
                        FOREIGN KEY (role_id) REFERENCES Roles(role_id),
                        FOREIGN KEY (permission_id) REFERENCES Permissions(permission_id)
                    )""",

                    """CREATE TABLE IF NOT EXISTS Teams (
                        team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        team_name VARCHAR(100),
                        department_id INTEGER,
                        FOREIGN KEY (department_id) REFERENCES Departments(department_id)
                    )""",

                    """CREATE TABLE IF NOT EXISTS User_Roles (
                        user_id INTEGER,
                        role_id INTEGER,
                        assigned_date DATE,
                        PRIMARY KEY (user_id, role_id),
                        FOREIGN KEY (user_id) REFERENCES Users(user_id),
                        FOREIGN KEY (role_id) REFERENCES Roles(role_id)
                    )""",

                    """CREATE TABLE IF NOT EXISTS User_Teams (
                        user_id INTEGER,
                        team_id INTEGER,
                        PRIMARY KEY (user_id, team_id),
                        FOREIGN KEY (user_id) REFERENCES Users(user_id),
                        FOREIGN KEY (team_id) REFERENCES Teams(team_id)
                    )"""
                ]

                for create_sql in create_tables:
                    await db.execute(create_sql)

                # Insert sample data
                print("Starting to insert sample data...")
                await self.insert_sample_data(db)
                print("Sample data insertion completed!")

                await db.commit()
                print("Database initialized successfully!")
                
                # Verify data was inserted
                cursor = await db.execute("SELECT COUNT(*) FROM Users")
                count = await cursor.fetchone()
                print(f"Total users inserted: {count[0] if count else 0}")

    async def insert_sample_data(self, db):
        """Insert all sample data"""

        # Insert Departments
        departments_data = [
            ('Human Resources', None),
            ('Engineering', None),
            ('Marketing', None),
            ('Sales', None),
            ('Finance', None),
            ('Operations', None),
            ('Customer Support', None)
        ]
        await db.executemany(
            "INSERT INTO Departments (department_name, manager_id) VALUES (?, ?)",
            departments_data
        )

        # Insert Users
        users_data = [
            ('Sarah Johnson', 'sarah.johnson@company.com', 1, '2022-03-15'),
            ('Michael Chen', 'michael.chen@company.com', 2, '2021-07-22'),
            ('Emily Rodriguez', 'emily.rodriguez@company.com', 3, '2023-01-10'),
            ('David Kim', 'david.kim@company.com', 2, '2020-11-08'),
            ('Lisa Wang', 'lisa.wang@company.com', 4, '2022-09-14'),
            ('James Thompson', 'james.thompson@company.com', 5, '2021-05-03'),
            ('Maria Garcia', 'maria.garcia@company.com', 1, '2023-02-28'),
            ('Robert Anderson', 'robert.anderson@company.com', 6, '2020-08-19'),
            ('Jennifer Liu', 'jennifer.liu@company.com', 2, '2022-12-07'),
            ('Thomas Brown', 'thomas.brown@company.com', 7, '2021-10-12'),
            ('Amanda Davis', 'amanda.davis@company.com', 3, '2023-04-05'),
            ('Kevin Wilson', 'kevin.wilson@company.com', 4, '2022-06-18'),
            ('Rachel Taylor', 'rachel.taylor@company.com', 5, '2021-12-01'),
            ('Daniel Martinez', 'daniel.martinez@company.com', 6, '2023-03-22'),
            ('Jessica Lee', 'jessica.lee@company.com', 7, '2022-08-11')
        ]
        await db.executemany(
            "INSERT INTO Users (name, email, department_id, hire_date) VALUES (?, ?, ?, ?)",
            users_data
        )

        # Update Departments with manager_id
        manager_updates = [
            (1, 1), (2, 2), (3, 3), (5, 4), (6, 5), (8, 6), (10, 7)
        ]
        for manager_id, dept_id in manager_updates:
            await db.execute(
                "UPDATE Departments SET manager_id = ? WHERE department_id = ?",
                (manager_id, dept_id)
            )

        # Insert Roles
        roles_data = [
            ('Administrator', 'Full system access and management capabilities'),
            ('Manager', 'Department or team management responsibilities'),
            ('Senior Developer', 'Advanced development skills with mentoring responsibilities'),
            ('Developer', 'Software development and maintenance'),
            ('Analyst', 'Data analysis and reporting'),
            ('Coordinator', 'Project coordination and administrative support'),
            ('Specialist', 'Subject matter expertise in specific domain'),
            ('Support Agent', 'Customer service and technical support'),
            ('Intern', 'Learning role with limited responsibilities'),
            ('Consultant', 'External advisor with specialized knowledge')
        ]
        await db.executemany(
            "INSERT INTO Roles (role_name, description) VALUES (?, ?)",
            roles_data
        )

        # Insert Permissions
        permissions_data = [
            ('user_management', 'Create, modify, and delete user accounts'),
            ('role_assignment', 'Assign and modify user roles'),
            ('system_config', 'Configure system settings and parameters'),
            ('data_export', 'Export data from the system'),
            ('financial_data', 'Access to financial information and reports'),
            ('employee_records', 'Access to employee personal information'),
            ('project_create', 'Create new projects and initiatives'),
            ('project_manage', 'Manage existing projects and teams'),
            ('customer_data', 'Access to customer information'),
            ('support_tickets', 'Manage customer support requests'),
            ('marketing_campaigns', 'Create and manage marketing initiatives'),
            ('sales_reports', 'Access to sales data and analytics'),
            ('inventory_management', 'Manage product inventory'),
            ('audit_logs', 'View system audit trails'),
            ('backup_restore', 'Perform system backups and restores')
        ]
        await db.executemany(
            "INSERT INTO Permissions (permission_name, description) VALUES (?, ?)",
            permissions_data
        )

        # Insert Role_Permissions
        role_permissions_data = [
            # Administrator - all permissions
            *[(1, i) for i in range(1, 16)],
            # Manager
            (2, 2), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (2, 12), (2, 14),
            # Senior Developer
            (3, 4), (3, 7), (3, 8), (3, 14),
            # Developer
            (4, 4), (4, 7), (4, 14),
            # Analyst
            (5, 4), (5, 5), (5, 12), (5, 14),
            # Coordinator
            (6, 4), (6, 7), (6, 8),
            # Specialist
            (7, 4), (7, 9), (7, 10), (7, 11), (7, 13),
            # Support Agent
            (8, 9), (8, 10),
            # Intern
            (9, 4),
            # Consultant
            (10, 4), (10, 12), (10, 14)
        ]
        await db.executemany(
            "INSERT INTO Role_Permissions (role_id, permission_id) VALUES (?, ?)",
            role_permissions_data
        )

        # Insert Teams
        teams_data = [
            ('Recruitment Team', 1), ('Employee Relations', 1),
            ('Backend Development', 2), ('Frontend Development', 2), ('DevOps Team', 2),
            ('Digital Marketing', 3), ('Content Creation', 3),
            ('Enterprise Sales', 4), ('Customer Success', 4),
            ('Accounting', 5), ('Financial Planning', 5),
            ('Supply Chain', 6), ('Quality Assurance', 6),
            ('Technical Support', 7), ('Customer Experience', 7)
        ]
        await db.executemany(
            "INSERT INTO Teams (team_name, department_id) VALUES (?, ?)",
            teams_data
        )

        # Insert User_Roles
        user_roles_data = [
            (1, 1, '2022-03-15'), (1, 2, '2022-03-15'),
            (2, 2, '2021-07-22'), (2, 3, '2021-07-22'),
            (3, 2, '2023-01-10'), (3, 7, '2023-01-10'),
            (4, 3, '2020-11-08'),
            (5, 2, '2022-09-14'), (5, 7, '2022-09-14'),
            (6, 2, '2021-05-03'), (6, 5, '2021-05-03'),
            (7, 6, '2023-02-28'),
            (8, 2, '2020-08-19'), (8, 7, '2020-08-19'),
            (9, 4, '2022-12-07'),
            (10, 2, '2021-10-12'), (10, 8, '2021-10-12'),
            (11, 7, '2023-04-05'),
            (12, 7, '2022-06-18'),
            (13, 5, '2021-12-01'),
            (14, 7, '2023-03-22'),
            (15, 8, '2022-08-11')
        ]
        await db.executemany(
            "INSERT INTO User_Roles (user_id, role_id, assigned_date) VALUES (?, ?, ?)",
            user_roles_data
        )

        # Insert User_Teams
        user_teams_data = [
            (1, 1), (1, 2), (7, 1),  # HR
            (2, 3), (2, 5), (4, 3), (9, 4),  # Engineering
            (3, 6), (3, 7), (11, 6),  # Marketing
            (5, 8), (5, 9), (12, 8),  # Sales
            (6, 10), (6, 11), (13, 11),  # Finance
            (8, 12), (8, 13), (14, 12),  # Operations
            (10, 14), (10, 15), (15, 14)  # Customer Support
        ]
        await db.executemany(
            "INSERT INTO User_Teams (user_id, team_id) VALUES (?, ?)",
            user_teams_data
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
app = FastAPI(title="Company Database API", description="AsyncSQLite Database Server for Company Management")
db_server = AsyncSQLiteServer()


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
async def startup_event():
    print("Starting database initialization...")
    await db_server.initialize_database()
    print("Database initialization completed!")
    
    # Test if data was inserted
    try:
        test_result = await db_server.execute_query("SELECT COUNT(*) as count FROM Users")
        print(f"Users count after initialization: {test_result}")
    except Exception as e:
        print(f"Error checking user count: {e}")


@app.post("/query")
async def execute_query(request: QueryRequest):
    """Execute SQL query and return results"""
    try:
        # print((f"Received request: {request.query}")

        results = await db_server.execute_query(request.query)

        # print((f"Query result: {results}")

        return {"success": True, "data": results}
    except Exception as e:
        # print((f"Query error: {str(e)}")
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
        test_query = "SELECT COUNT(*) as user_count FROM Users"
        result = await db_server.execute_query(test_query)
        user_count = result[0]['user_count'] if result else 0
        
        return {
            "status": "healthy", 
            "database_path": db_server.db_path,
            "user_count": user_count,
            "tables_initialized": user_count > 0
        }
    except Exception as e:
        return {
            "status": "error", 
            "database_path": db_server.db_path,
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8762)