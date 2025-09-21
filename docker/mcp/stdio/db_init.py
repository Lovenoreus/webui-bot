import sqlite3
import random

DB_PATH = "ad_users.db"

def normalize_to_set(value):
    if not value:
        return set()
    if isinstance(value, list):
        return set(value)
    if isinstance(value, str):
        return set(map(str.strip, value.split(",")))
    return set()


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            department TEXT,
            groups TEXT,
            permissions TEXT,
            role TEXT,
            UNIQUE(first_name, last_name)
        )
        """)

        conn.commit()

def save_user_to_db(user: dict, department: str = None, groups: list = None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO users (id, display_name, user_principal_name, department, groups)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                display_name = excluded.display_name,
                user_principal_name = excluded.user_principal_name,
                department = excluded.department,
                groups = excluded.groups
        """, (
            user["id"],
            user["displayName"],
            user["userPrincipalName"],
            department,
            ",".join(groups) if groups else None
        ))

def get_user_from_db_by_name(first_name, last_name):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT * FROM users WHERE first_name=? AND last_name=?", (first_name, last_name))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "first_name": row[1],
                "last_name": row[2],
                "status": row[3],
                "department": row[4],
                "groups": row[5].split(",") if row[4] else [],
                "permissions": row[6].split(",") if row[6] else [],
                "role": row[7]
            }
        return None

def upsert_user(first_name, last_name, department=None, groups=None, permissions=None, role=None, status=None):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # Check if user exists
            cursor.execute("SELECT id, department, groups, permissions, role, status FROM users WHERE first_name=? AND last_name=?", (first_name, last_name))
            existing = cursor.fetchone()

            if existing:
                user_id, curr_dept, curr_groups, curr_perms, curr_role, curr_status = existing

                # Convert to sets if values exist
                curr_dept_set = normalize_to_set(curr_dept)
                new_dept_set = normalize_to_set(department)
                department_merged = ",".join(sorted(curr_dept_set | new_dept_set))

                curr_perms_set = normalize_to_set(curr_perms)
                new_perms_set = normalize_to_set(permissions)
                permissions_merged = ",".join(sorted(curr_perms_set | new_perms_set))

                curr_groups_set = normalize_to_set(curr_groups)
                new_groups_set = normalize_to_set(groups)
                groups_merged = ",".join(sorted(curr_groups_set | new_groups_set))

                # Fallback to current values if not provided
                # department = department if department is not None else curr_dept
                role = role if role is not None else curr_role
                status = status if status is not None else curr_status

                cursor.execute("""
                               UPDATE users
                               SET department  = ?,
                                   groups      = ?,
                                   permissions = ?,
                                   role        = ?,
                                   status      = ?
                               WHERE id = ?
                               """, (department_merged, groups_merged, permissions_merged, role, status, user_id))


            else:
                # Default to 'active' if no status provided on insert
                status = status or 'active'
                department_str = ",".join(department) if isinstance(department, list) else department
                groups_str = ",".join(groups) if isinstance(groups, list) else groups
                permissions_str = ",".join(permissions) if isinstance(permissions, list) else permissions

                # Insert new user
                cursor.execute("""
                               INSERT INTO users (first_name, last_name, department, groups, permissions, role, status)
                               VALUES (?, ?, ?, ?, ?, ?, ?)
                               """, (first_name, last_name, department_str, groups_str, permissions_str, role, status))

                # cursor.execute("""
                #     INSERT INTO users (first_name, last_name, department, groups, permissions, role, status)
                #     VALUES (?, ?, ?, ?, ?, ?, ?)
                # """, (first_name, last_name, department, groups, permissions, role, status))

            conn.commit()
        return "Successfully upserted user."
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return "Failed to upsert user."

def view_users():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        users = []
        for row in rows:
            users.append({
                "id": row[0],
                "first_name": row[1],
                "last_name": row[2],
                "status": row[3],
                "department": row[4],
                "groups": row[5].split(",") if row[4] else [],
                "permissions": row[6].split(",") if row[6] else [],
                "role": row[7]
            })
        return users


def remove_user_fields(first_name: str, last_name: str, permissions: list = None, groups: list = None, remove_department: list = None):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # Get current user record
            cursor.execute("SELECT id, department, permissions, groups FROM users WHERE first_name=? AND last_name=?", (first_name, last_name))
            row = cursor.fetchone()
            if not row:
                return f"User {first_name} {last_name} not found."

            user_id, current_department, current_permissions, current_groups = row

            # Normalize and remove department
            if remove_department:
                existing = normalize_to_set(current_department)
                to_remove = normalize_to_set(remove_department)
                updated_departments = ",".join(sorted(existing - to_remove))
            else:
                updated_departments = current_department

            # Normalize and remove permissions
            if permissions:
                existing = normalize_to_set(current_permissions)
                to_remove = normalize_to_set(permissions)
                updated_permissions = ",".join(sorted(existing - to_remove))
            else:
                updated_permissions = current_permissions

            # Normalize and remove groups
            if groups:
                existing = normalize_to_set(current_groups)
                to_remove = normalize_to_set(groups)
                updated_groups = ",".join(sorted(existing - to_remove))
            else:
                updated_groups = current_groups


            cursor.execute("""
                UPDATE users SET department = ?, permissions = ?, groups = ?
                WHERE id = ?
            """, (updated_departments, updated_permissions, updated_groups, user_id))

            conn.commit()
            return f"{first_name} {last_name} has been updated successfully."

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return "Failed to update user."




roles = ['Administrator', 'Nurse', 'Technician', 'Receptionist', 'Surgeon']
departments = ['Cancer Centrum', 'Radiology', 'Cardiology']
permissions = ['Room Access', 'Equipment Access', 'Medication Access']
groups = [
    'Cancer Centrum Rooms Entry',
    'Cancer Centrum Changing Rooms',
    'Radiology Secure Room',
    'Cardiology Lab Entry'
]

first_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward', 'Fiona', 'George', 'Hannah', 'Ian', 'Jane']
last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Wilson', 'Clark']


def hard_reset_db():
    """Drops and recreates the `users` table."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS users")
        conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            department TEXT,
            groups TEXT,
            permissions TEXT,
            role TEXT,
            UNIQUE(first_name, last_name)
        )
        """)
        conn.commit()

def reset_ad():
    """Performs a hard reset and populates the database with 10 randomized users."""
    hard_reset_db()

    for i in range(10):
        first_name = first_names[i]
        last_name = last_names[i]

        department = random.choice(departments)
        role = random.choice(roles)

        # Pick 1–2 groups and permissions
        user_groups = ', '.join(random.sample(groups, k=random.randint(1, 2)))
        user_permissions = ', '.join(random.sample(permissions, k=random.randint(1, 2)))

        upsert_user(
            first_name=first_name,
            last_name=last_name,
            department=department,
            groups=user_groups,
            permissions=user_permissions,
            role=role
        )
    return "Database reset and populated with 10 sample users."
