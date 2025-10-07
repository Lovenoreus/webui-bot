import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    "host": "hh-pgsql-public.ebi.ac.uk",
    "port": 5432,
    "database": "pfmegrnargs",
    "user": "reader",
    "password": "NWDMCE5xdipIjRrp"
}

def run_query(query):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                result = cur.fetchall()
                return result
    except Exception as e:
        print("❌ Error:", e)
        return None


if __name__ == "__main__":
    print("✅ Connected successfully!\n")

    queries = [
        # 1️⃣ Show all tables under rnacen schema
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'rnacen';",

        # 2️⃣ Preview data from one table
        "SELECT * FROM rnacen.protein_info LIMIT 5;",

        # 3️⃣ Count total rows in that table
        "SELECT COUNT(*) AS total_rows FROM rnacen.protein_info;",

        # 4️⃣ List unique protein accessions or IDs if column exists
        "SELECT DISTINCT accession FROM rnacen.protein_info LIMIT 10;"
    ]

    for q in queries:
        print(f"\n🧠 Running query:\n{q}\n")
        result = run_query(q)
        if result:
            for row in result[:10]:
                print(row)
        else:
            print("No result or query failed.")


# import psycopg2
# from psycopg2.extras import RealDictCursor

# DB_CONFIG = {
#     "host": "hh-pgsql-public.ebi.ac.uk",
#     "port": 5432,
#     "database": "pfmegrnargs",
#     "user": "reader",
#     "password": "NWDMCE5xdipIjRrp"
# }

# def run_query(query):
#     try:
#         with psycopg2.connect(**DB_CONFIG) as conn:
#             with conn.cursor(cursor_factory=RealDictCursor) as cur:
#                 cur.execute(query)
#                 result = cur.fetchall()
#                 return result
#     except Exception as e:
#         print("❌ Error:", e)
#         return None

# if __name__ == "__main__":
#     print("✅ Connected successfully!\n")

#     queries = [
#         # Step 1: list all schemas
#         "SELECT schema_name FROM information_schema.schemata;",

#         # Step 2: list all tables across all schemas
#         "SELECT table_schema, table_name FROM information_schema.tables WHERE table_type='BASE TABLE';",

#         # Step 3: pick a table and preview some data
#         # We'll replace this after knowing which tables exist
#     ]

#     for q in queries:
#         print(f"\n🧠 Running query:\n{q}\n")
#         result = run_query(q)
#         if result:
#             for row in result[:10]:
#                 print(row)
#         else:
#             print("No result or query failed.")


# import psycopg2
# from psycopg2.extras import RealDictCursor

# # Connection details (public read-only DB)
# DB_CONFIG = {
#     "host": "hh-pgsql-public.ebi.ac.uk",
#     "port": 5432,
#     "database": "pfmegrnargs",
#     "user": "reader",
#     "password": "NWDMCE5xdipIjRrp"
# }

# def run_query(query):
#     """Run a SQL query and return the results as a list of dictionaries."""
#     try:
#         with psycopg2.connect(**DB_CONFIG) as conn:
#             with conn.cursor(cursor_factory=RealDictCursor) as cur:
#                 cur.execute(query)
#                 result = cur.fetchall()
#                 return result
#     except Exception as e:
#         print("❌ Error:", e)
#         return None

# if __name__ == "__main__":
#     print("✅ Connecting to RNAcentral public PostgreSQL database...")
    
#     # Example queries
#     test_queries = [
#         "SELECT table_name FROM information_schema.tables WHERE table_schema='public';",
#         "SELECT COUNT(*) FROM rnc_rna;",
#         "SELECT * FROM rnc_rna LIMIT 5;"
#     ]
    
#     for q in test_queries:
#         print(f"\n🧠 Running query:\n{q}")
#         result = run_query(q)
#         if result:
#             for row in result[:5]:  # show only first few rows
#                 print(row)
#         else:
#             print("No result or query failed.")
