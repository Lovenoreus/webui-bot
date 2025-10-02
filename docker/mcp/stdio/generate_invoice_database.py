# generate_invoice_database.py
"""
Generate SQLite invoice database with sample data
"""

import sqlite3
import random
from datetime import datetime, timedelta


def create_database(db_path="sqlite_invoices_full.db"):
    """Create and populate invoice database"""

    # Connect to database (creates file if doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"Creating database: {db_path}")

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys=ON")

    # Drop existing tables
    cursor.execute("DROP TABLE IF EXISTS Invoice_Line")
    cursor.execute("DROP TABLE IF EXISTS Invoice")

    # Create Invoice table
    cursor.execute("""
        CREATE TABLE Invoice (
            INVOICE_ID TEXT NOT NULL PRIMARY KEY,
            ISSUE_DATE TEXT NOT NULL,
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT NOT NULL,
            SUPPLIER_PARTY_NAME TEXT,
            SUPPLIER_PARTY_STREET_NAME TEXT,
            SUPPLIER_PARTY_ADDITIONAL_STREET_NAME TEXT,
            SUPPLIER_PARTY_POSTAL_ZONE TEXT,
            SUPPLIER_PARTY_CITY TEXT,
            SUPPLIER_PARTY_COUNTRY TEXT,
            SUPPLIER_PARTY_ADDRESS_LINE TEXT,
            SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME TEXT,
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM TEXT,
            SUPPLIER_PARTY_CONTACT_NAME TEXT,
            SUPPLIER_PARTY_CONTACT_EMAIL TEXT,
            SUPPLIER_PARTY_CONTACT_PHONE TEXT,
            SUPPLIER_PARTY_ENDPOINT_ID TEXT,
            CUSTOMER_PARTY_ID TEXT,
            CUSTOMER_PARTY_ID_SCHEME_ID TEXT,
            CUSTOMER_PARTY_ENDPOINT_ID TEXT,
            CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID TEXT,
            CUSTOMER_PARTY_NAME TEXT,
            CUSTOMER_PARTY_STREET_NAME TEXT,
            CUSTOMER_PARTY_POSTAL_ZONE TEXT,
            CUSTOMER_PARTY_COUNTRY TEXT,
            CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME TEXT,
            CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT,
            CUSTOMER_PARTY_CONTACT_NAME TEXT,
            CUSTOMER_PARTY_CONTACT_EMAIL TEXT,
            CUSTOMER_PARTY_CONTACT_PHONE TEXT,
            DUE_DATE TEXT,
            DOCUMENT_CURRENCY_CODE TEXT,
            DELIVERY_LOCATION_STREET_NAME TEXT,
            DELIVERY_LOCATION_ADDITIONAL_STREET_NAME TEXT,
            DELIVERY_LOCATION_CITY_NAME TEXT,
            DELIVERY_LOCATION_POSTAL_ZONE TEXT,
            DELIVERY_LOCATION_ADDRESS_LINE TEXT,
            DELIVERY_LOCATION_COUNTRY TEXT,
            DELIVERY_PARTY_NAME TEXT,
            ACTUAL_DELIVERY_DATE TEXT,
            TAX_AMOUNT_CURRENCY TEXT,
            TAX_AMOUNT REAL,
            PERIOD_START_DATE TEXT,
            PERIOD_END_DATE TEXT,
            LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT REAL,
            BUYER_REFERENCE TEXT,
            PROJECT_REFERENCE_ID TEXT,
            INVOICE_TYPE_CODE TEXT,
            NOTE TEXT,
            TAX_POINT_DATE TEXT,
            ACCOUNTING_COST TEXT,
            ORDER_REFERENCE_ID TEXT,
            ORDER_REFERENCE_SALES_ORDER_ID TEXT,
            PAYMENT_TERMS_NOTE TEXT,
            BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID TEXT,
            BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE TEXT,
            CONTRACT_DOCUMENT_REFERENCE_ID TEXT,
            DESPATCH_DOCUMENT_REFERENCE_ID TEXT,
            ETL_LOAD_TS TEXT
        )
    """)

    # Create Invoice_Line table
    cursor.execute("""
        CREATE TABLE Invoice_Line (
            INVOICE_ID TEXT NOT NULL,
            ISSUE_DATE TEXT NOT NULL,
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT NOT NULL,
            INVOICE_LINE_ID TEXT NOT NULL,
            ORDER_LINE_REFERENCE_LINE_ID TEXT,
            ACCOUNTING_COST TEXT,
            INVOICED_QUANTITY REAL,
            INVOICED_QUANTITY_UNIT_CODE TEXT,
            INVOICED_LINE_EXTENSION_AMOUNT REAL,
            INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID TEXT,
            INVOICE_PERIOD_START_DATE TEXT,
            INVOICE_PERIOD_END_DATE TEXT,
            INVOICE_LINE_DOCUMENT_REFERENCE_ID TEXT,
            INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE TEXT,
            INVOICE_LINE_NOTE TEXT,
            ITEM_DESCRIPTION TEXT,
            ITEM_NAME TEXT,
            ITEM_TAXCAT_ID TEXT,
            ITEM_TAXCAT_PERCENT REAL,
            ITEM_BUYERS_ID TEXT,
            ITEM_SELLERS_ITEM_ID TEXT,
            ITEM_STANDARD_ITEM_ID TEXT,
            ITEM_COMMODITYCLASS_CLASSIFICATION TEXT,
            ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID TEXT,
            PRICE_AMOUNT REAL,
            PRICE_AMOUNT_CURRENCY_ID TEXT,
            PRICE_BASE_QUANTITY REAL,
            PRICE_BASE_QUANTITY_UNIT_CODE TEXT,
            PRICE_ALLOWANCE_CHARGE_AMOUNT REAL,
            PRICE_ALLOWANCE_CHARGE_INDICATOR TEXT,
            ETL_LOAD_TS TEXT,
            PRIMARY KEY (INVOICE_ID, INVOICE_LINE_ID),
            FOREIGN KEY (INVOICE_ID) REFERENCES Invoice(INVOICE_ID)
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX idx_invoice_issue_date ON Invoice(ISSUE_DATE)")
    cursor.execute("CREATE INDEX idx_invoice_supplier ON Invoice(SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID)")
    cursor.execute("CREATE INDEX idx_invoice_customer ON Invoice(CUSTOMER_PARTY_ID)")
    cursor.execute("CREATE INDEX idx_invoice_line_invoice_id ON Invoice_Line(INVOICE_ID)")
    cursor.execute("CREATE INDEX idx_invoice_line_issue_date ON Invoice_Line(ISSUE_DATE)")
    cursor.execute("CREATE INDEX idx_invoice_line_supplier ON Invoice_Line(SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID)")

    print("Tables and indexes created successfully")

    # Insert sample data
    insert_sample_data(cursor)

    # Commit and close
    conn.commit()

    # Verify data
    cursor.execute("SELECT COUNT(*) FROM Invoice")
    invoice_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM Invoice_Line")
    line_count = cursor.fetchone()[0]

    print(f"\nDatabase created successfully!")
    print(f"Total invoices: {invoice_count}")
    print(f"Total invoice lines: {line_count}")

    conn.close()


def insert_sample_data(cursor):
    """Insert sample invoice data"""

    suppliers = [
        ('5592985237', 'JA Hotel Karlskrona', 'Borgmästaregatan 13', '37115', 'Karlskrona', 'info@jahotel.se',
         '045555560'),
        ('5565783957', 'Visma Draftit AB', 'Styrmansgatan 2', '21118', 'Malmo', 'support@vismadraftit.se',
         '+46101992350'),
        ('5560466137', 'Abbott Scandinavia', 'Hemvärnsgatan 9', '171 29', 'Solna', 'contact@abbott.se', '+46850123400'),
        ('5591234567', 'Nordic IT Solutions AB', 'Sveavägen 45', '111 34', 'Stockholm', 'info@nordicit.se',
         '+46812345678'),
        ('5598765432', 'Malmö Tech Services', 'Drottninggatan 22', '211 15', 'Malmö', 'support@malmotech.se',
         '+46401234567'),
        ('5587654321', 'Göteborg Consulting Group', 'Avenyn 12', '411 36', 'Göteborg', 'hello@gbgconsult.se',
         '+46312345678'),
        ('5576543210', 'Uppsala Innovation Labs', 'Kungsgatan 8', '753 21', 'Uppsala', 'lab@uppsalainnovation.se',
         '+46181234567'),
        ('5565432109', 'Linköping Software House', 'Storgatan 33', '582 23', 'Linköping', 'dev@linkopingsw.se',
         '+46131234567')
    ]

    customers = [
        ('7362321000224', 'Region Västerbotten', '2321000222', 'DEJA01', 'daniel.stromberg@regionvasterbotten.se'),
        ('7365432100001', 'Stockholms Stad', '2123456789', 'STOCK01', 'procurement@stockholm.se'),
        ('7367654321002', 'Region Skåne', '2987654321', 'SKANE01', 'inkop@skane.se'),
        ('7361234567003', 'Västra Götaland', '2456789012', 'VGR01', 'inköp@vgregion.se'),
        ('7368765432004', 'Region Uppsala', '2345678901', 'UPP01', 'ekonomi@regionuppsala.se')
    ]

    services = [
        ('IT Consulting', 'ITCONS', 'HR', 25.0),
        ('Software License', 'SWLIC', 'S', 25.0),
        ('Hotel Accommodation', 'HOTEL', 'S', 12.0),
        ('Training Services', 'TRAIN', 'S', 25.0),
        ('Medical Supplies', 'MEDSUP', 'S', 25.0),
        ('Office Equipment', 'OFFICE', 'S', 25.0),
        ('Maintenance Service', 'MAINT', 'S', 25.0),
        ('Consulting Fee', 'CONFEE', 'S', 25.0)
    ]

    current_date = datetime.now()
    invoice_data = []
    invoice_line_data = []

    print("Generating invoice data...")

    for i in range(50):
        invoice_id = f"INV{str(i + 1).zfill(6)}"
        issue_date = (current_date - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d')
        due_date = (datetime.strptime(issue_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')

        supplier = random.choice(suppliers)
        customer = random.choice(customers)

        tax_amount = round(random.uniform(100, 5000), 2)
        line_ext_amount = round(random.uniform(1000, 25000), 2)
        tax_incl_amount = line_ext_amount + tax_amount

        etl_timestamp = (current_date + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d %H:%M:%S.000')

        # All 73 columns for Invoice table
        invoice_data.append((
            invoice_id,  # INVOICE_ID
            issue_date,  # ISSUE_DATE
            supplier[0],  # SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID
            supplier[1],  # SUPPLIER_PARTY_NAME
            supplier[2],  # SUPPLIER_PARTY_STREET_NAME
            None,  # SUPPLIER_PARTY_ADDITIONAL_STREET_NAME
            supplier[3],  # SUPPLIER_PARTY_POSTAL_ZONE
            supplier[4],  # SUPPLIER_PARTY_CITY
            'SE',  # SUPPLIER_PARTY_COUNTRY
            None,  # SUPPLIER_PARTY_ADDRESS_LINE
            supplier[1],  # SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME
            'Aktiebolag',  # SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM
            None,  # SUPPLIER_PARTY_CONTACT_NAME
            supplier[5],  # SUPPLIER_PARTY_CONTACT_EMAIL
            supplier[6],  # SUPPLIER_PARTY_CONTACT_PHONE
            supplier[0],  # SUPPLIER_PARTY_ENDPOINT_ID
            customer[0],  # CUSTOMER_PARTY_ID
            '0088',  # CUSTOMER_PARTY_ID_SCHEME_ID
            customer[0],  # CUSTOMER_PARTY_ENDPOINT_ID
            '0088',  # CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID
            customer[1],  # CUSTOMER_PARTY_NAME
            None,  # CUSTOMER_PARTY_STREET_NAME
            None,  # CUSTOMER_PARTY_POSTAL_ZONE
            'SE',  # CUSTOMER_PARTY_COUNTRY
            customer[1],  # CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME
            customer[2],  # CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID
            customer[3],  # CUSTOMER_PARTY_CONTACT_NAME
            customer[4],  # CUSTOMER_PARTY_CONTACT_EMAIL
            None,  # CUSTOMER_PARTY_CONTACT_PHONE
            due_date,  # DUE_DATE
            'SEK',  # DOCUMENT_CURRENCY_CODE
            None,  # DELIVERY_LOCATION_STREET_NAME
            None,  # DELIVERY_LOCATION_ADDITIONAL_STREET_NAME
            None,  # DELIVERY_LOCATION_CITY_NAME
            None,  # DELIVERY_LOCATION_POSTAL_ZONE
            None,  # DELIVERY_LOCATION_ADDRESS_LINE
            None,  # DELIVERY_LOCATION_COUNTRY
            None,  # DELIVERY_PARTY_NAME
            None,  # ACTUAL_DELIVERY_DATE
            'SEK',  # TAX_AMOUNT_CURRENCY
            tax_amount,  # TAX_AMOUNT
            None,  # PERIOD_START_DATE
            None,  # PERIOD_END_DATE
            'SEK',  # LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY
            line_ext_amount,  # LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT
            'SEK',  # LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY
            line_ext_amount,  # LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT
            'SEK',  # LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY
            tax_incl_amount,  # LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT
            'SEK',  # LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY
            tax_incl_amount,  # LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
            None,  # LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY
            None,  # LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT
            None,  # LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY
            None,  # LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT
            'SEK',  # LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY
            0.0,  # LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT
            None,  # LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY
            None,  # LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT
            customer[3],  # BUYER_REFERENCE
            None,  # PROJECT_REFERENCE_ID
            '380',  # INVOICE_TYPE_CODE
            f'Invoice for {supplier[1]} services',  # NOTE
            None,  # TAX_POINT_DATE
            None,  # ACCOUNTING_COST
            f'PO-{str(i + 1).zfill(4)}',  # ORDER_REFERENCE_ID
            None,  # ORDER_REFERENCE_SALES_ORDER_ID
            'Net 30',  # PAYMENT_TERMS_NOTE
            None,  # BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID
            None,  # BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE
            None,  # CONTRACT_DOCUMENT_REFERENCE_ID
            None,  # DESPATCH_DOCUMENT_REFERENCE_ID
            etl_timestamp  # ETL_LOAD_TS
        ))

        # Generate 1-3 lines per invoice
        num_lines = random.randint(1, 3)
        for line_num in range(1, num_lines + 1):
            service = random.choice(services)
            quantity = round(random.uniform(1, 10), 3)
            unit_price = round(random.uniform(500, 3000), 2)
            line_amount = round(quantity * unit_price, 2)

            invoice_line_data.append((
                invoice_id,  # INVOICE_ID
                issue_date,  # ISSUE_DATE
                supplier[0],  # SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID
                str(line_num),  # INVOICE_LINE_ID
                None,  # ORDER_LINE_REFERENCE_LINE_ID
                None,  # ACCOUNTING_COST
                quantity,  # INVOICED_QUANTITY
                'EA',  # INVOICED_QUANTITY_UNIT_CODE
                line_amount,  # INVOICED_LINE_EXTENSION_AMOUNT
                'SEK',  # INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID
                None,  # INVOICE_PERIOD_START_DATE
                None,  # INVOICE_PERIOD_END_DATE
                None,  # INVOICE_LINE_DOCUMENT_REFERENCE_ID
                None,  # INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE
                None,  # INVOICE_LINE_NOTE
                None,  # ITEM_DESCRIPTION
                service[0],  # ITEM_NAME
                service[2],  # ITEM_TAXCAT_ID
                service[3],  # ITEM_TAXCAT_PERCENT
                None,  # ITEM_BUYERS_ID
                service[1],  # ITEM_SELLERS_ITEM_ID
                None,  # ITEM_STANDARD_ITEM_ID
                None,  # ITEM_COMMODITYCLASS_CLASSIFICATION
                None,  # ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID
                unit_price,  # PRICE_AMOUNT
                'SEK',  # PRICE_AMOUNT_CURRENCY_ID
                None,  # PRICE_BASE_QUANTITY
                None,  # PRICE_BASE_QUANTITY_UNIT_CODE
                None,  # PRICE_ALLOWANCE_CHARGE_AMOUNT
                None,  # PRICE_ALLOWANCE_CHARGE_INDICATOR
                etl_timestamp  # ETL_LOAD_TS
            ))

    print(f"Inserting {len(invoice_data)} invoices...")
    cursor.executemany("""
        INSERT INTO Invoice VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, invoice_data)

    print(f"Inserting {len(invoice_line_data)} invoice lines...")
    cursor.executemany("""
        INSERT INTO Invoice_Line VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, invoice_line_data)

    print("Sample data inserted successfully")


if __name__ == "__main__":
    create_database("sqlite_invoices_full.db")
    print("\n✓ Database 'sqlite_invoices_full.db' created successfully!")