#!/bin/bash

# Complete MSSQL Docker Setup and Data Migration Script
# This script handles MSSQL Server setup, data migration from compacted.db, and management

set -e

# Configuration
CONTAINER_NAME="mssql-server-compacted"
DB_HOST="localhost"
DB_PORT="1433"
DB_DATABASE="Compacted"
DB_USERNAME="sa"
DB_PASSWORD="YourPassword123!"
MSSQL_IMAGE="mcr.microsoft.com/mssql/server:2022-latest"
SQLITE_DB="/home/hammad/Paul_Healtcare/Git_working_regional_sql/webui-bot/docker/mcp/stdio/compacted.db"
TEMP_DIR="/tmp/mssql_migration"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    # First check if docker command exists
    docker_path=$(which docker 2>/dev/null)
    if [ -z "$docker_path" ]; then
        print_error "Docker command not found. Please install Docker first."
        exit 1
    fi
    
    # Then check if Docker daemon is running
    if ! docker ps >/dev/null 2>&1; then
        print_error "Docker daemon is not running or you don't have permission to use Docker."
        print_error "Try running: sudo systemctl start docker"
        print_error "Or add your user to the docker group with: sudo usermod -aG docker $USER"
        exit 1
    fi
    
    print_status "Docker is available and running at $docker_path"
}

# Check if container already exists
check_existing_container() {
    if docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        return 0
    else
        return 1
    fi
}

# Start the MSSQL container
start_container() {
    print_status "Starting MSSQL Server container..."
    
    docker run -d \
        --name "${CONTAINER_NAME}" \
        -e "ACCEPT_EULA=Y" \
        -e "MSSQL_SA_PASSWORD=${DB_PASSWORD}" \
        -e "MSSQL_PID=Express" \
        -p "${DB_PORT}:1433" \
        --restart unless-stopped \
        "${MSSQL_IMAGE}"
    
    print_success "Container '${CONTAINER_NAME}' started successfully"
}

# Wait for MSSQL to be ready
wait_for_mssql() {
    print_status "Waiting for MSSQL Server to be ready..."
    print_status "This may take a few minutes on first startup due to database initialization..."
    
    local max_attempts=60  # Increased timeout to 5 minutes
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        # Check if container is still running first
        if ! docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            print_error "Container stopped unexpectedly. Check logs with: $0 logs"
            return 1
        fi
        
        # Try to connect to MSSQL
        if docker exec "${CONTAINER_NAME}" /opt/mssql-tools18/bin/sqlcmd \
            -S localhost -U "${DB_USERNAME}" -P "${DB_PASSWORD}" \
            -C -Q "SELECT 1" -t 1 -r 0 &> /dev/null; then
            print_success "MSSQL Server is ready!"
            return 0
        fi
        
        # Show progress every 5 attempts
        if [ $((attempt % 5)) -eq 0 ]; then
            print_status "Attempt $attempt/$max_attempts - still waiting for MSSQL to start..."
        fi
        
        sleep 5
        ((attempt++))
    done
    
    print_error "MSSQL Server failed to start within expected time (5 minutes)"
    print_status "Check container logs with: $0 logs"
    return 1
}

# Create the database
create_database() {
    print_status "Creating database '${DB_DATABASE}'..."
    
    docker exec "${CONTAINER_NAME}" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U "${DB_USERNAME}" -P "${DB_PASSWORD}" \
        -C -Q "CREATE DATABASE [${DB_DATABASE}];" -t 30
    
    if [ $? -eq 0 ]; then
        print_success "Database '${DB_DATABASE}' created successfully"
    else
        print_warning "Database creation failed or database already exists"
    fi
}

# Stop and remove the container
stop_container() {
    print_status "Stopping MSSQL Server container..."
    
    if docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        docker stop "${CONTAINER_NAME}"
        print_success "Container stopped"
    else
        print_warning "Container is not running"
    fi
    
    if check_existing_container; then
        docker rm "${CONTAINER_NAME}"
        print_success "Container removed"
    fi
}

# Show connection information
show_connection_info() {
    echo ""
    echo "======================================================"
    echo "MSSQL Server Connection Information:"
    echo "======================================================"
    echo "Host: ${DB_HOST}"
    echo "Port: ${DB_PORT}"
    echo "Database: ${DB_DATABASE}"
    echo "Username: ${DB_USERNAME}"
    echo "Password: ${DB_PASSWORD}"
    echo ""
    echo "Connection String:"
    echo "Server=${DB_HOST},${DB_PORT};Database=${DB_DATABASE};User Id=${DB_USERNAME};Password=${DB_PASSWORD};TrustServerCertificate=true;"
    echo ""
    echo "Docker Container: ${CONTAINER_NAME}"
    echo "======================================================"
}

# Show container status
show_status() {
    if check_existing_container; then
        echo "Container Status:"
        docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        
        if docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            echo ""
            show_connection_info
        fi
    else
        print_warning "Container '${CONTAINER_NAME}' does not exist"
    fi
}

# Show container logs
show_logs() {
    if check_existing_container; then
        docker logs "${CONTAINER_NAME}"
    else
        print_error "Container '${CONTAINER_NAME}' does not exist"
    fi
}

# Check prerequisites for migration
check_migration_prerequisites() {
    print_status "Checking migration prerequisites..."
    
    # Check if SQLite file exists
    if [ ! -f "$SQLITE_DB" ]; then
        print_error "SQLite database file not found: $SQLITE_DB"
        exit 1
    fi
    
    # Check if MSSQL container is running
    if ! docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        print_error "MSSQL container is not running. Start it with: $0 start"
        exit 1
    fi
    
    # Check if sqlite3 is available
    if command -v sqlite3 &> /dev/null; then
        SQLITE3_CMD=$(command -v sqlite3)
        print_status "Found SQLite3 at $SQLITE3_CMD"
    elif [ -f "/home/hammad/anaconda3/bin/sqlite3" ]; then
        SQLITE3_CMD="/home/hammad/anaconda3/bin/sqlite3"
        print_status "Using SQLite3 from Anaconda at $SQLITE3_CMD"
    else
        print_error "sqlite3 command not found. Please install sqlite3."
        exit 1
    fi
    
    print_success "All migration prerequisites met"
}

# Create temporary directory
setup_temp_dir() {
    print_status "Setting up temporary directory..."
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"
    print_success "Temporary directory created: $TEMP_DIR"
}

# Create MSSQL tables
create_mssql_tables() {
    print_status "Creating MSSQL tables..."
    
    # Drop existing tables if they exist (except test_table)
    docker exec "$CONTAINER_NAME" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U "$DB_USERNAME" -P "$DB_PASSWORD" \
        -C -d "$DB_DATABASE" -Q "
        IF EXISTS (SELECT * FROM sys.tables WHERE name = 'Invoice_Line')
            DROP TABLE Invoice_Line;
        IF EXISTS (SELECT * FROM sys.tables WHERE name = 'Invoice')
            DROP TABLE Invoice;
        " -t 30
    
    # Create Invoice table
    docker exec "$CONTAINER_NAME" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U "$DB_USERNAME" -P "$DB_PASSWORD" \
        -C -d "$DB_DATABASE" -Q "
        CREATE TABLE Invoice (
            INVOICE_ID NVARCHAR(255),
            ISSUE_DATE NVARCHAR(50),
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(255),
            SUPPLIER_PARTY_NAME NVARCHAR(500),
            SUPPLIER_PARTY_STREET_NAME NVARCHAR(500),
            SUPPLIER_PARTY_ADDITIONAL_STREET_NAME NVARCHAR(500),
            SUPPLIER_PARTY_POSTAL_ZONE NVARCHAR(50),
            SUPPLIER_PARTY_CITY NVARCHAR(255),
            SUPPLIER_PARTY_COUNTRY NVARCHAR(100),
            SUPPLIER_PARTY_ADDRESS_LINE NVARCHAR(1000),
            SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(500),
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM NVARCHAR(255),
            SUPPLIER_PARTY_CONTACT_NAME NVARCHAR(255),
            SUPPLIER_PARTY_CONTACT_EMAIL NVARCHAR(255),
            SUPPLIER_PARTY_CONTACT_PHONE NVARCHAR(50),
            SUPPLIER_PARTY_ENDPOINT_ID NVARCHAR(255),
            CUSTOMER_PARTY_ID NVARCHAR(255),
            CUSTOMER_PARTY_ID_SCHEME_ID NVARCHAR(255),
            CUSTOMER_PARTY_ENDPOINT_ID NVARCHAR(255),
            CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID NVARCHAR(255),
            CUSTOMER_PARTY_NAME NVARCHAR(500),
            CUSTOMER_PARTY_STREET_NAME NVARCHAR(500),
            CUSTOMER_PARTY_POSTAL_ZONE NVARCHAR(50),
            CUSTOMER_PARTY_COUNTRY NVARCHAR(100),
            CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(500),
            CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(255),
            CUSTOMER_PARTY_CONTACT_NAME NVARCHAR(255),
            CUSTOMER_PARTY_CONTACT_EMAIL NVARCHAR(255),
            CUSTOMER_PARTY_CONTACT_PHONE NVARCHAR(50),
            DUE_DATE NVARCHAR(50),
            DOCUMENT_CURRENCY_CODE NVARCHAR(10),
            DELIVERY_LOCATION_STREET_NAME NVARCHAR(500),
            DELIVERY_LOCATION_ADDITIONAL_STREET_NAME NVARCHAR(500),
            DELIVERY_LOCATION_CITY_NAME NVARCHAR(255),
            DELIVERY_LOCATION_POSTAL_ZONE NVARCHAR(50),
            DELIVERY_LOCATION_ADDRESS_LINE NVARCHAR(1000),
            DELIVERY_LOCATION_COUNTRY NVARCHAR(100),
            DELIVERY_PARTY_NAME NVARCHAR(500),
            ACTUAL_DELIVERY_DATE NVARCHAR(50),
            TAX_AMOUNT_CURRENCY NVARCHAR(10),
            TAX_AMOUNT NVARCHAR(50),
            PERIOD_START_DATE NVARCHAR(50),
            PERIOD_END_DATE NVARCHAR(50),
            LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY NVARCHAR(10),
            LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT NVARCHAR(50),
            LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY NVARCHAR(10),
            LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT NVARCHAR(50),
            LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY NVARCHAR(10),
            LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT NVARCHAR(50),
            LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY NVARCHAR(10),
            LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT NVARCHAR(50),
            LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY NVARCHAR(10),
            LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT NVARCHAR(50),
            LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY NVARCHAR(10),
            LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT NVARCHAR(50),
            LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY NVARCHAR(10),
            LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT NVARCHAR(50),
            LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY NVARCHAR(10),
            LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT NVARCHAR(50),
            BUYER_REFERENCE NVARCHAR(255),
            PROJECT_REFERENCE_ID NVARCHAR(255),
            INVOICE_TYPE_CODE NVARCHAR(50),
            NOTE NVARCHAR(MAX),
            TAX_POINT_DATE NVARCHAR(50),
            ACCOUNTING_COST NVARCHAR(255),
            ORDER_REFERENCE_ID NVARCHAR(255),
            ORDER_REFERENCE_SALES_ORDER_ID NVARCHAR(255),
            PAYMENT_TERMS_NOTE NVARCHAR(MAX),
            BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID NVARCHAR(255),
            BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE NVARCHAR(50),
            CONTRACT_DOCUMENT_REFERENCE_ID NVARCHAR(255),
            DESPATCH_DOCUMENT_REFERENCE_ID NVARCHAR(255),
            ETL_LOAD_TS NVARCHAR(50)
        );
        " -t 60
    
    # Create Invoice_Line table
    docker exec "$CONTAINER_NAME" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U "$DB_USERNAME" -P "$DB_PASSWORD" \
        -C -d "$DB_DATABASE" -Q "
        CREATE TABLE Invoice_Line (
            INVOICE_ID NVARCHAR(255),
            ISSUE_DATE NVARCHAR(50),
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(255),
            INVOICE_LINE_ID NVARCHAR(255),
            ORDER_LINE_REFERENCE_LINE_ID NVARCHAR(255),
            ACCOUNTING_COST NVARCHAR(255),
            INVOICED_QUANTITY NVARCHAR(50),
            INVOICED_QUANTITY_UNIT_CODE NVARCHAR(50),
            INVOICED_LINE_EXTENSION_AMOUNT NVARCHAR(50),
            INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID NVARCHAR(10),
            INVOICE_PERIOD_START_DATE NVARCHAR(50),
            INVOICE_PERIOD_END_DATE NVARCHAR(50),
            INVOICE_LINE_DOCUMENT_REFERENCE_ID NVARCHAR(255),
            INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE NVARCHAR(50),
            INVOICE_LINE_NOTE NVARCHAR(MAX),
            ITEM_DESCRIPTION NVARCHAR(MAX),
            ITEM_NAME NVARCHAR(500),
            ITEM_TAXCAT_ID NVARCHAR(50),
            ITEM_TAXCAT_PERCENT NVARCHAR(50),
            ITEM_BUYERS_ID NVARCHAR(255),
            ITEM_SELLERS_ITEM_ID NVARCHAR(255),
            ITEM_STANDARD_ITEM_ID NVARCHAR(255),
            ITEM_COMMODITYCLASS_CLASSIFICATION NVARCHAR(255),
            ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID NVARCHAR(255),
            PRICE_AMOUNT NVARCHAR(50),
            PRICE_AMOUNT_CURRENCY_ID NVARCHAR(10),
            PRICE_BASE_QUANTITY NVARCHAR(50),
            PRICE_BASE_QUANTITY_UNIT_CODE NVARCHAR(50),
            PRICE_ALLOWANCE_CHARGE_AMOUNT NVARCHAR(50),
            PRICE_ALLOWANCE_CHARGE_INDICATOR NVARCHAR(10),
            ETL_LOAD_TS NVARCHAR(50)
        );
        " -t 60
    
    print_success "MSSQL tables created successfully"
}

# Export data from SQLite to CSV
export_sqlite_data() {
    print_status "Exporting data from SQLite..."
    
    # Export Invoice table
    sqlite3 "$SQLITE_DB" <<EOF
.headers on
.mode csv
.output $TEMP_DIR/Invoice.csv
SELECT * FROM Invoice;
.quit
EOF
    
    # Export Invoice_Line table
    sqlite3 "$SQLITE_DB" <<EOF
.headers on
.mode csv
.output $TEMP_DIR/Invoice_Line.csv
SELECT * FROM Invoice_Line;
.quit
EOF
    
    print_success "Data exported to CSV files"
}

# Import data to MSSQL
import_mssql_data() {
    print_status "Importing data to MSSQL..."
    
    # Copy CSV files to container
    docker cp "$TEMP_DIR/Invoice.csv" "$CONTAINER_NAME:/tmp/Invoice.csv"
    docker cp "$TEMP_DIR/Invoice_Line.csv" "$CONTAINER_NAME:/tmp/Invoice_Line.csv"
    
    # Import Invoice data
    print_status "Importing Invoice data..."
    docker exec "$CONTAINER_NAME" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U "$DB_USERNAME" -P "$DB_PASSWORD" \
        -C -d "$DB_DATABASE" -Q "
        BULK INSERT Invoice
        FROM '/tmp/Invoice.csv'
        WITH (
            FIRSTROW = 2,
            FIELDTERMINATOR = ',',
            ROWTERMINATOR = '\n',
            TABLOCK
        );
        " -t 120
    
    # Import Invoice_Line data
    print_status "Importing Invoice_Line data..."
    docker exec "$CONTAINER_NAME" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U "$DB_USERNAME" -P "$DB_PASSWORD" \
        -C -d "$DB_DATABASE" -Q "
        BULK INSERT Invoice_Line
        FROM '/tmp/Invoice_Line.csv'
        WITH (
            FIRSTROW = 2,
            FIELDTERMINATOR = ',',
            ROWTERMINATOR = '\n',
            TABLOCK
        );
        " -t 120
    
    print_success "Data imported successfully"
}

# Verify data import
verify_import() {
    print_status "Verifying data import..."
    
    # Get record counts
    local counts
    counts=$(docker exec "$CONTAINER_NAME" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U "$DB_USERNAME" -P "$DB_PASSWORD" \
        -C -d "$DB_DATABASE" -Q "
        SELECT 'Invoice' as TableName, COUNT(*) as Records FROM Invoice
        UNION ALL
        SELECT 'Invoice_Line' as TableName, COUNT(*) as Records FROM Invoice_Line;
        " -t 30 -h -1)
    
    echo "Data verification:"
    echo "$counts"
    
    if echo "$counts" | grep -q "Invoice.*50" && echo "$counts" | grep -q "Invoice_Line.*50"; then
        print_success "Data import verified successfully!"
    else
        print_warning "Please check data counts manually"
    fi
}

# Cleanup temporary files
cleanup_migration() {
    print_status "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    docker exec "$CONTAINER_NAME" rm -f /tmp/Invoice.csv /tmp/Invoice_Line.csv 2>/dev/null || true
    print_success "Cleanup completed"
}

# Perform full migration
migrate_data() {
    echo "========================================================"
    echo "SQLite to MSSQL Migration"
    echo "========================================================"
    
    check_migration_prerequisites
    setup_temp_dir
    create_mssql_tables
    export_sqlite_data
    import_mssql_data
    verify_import
    cleanup_migration
    
    echo ""
    print_success "Migration completed successfully!"
    show_connection_info
}

# Verify migrated data
verify_data() {
    if ! docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        print_error "MSSQL container is not running. Start it with: $0 start"
        exit 1
    fi
    
    echo -e "${BLUE}Compacted Database Verification${NC}"
    echo "=================================="
    
    echo "Tables and record counts:"
    docker exec "$CONTAINER_NAME" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U "$DB_USERNAME" -P "$DB_PASSWORD" \
        -C -d "$DB_DATABASE" -Q "
        SELECT 'Invoice' as TableName, COUNT(*) as Records FROM Invoice
        UNION ALL
        SELECT 'Invoice_Line' as TableName, COUNT(*) as Records FROM Invoice_Line;
        " -t 30
    
    echo ""
    echo "Sample Invoice data:"
    docker exec "$CONTAINER_NAME" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U "$DB_USERNAME" -P "$DB_PASSWORD" \
        -C -d "$DB_DATABASE" -Q "
        SELECT TOP 2 
            INVOICE_ID,
            ISSUE_DATE,
            LEFT(SUPPLIER_PARTY_NAME, 30) as Supplier,
            LEFT(CUSTOMER_PARTY_NAME, 30) as Customer
        FROM Invoice;
        " -t 30
    
    echo ""
    echo "Sample Invoice_Line data:"
    docker exec "$CONTAINER_NAME" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U "$DB_USERNAME" -P "$DB_PASSWORD" \
        -C -d "$DB_DATABASE" -Q "
        SELECT TOP 2 
            INVOICE_ID,
            INVOICE_LINE_ID,
            LEFT(ITEM_NAME, 25) as Item,
            INVOICED_QUANTITY,
            PRICE_AMOUNT
        FROM Invoice_Line;
        " -t 30
    
    echo ""
    print_success "Your compacted.db data is available in MSSQL Server!"
    show_connection_info
}

# Connection test
test_connection() {
    if ! docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        print_error "MSSQL container is not running. Start it with: $0 start"
        exit 1
    fi
    
    echo -e "${BLUE}Testing MSSQL Server Connection...${NC}"
    echo "=================================="
    
    # Test basic connection
    echo -n "Testing server connection: "
    if docker exec "${CONTAINER_NAME}" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U "${DB_USERNAME}" -P "${DB_PASSWORD}" \
        -C -Q "SELECT 1" -t 10 -r 0 &> /dev/null; then
        echo -e "${GREEN}✓ SUCCESS${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
        exit 1
    fi
    
    # Test database access
    echo -n "Testing database access: "
    if docker exec "${CONTAINER_NAME}" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U "${DB_USERNAME}" -P "${DB_PASSWORD}" \
        -C -d "${DB_DATABASE}" -Q "SELECT DB_NAME() AS CurrentDatabase" -t 10 -r 0 | grep -q "${DB_DATABASE}"; then
        echo -e "${GREEN}✓ SUCCESS${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
        exit 1
    fi
    
    echo ""
    print_success "All connection tests passed!"
    show_connection_info
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "COMMANDS:"
    echo "  start       - Start the MSSQL Server container and create database"
    echo "  stop        - Stop and remove the MSSQL Server container"
    echo "  restart     - Stop and start the container"
    echo "  status      - Show container status and connection info"
    echo "  logs        - Show container logs"
    echo "  migrate     - Migrate data from compacted.db to MSSQL"
    echo "  verify      - Verify migrated data in MSSQL"
    echo "  test        - Test connection to MSSQL Server"
    echo "  help        - Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 start           # Start MSSQL Server"
    echo "  $0 migrate         # Import compacted.db data"
    echo "  $0 verify          # Check imported data"
    echo "  $0 stop            # Stop and cleanup"
    echo ""
}

# Main script logic
main() {
    case "${1:-help}" in
        "start")
            check_docker
            
            if check_existing_container; then
                if docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
                    print_warning "Container '${CONTAINER_NAME}' is already running"
                    show_connection_info
                    exit 0
                else
                    print_status "Starting existing container..."
                    docker start "${CONTAINER_NAME}"
                    wait_for_mssql
                    show_connection_info
                    exit 0
                fi
            fi
            
            start_container
            wait_for_mssql
            create_database
            show_connection_info
            ;;
            
        "stop")
            stop_container
            ;;
            
        "restart")
            stop_container
            sleep 2
            main start
            ;;
            
        "status")
            show_status
            ;;
            
        "logs")
            show_logs
            ;;
            
        "migrate")
            migrate_data
            ;;
            
        "verify")
            verify_data
            ;;
            
        "test")
            test_connection
            ;;
            
        "help"|"-h"|"--help")
            show_usage
            ;;
            
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Trap to handle script interruption
trap 'print_warning "Script interrupted"; exit 130' INT TERM

# Run main function
main "$@"