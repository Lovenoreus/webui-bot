# Guide on how to deploy the MCPO integrated version of the omnigate Test Bot.
## WHAT IS LACKING.
- Praveen's RAG improvements.
- PGvector database.
- List all tool. (MCPO probably blocks this by default. Need to do more research on this.)

## WHAT CAN BE TESTED.
- SQL DATABASE (No need to say from the database all the time. Sample questions can be found at the bottom of this .md)

## WHAT CAN BE CONFIGURED.
- RAG (Big chunks - cosmic_documents_clean). All you need is the collection on your qdrant running.
  **Try with: use cosmic database tool to answer this. check results if cleaned and good.**
  **Then, try with: Normal utterance, no use cosmic database tool utterances. check results if cleaned and good.**

## Deployment steps.
- Git fetch.
- Git checkout to regional_main branch.
- docker compose down.
- Destroy all current docker images.
- navigate to webui root.
- create ENV: https://healthcaretechglobal.slack.com/files/U087JMT2XNJ/F09HZUB45T6/mcpo_.env
- create a directory called: local-binaries in the root directory
- copy the slack tgz file into it: https://healthcaretechglobal.slack.com/archives/C08CLG263D2/p1759219738726859
- Run command: docker compose build --no-cache
  **If above command gives issue, do: docker compose up**
- After build, run command: docker compose up
- After webui login, navigate to system prompt and inject this: 
  "Always call a tool first. Tools have enough information to answer any user question. If tool cannot answer before you do. If any tool response returns an sql_query or code, display it in a properly formatted manner."
- Configure model to: gpt-4o-mini
- Set Tool:
- Go ahead an test.

## SQL DATABASE SCHEMA (SQLITE)
- Note: Schema no longer deals with users. That is active directory stuff now.

Here are 10 questions with specific, verifiable answers based on the sample data in the code:

### POSSIBLE QUESTIONS FOR THE INVOICE SQL DATABASE

#### Simple Queries (Single Table)

1. **"How many invoices are in the database?"**
   - Keywords: `["invoices", "count"]`
   - Expected SQL: `SELECT COUNT(*) FROM Invoice`
   - Expected result: 50

2. **"Show me invoices from JA Hotel Karlskrona"**
   - Keywords: `["invoices", "JA Hotel Karlskrona"]`
   - Expected SQL: `SELECT INVOICE_ID, ISSUE_DATE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE SUPPLIER_PARTY_NAME = 'JA Hotel Karlskrona'`
   - Expected result: Multiple invoices from supplier ID 5592985237

3. **"What is the supplier company ID for Abbott Scandinavia?"**
   - Keywords: `["supplier", "company ID", "Abbott Scandinavia"]`
   - Expected SQL: `SELECT SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID FROM Invoice WHERE SUPPLIER_PARTY_NAME = 'Abbott Scandinavia' LIMIT 1`
   - Expected result: 5560466137

4. **"List all unique service names from invoice line items"**
   - Keywords: `["service", "names", "line items"]`
   - Expected SQL: `SELECT DISTINCT ITEM_NAME FROM Invoice_Line ORDER BY ITEM_NAME`
   - Expected result: 8 services (Consulting Fee, Hotel Accommodation, IT Consulting, Maintenance Service, Medical Supplies, Office Equipment, Software License, Training Services)

#### Medium Complexity (Joins)

5. **"Which suppliers are located in Stockholm?"**
   - Keywords: `["suppliers", "Stockholm"]`
   - Expected SQL: `SELECT DISTINCT SUPPLIER_PARTY_NAME, SUPPLIER_PARTY_STREET_NAME, SUPPLIER_PARTY_POSTAL_ZONE FROM Invoice WHERE SUPPLIER_PARTY_CITY = 'Stockholm'`
   - Expected result: 1 supplier (Nordic IT Solutions AB at Sveavägen 45, 111 34)

6. **"Show all line items for Hotel Accommodation services"**
   - Keywords: `["line items", "Hotel Accommodation"]`
   - Expected SQL: `SELECT il.INVOICE_ID, il.ITEM_NAME, il.INVOICED_QUANTITY, il.PRICE_AMOUNT, il.INVOICED_LINE_EXTENSION_AMOUNT FROM Invoice_Line il WHERE il.ITEM_NAME = 'Hotel Accommodation'`
   - Expected result: Multiple line items with ITEM_SELLERS_ITEM_ID = 'HOTEL', ITEM_TAXCAT_ID = 'S', ITEM_TAXCAT_PERCENT = 12.0

7. **"What is the contact email for Region Skåne?"**
   - Keywords: `["contact", "email", "Region Skåne"]`
   - Expected SQL: `SELECT DISTINCT CUSTOMER_PARTY_CONTACT_EMAIL FROM Invoice WHERE CUSTOMER_PARTY_NAME = 'Region Skåne'`
   - Expected result: inkop@skane.se

#### Complex Queries (Multiple Joins + Aggregation)

8. **"How many different customers have been invoiced?"**
   - Keywords: `["customers", "count", "invoiced"]`
   - Expected SQL: `SELECT COUNT(DISTINCT CUSTOMER_PARTY_NAME) as customer_count FROM Invoice`
   - Expected result: 5 (Region Västerbotten, Stockholms Stad, Region Skåne, Västra Götaland, Region Uppsala)

9. **"What is the total quantity of IT Consulting services invoiced?"**
   - Keywords: `["total", "quantity", "IT Consulting"]`
   - Expected SQL: `SELECT SUM(INVOICED_QUANTITY) as total_quantity FROM Invoice_Line WHERE ITEM_NAME = 'IT Consulting'`
   - Expected result: Varies based on random generation, but will be sum of all IT Consulting quantities

10. **"Which supplier has the phone number +46101992350?"**
    - Keywords: `["supplier", "phone", "+46101992350"]`
    - Expected SQL: `SELECT DISTINCT SUPPLIER_PARTY_NAME, SUPPLIER_PARTY_CITY FROM Invoice WHERE SUPPLIER_PARTY_CONTACT_PHONE = '+46101992350'`
    - Expected result: Visma Draftit AB, Malmo

# In-case of any deployment issues
- Contact: **The one who shall not be named**


# region_main: Regional Final (Nobody works on this branch)
# external_main: External Final (Nobody works on this branch)
# Main: ------ Latest Fork version of webui repo
# Everybody pulls from main. and updates their repos

# How can we get latest webui update? ---

# RAG (Qdrant - godlove rag/pg)
# RAG (PGVECTor - latest branch)
- vector database tool pg
- ...