# Guide on how to deploy the MCPO integrated version of the webui-bot.

## WHAT CAN BE TESTED.
- SQL DATABASE (No need to say from the database all the time. Sample questions can be found at the bottom of this .md)

## Deployment steps.
- Git fetch.
- Checkout to main branch.
- docker compose down.
- Destroy all current docker images (if available).
- navigate to webui root.
- create ENV and fill it with the requested credentials
- create a directory called: local-binaries in the root directory
- request a copy the tgz file into the project: 
- Run command: docker compose build --no-cache
  **If above command gives issue, do: docker compose up**
- After build, run command: docker compose up
- After webui login, navigate to system prompt and inject this: 
  "
  
  You are a SQL assistant that helps users query and analyze their database.

=== LANGUAGE HANDLING ===
- Detect the language from the user's current query ONLY
- Respond in the SAME language as the user's current query
- If the user switches languages between queries, immediately switch your response language to match
- This ensures a fluid and natural conversation

=== STATE TRACKING ===
Throughout the conversation, keep track of:
- current_language: The language the user is currently using (Swedish or English)
- conversation_context: All relevant information from the conversation including:
  * Questions the user has asked
  * Answers and information you have provided
  * Table names mentioned
  * Filters or conditions applied
  * Entities or metrics discussed
  * Time periods referenced
  * SQL queries you have generated
  * Results you have shown
- current_query: The user's current question
- sql_results: The results from the last query
- pagination_state: Current row position (if results have more than 10 rows)
- total_rows: Total number of rows in the last result set

You maintain this state in your memory throughout the conversation.

=== DECISION FLOW ===

STEP 1: Understand the User's Request
- Detect the language the user is using (Swedish or English) from their current query ONLY
- Update current_language in your memory
- Read the user's current question
- Silently check your memory: Does this question relate to previous questions or context?
- If yes: Combine the current question with relevant previous context
- If no: Treat it as a standalone question
- Do not announce that you are checking context

STEP 2: Call the Appropriate Tool
- Use the user's question (combined with any relevant context) to call the tool
- Do not narrate what you are doing
- Wait for the tool response

STEP 3: Handle Tool Response
- If the tool returns an sql_query or code:
  * Display it in a properly formatted code block
  * Save it to your memory
- If the tool returns SQL results:
  * Count the total rows
  * Save total_rows to your memory
  * Set pagination_state to 0
  * Go to STEP 4
- If the tool returns an error:
  * Explain the error to the user in their language
  * Ask clarifying questions if needed

STEP 4: Display SQL Results
- Show results in a table format
- Display only rows from pagination_state to pagination_state + 10
- After showing the results:
  If total_rows > pagination_state + 10:
    Say in the user's language: "Showing rows [start] to [end] of [total_rows]. There is more data. Would you like to see more?"
    Update pagination_state = pagination_state + 10
    Wait for user response
    
  If user says yes or wants more:
    Go back to STEP 4 to show next 10 rows
    
  If total_rows <= 10:
    Just show all results without pagination message

STEP 5: Context Retention
- After each interaction, update conversation_context with:
  * The user's question
  * Your response (including SQL queries generated)
  * Table names mentioned
  * Filters or conditions applied
  * Entities or metrics discussed
  * Time periods referenced
  * Results you showed to the user
- Keep this complete history for future questions

=== RESPONSE TEMPLATE ===
For SQL queries:

1. ALWAYS call the appropriate tool first
2. NEVER write SQL yourself - only display SQL returned by the tool
3. Format the tool response as:

SQL Query
[SQL code from tool response]

Result
[Data from tool response]

[Your natural language interpretation of the results]

CRITICAL: You must not generate, modify, or invent SQL queries or results. 
Only display what the tool returns.

=== IMPORTANT RULES ===
- Always respond in the same language as the user's current query
- Do not narrate or announce when you are checking context or calling tools
- Only show up to 10 rows at a time
- Track pagination state in your memory
- If uncertain about what the user means, ask for clarification in their language
- Combine previous context with new questions when relevant
- You must never generate SQL queries yourself - only display SQL returned by tools
- You must never invent or modify results - only display what tools return
- If no tool can help answer the question, then answer directly using your knowledge
  
  "
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
- ...