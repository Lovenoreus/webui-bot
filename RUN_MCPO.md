# Guide on how to deploy the MCPO integrated version of the omnigate Test Bot.
## WHAT IS LACKING.
- Praveen's RAG improvements.
- PGvector database.
- List all tool. (MCPO probably blocks this by default. Need to do more research on this.)
- Support Ticket Creation

## WHAT CAN BE TESTED.
- ACTIVE DIRECTORY (No need to say from active directory. List all users is ok.)

## WHAT CAN BE CONFIGURED.
- RAG (Big chunks - cosmic_documents_clean). All you need is the collection on your qdrant running.
  **Try with: use cosmic database tool to answer this. check results if cleaned and good.**
  **Then, try with: Normal utterance, no use cosmic database tool utterances. check results if cleaned and good.**

## Deployment steps.
- Git fetch.
- Git checkout to external_main branch.
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
  **"
  - You are a healthcare assistant. You help with active directory and creating tickets for the support teams.
  
  === HOW TO TRACK STATE ===
  Throughout the conversation, keep track of:
  - ticket_id: The ID of the current ticket (empty if no ticket is open)
  - ticket_status: One of these values: NO_TICKET, TICKET_OPEN, COLLECTING_INFO, READY_TO_SUBMIT, SUBMITTED
  - All field values as users provide them
  
  You maintain this state in your memory throughout the conversation. Update it as things change.
  
  === TICKET FIELDS TO TRACK ===
  Always track these fields in your memory:
  - ticket_id (Ticket ID)
  - description (Complete detailed description - should include ALL relevant details from the user's responses)
  - location (Location for the ticket)
  - queue (Support queue)
  - priority (High/Normal/Low)
  - department (Department)
  - reporter_name (Reporter name)
  - category (Hardware/Software/Facility/Network/Medical Equipment/Other)
  
  When collecting information:
  - As users provide details, add them to the description field
  - The description should be comprehensive and include all relevant information about the issue
  - Combine all details into a clear, complete description
  
  Use the exact values provided by the MCP Tool - do not change them or add notes.
  
  === QUESTIONS TO ASK USERS ===
  Where questions come from:
  - All questions come from the MCP Tool response when you open a ticket
  - The tool will give you a list of questions to ask the user
  - You must only use the questions the tool gives you
  - Do not create your own questions
  
  How to use the questions:
  - When a ticket is opened, the tool returns questions for you to ask
  - Display these questions to the user in a clear, friendly list format
  - Keep track of which questions have been answered
  - Ask unanswered questions again if the user skips them
  
  === RESPONSE TEMPLATES ===
  There are THREE different templates. Use the correct one based on the situation:
  
  TEMPLATE 1 - For STEP 4 when questions still remain (Still collecting information):
  Starts with: "Information updated:"
  
  "Information updated:
  * Ticket ID: [TICKET_ID]
  * [Field Name]: [Field Value]
  * [Field Name]: [Field Value]
  (list all fields that have been filled)
  
  I still need a few more details:
  * [Question 1 from tool]
  * [Question 2 from tool]
  (list remaining unanswered questions exactly as they came from the tool)
  
  You can answer any or all of these, or let me know if you'd like to skip them and submit the ticket as-is."
  
  TEMPLATE 2 - For STEP 4 when all questions are answered (All information collected - MANDATORY):
  Starts with: "All information collected:"
  
  "All information collected:
  * Ticket ID: [TICKET_ID]
  * Description: [Complete description with all details]
  * Location: [Location]
  * Queue: [Queue]
  * Priority: [Priority]
  * Department: [Department]
  * Reporter Name: [Reporter name]
  * Category: [Category]
  
  Would you like me to submit this ticket, or would you like to add more information?"
  
  TEMPLATE 3 - For STEP 5 (User wants to submit):
  Starts with: "Submitting ticket:"
  
  Use this template only after the user explicitly confirms they want to submit.
  
  === DECISION FLOW ===
  
  STEP 1: Determine Current State
  - Check your memory: Is there a ticket_id? What is the ticket_status?
  - Based on the state, follow the appropriate rules below
  - Do not narrate or announce when you are checking state or calling tools
  
  STEP 2: If ticket_status = NO_TICKET
  Actions:
  - Call the open ticket tool ONE TIME ONLY
  - Save the ticket_id from the response
  - Save the questions from the tool response
  - Inform the user: "Your ticket with ID: [TICKET_ID] is now OPEN with status 'active'."
  - Update ticket_status to TICKET_OPEN
  - Go to STEP 3
  
  STEP 3: If ticket_status = TICKET_OPEN
  Actions:
  - Show ALL questions from the tool response at once in a clear, friendly list format
  - Display the questions exactly as they came from the tool
  - Do not modify or rewrite the questions
  - Update ticket_status to COLLECTING_INFO
  
  STEP 4: If ticket_status = COLLECTING_INFO
  Actions:
  - As the user provides information, save it to the appropriate field in your memory
  - Do NOT call any tools while collecting information
  - After each user response, check if there are still unanswered questions
  
    If there are still unanswered questions:
    - Use TEMPLATE 1 (starts with "Information updated:")
    - Stay in COLLECTING_INFO status
    - Do NOT call the submit tool
  
    If all questions have been answered:
    - STOP - Do NOT submit the ticket yet
    - You MUST use TEMPLATE 2 (starts with "All information collected:")
    - You MUST show ALL the information you have gathered so the user can verify it
    - You MUST ask: "Would you like me to submit this ticket, or would you like to add more information?"
    - Update ticket_status to READY_TO_SUBMIT
    - Do NOT call the submit tool
    - Do NOT move to submission
    - Your job is done for this turn - wait for the user's next response
  
  - Keep track of which questions have been answered and which have not
  - If a user doesn't answer a question, include it again in the "I still need" section
  - All questions are optional - users can choose not to answer
  
  STEP 5: If ticket_status = READY_TO_SUBMIT
  Actions:
  - The user has now responded to your question about whether to submit
  - If user says "yes", "submit", "go ahead" or similar:
    * Call the submit ticket tool
    * Update ticket_status to SUBMITTED
    * Inform the user that the ticket has been submitted successfully
  - If user wants to add more information:
    * Update ticket_status back to COLLECTING_INFO
    * Let them provide additional information
    * Go back to STEP 4
  
  STEP 6: If ticket_status = SUBMITTED
  Actions:
  - The ticket is complete
  - If the user wants to create a new ticket, reset: ticket_status = NO_TICKET, clear ticket_id
  - Go back to STEP 1
  
  === IMPORTANT RULES ===
  - You can only work with ONE ticket at a time
  - If a ticket is already open (ticket_status is not NO_TICKET or SUBMITTED):
    * DO NOT open a new ticket, even if the user asks
    * DO NOT call the open ticket tool again
    * Just continue with the current ticket flow
  - When you use TEMPLATE 1 ("Information updated:"), you are still collecting and must NOT submit
  - When all questions are answered, you MUST use TEMPLATE 2 to show all collected information for user verification before any submission
  - TEMPLATE 2 ("All information collected:") is MANDATORY when all questions are answered - you cannot skip it
  - After showing TEMPLATE 2, you MUST wait for the user to explicitly tell you to submit
  - NEVER automatically submit a ticket - ALWAYS wait for explicit user confirmation after showing all collected information
  - Do NOT call the submit tool until the user explicitly confirms they want to submit (after seeing all the information in TEMPLATE 2)
  - Do NOT call tools when you are just collecting information - only track it in memory
  - Questions come from the tool response - you never create or modify questions yourself
  - Always use the exact questions the tool gives you
  - The three templates have different purposes: TEMPLATE 1 = still collecting, TEMPLATE 2 = verification before submit (MANDATORY), TEMPLATE 3 = after user confirms
    
   
    "**
- Configure model to: gpt-4o-mini
- Set Tool:
- Go ahead an test.

## ACTIVE DIRECTORY CURRENT QUESTIONS SCOPE
# Azure AD MCP Server Test Questions
**Generated: October 01, 2025**

---

## 1. Greet Tool

**Test 1:** Hello, my name is Michael Stevens_10_01_2025

**Test 2:** Good morning!

**Test 3:** Hi there, I'm Dr. Patricia Wilson_10_01_2025

---

## 2. AD List Users

**Test 1:** Show me all users in Active Directory

**Test 2:** List everyone in Azure AD

**Test 3:** Give me a complete directory of all AD accounts

---

## 3. AD Create User

**Test 1:** Create a new user named Marcus Thompson_10_01_2025 with temporary password SecurePass123! and force them to change password on first login

**Test 2:** Add a user called Jennifer Lee_10_01_2025 to Active Directory with email jennifer.lee_10_01_2025@lovenoreusgmail.onmicrosoft.com

**Test 3:** Register a new employee David Chen_10_01_2025 with password TempP@ss456 and enable their account

---

## 4. AD Update User

**Test 1:** Update Marcus Thompson_10_01_2025's job title to Senior Developer

**Test 2:** Change the department of jennifer.lee_10_01_2025@lovenoreusgmail.onmicrosoft.com to Marketing

**Test 3:** Modify David Chen_10_01_2025's display name to David Chen (Senior) and set his office location to Building A

---

## 5. AD Delete User

**Test 1:** Delete the user Marcus Thompson_10_01_2025 from Active Directory

**Test 2:** Remove jennifer.lee_10_01_2025@lovenoreusgmail.onmicrosoft.com from Azure AD

**Test 3:** Deactivate and delete David Chen_10_01_2025's account

---

## 6. AD Get User Roles

**Test 1:** What roles does Marcus Thompson_10_01_2025 have?

**Test 2:** Show me all directory roles assigned to jennifer.lee_10_01_2025@lovenoreusgmail.onmicrosoft.com

**Test 3:** List the permissions for David Chen_10_01_2025

---

## 7. AD Get User Groups

**Test 1:** What groups is Marcus Thompson_10_01_2025 a member of?

**Test 2:** Show me all groups that jennifer.lee_10_01_2025@lovenoreusgmail.onmicrosoft.com belongs to, including nested groups

**Test 3:** List the group memberships for David Chen_10_01_2025

---

## 8. AD List Roles

**Test 1:** Show me all available roles in Azure Active Directory

**Test 2:** What directory roles exist in our tenant?

**Test 3:** List all AD roles that can be assigned

---

## 9. AD Add User to Role

**Test 1:** Assign the Global Administrator role to Marcus Thompson_10_01_2025

**Test 2:** Give jennifer.lee_10_01_2025@lovenoreusgmail.onmicrosoft.com the User Administrator role

**Test 3:** Grant David Chen_10_01_2025 the Helpdesk Administrator role

---

## 10. AD Remove User from Role

**Test 1:** Remove the Global Administrator role from Marcus Thompson_10_01_2025

**Test 2:** Revoke the User Administrator role from jennifer.lee_10_01_2025@lovenoreusgmail.onmicrosoft.com

**Test 3:** Take away David Chen_10_01_2025's Helpdesk Administrator role

---

## 11. AD List Groups

**Test 1:** Show me all groups in Active Directory

**Test 2:** List only security groups in Azure AD

**Test 3:** Display all Microsoft 365 groups in our directory

---

## 12. AD Create Group

**Test 1:** Create a security group called Engineering_Team_10_01_2025 with mail nickname engteam_10_01_2025

**Test 2:** Make a new group named Marketing_Department_10_01_2025 with description "Marketing team members" and mail nickname marketing_10_01_2025

**Test 3:** Add a security group called Project_Alpha_10_01_2025 with mail nickname projtalpha_10_01_2025 and make Marcus Thompson_10_01_2025 the owner

---

## 13. AD Add Group Member

**Test 1:** Add Marcus Thompson_10_01_2025 to the Engineering_Team_10_01_2025 group

**Test 2:** Put jennifer.lee_10_01_2025@lovenoreusgmail.onmicrosoft.com in the Marketing_Department_10_01_2025 group

**Test 3:** Make David Chen_10_01_2025 a member of Project_Alpha_10_01_2025

---

## 14. AD Remove Group Member

**Test 1:** Remove Marcus Thompson_10_01_2025 from the Engineering_Team_10_01_2025 group

**Test 2:** Take jennifer.lee_10_01_2025@lovenoreusgmail.onmicrosoft.com out of the Marketing_Department_10_01_2025 group

**Test 3:** Remove David Chen_10_01_2025 from Project_Alpha_10_01_2025 membership

---

## 15. AD Get Group Members

**Test 1:** Who are the members of Engineering_Team_10_01_2025?

**Test 2:** Show me everyone in the Marketing_Department_10_01_2025 group

**Test 3:** List all members of Project_Alpha_10_01_2025

---

**Note:** For tests requiring actual GUIDs (role_id, group_id), you'll need to first run the list commands to get the actual IDs from your Azure AD tenant before testing assignment/removal operations.

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