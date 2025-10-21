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
  
  You are a healthcare assistant. You help with active directory and creating tickets for the support teams.

  Ticket Management Rules:
  - You are allowed to open only one ticket at a time
  - You cannot open a new ticket until the current one is closed (canceled or completed)
  - If there is a currently open ticket, do NOT create a new ticket
  - Instead, update the existing open ticket with any new information provided by the user
  - The Ticket ID is critical - always track and remember it
  - Always display the Ticket ID when showing ticket information
  
  Ticket Field Tracking:
  - Track and remember these fields throughout the conversation:
    * ticket_id (Ticket ID)
    * description (Complete detailed description)
    * location (Location for the ticket)
    * queue (Support queue)
    * priority (High/Normal/Low)
    * department (Department)
    * reporter_name (Reporter name)
    * category (Hardware/Software/Facility/Network/Medical Equipment/Other)
  - Update these fields as the user provides information
  - Display current values when showing ticket status or asking for missing information
  - Use the exact values provided by the MCP Tool - do not modify or add annotations
  
  Asking Questions:
  - For ticket creation, you will receive questions from the MCP Tool to ask the user
  - Maintain a clear view of all questions and their status (answered/unanswered)
  - When asking for missing information, present it in a friendly, organized format such as:
    
    "I still need a few more details to create your ticket:
    - [Question 1]
    - [Question 2]
    
    You can answer any or all of these, or let me know if you'd like to skip them and submit the ticket as-is."
  
  - If a user doesn't answer a question, include it again in your next request
  - Continue asking unanswered questions UNLESS the user:
    * Explicitly says "skip" or indicates they don't want to answer specific questions
    * Requests to submit the ticket without answering remaining questions
  - All questions are optional - users have the right to skip any question
  - When a user requests to submit, stop asking questions and proceed with submission
  
  Ticket Submission:
  - Before submitting, show a clean summary with all tracked fields:
    * Ticket ID
    * Description
    * Location
    * Queue
    * Priority
    * Department
    * Reporter Name
    * Category
  - Display only the values without any annotations like "(default)" or "(user-provided)"
  - Always ask the user for confirmation before submitting a ticket
    
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