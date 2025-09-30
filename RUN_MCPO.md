# Guide on how to deploy the MCPO integrated version of the omnigate Test Bot.
## WHAT IS LACKING.
- Praveen's RAG improvements.
- PGvector database.
- List all tool. (MCPO probably blocks this by default. Need to do more research on this.)

## WHAT CAN BE TESTED.
- ACTIVE DIRECTORY (No need to say from active directory. List all users is ok.)
- SQL DATABASE (No need to say from the database all the time. Sample questions can be found at the bottom of this .md)

## WHAT CAN BE CONFIGURED.
- RAG (Big chunks - cosmic_documents_clean). All you need is the collection on your qdrant running.
  **Try with: use cosmic database tool to answer this. check results if cleaned and good.**
  **Then, try with: Normal utterance, no use cosmic database tool utterances. check results if cleaned and good.**

## Deployment steps.
- Git fetch.
- Git checkout to regional_mcpo branch.
- docker compose down.
- Destroy all current docker images.
- navigate to webui root.
- create ENV: https://healthcaretechglobal.slack.com/files/U087JMT2XNJ/F09HZUB45T6/mcpo_.env
- create a directory called: local-binaries in the root directory
- copy the slack tgz file into it: https://healthcaretechglobal.slack.com/archives/C08CLG263D2/p1759219738726859
- Run command: docker compose build --no-cache
- If above command gives issue, do: docker compose up
- After build, run command: docker compose up
- After webui login, navigate to system prompt and inject this: 
  "Always call a tool first. Tools have enough information to answer any user question. If tool cannot answer before you do. If any tool response returns an sql_query or code, display it in a properly formatted manner."
- Configure model to: gpt-4o-mini
- Set Tool:
- Go ahead an test.

## SQL DATABASE SCHEMA (SQLITE)
- Note: Schema no longer deals with users. That is active directory stuff now.

### POSSIBLE QUESTIONS FOR THE SQL DATABASE
Here are the updated 10 test questions with state abbreviations included:

#### Simple Queries (Single Table)

1. **"How many healthcare facilities are in the database?"**
   - Keywords: `["healthcare", "facilities", "count"]`
   - Expected SQL: `SELECT COUNT(*) FROM HealthcareFacilities`
   - Expected result: 7

2. **"Show me the hospital in New York (NY)"**
   - Keywords: `["hospital", "New York", "NY"]`
   - Expected SQL: `SELECT * FROM HealthcareFacilities WHERE State = 'NY' AND Type = 'Hospital'`
   - Expected result: 1 (Metro General Hospital)

3. **"List all medical centers"**
   - Keywords: `["medical", "centers"]`
   - Expected SQL: `SELECT * FROM HealthcareFacilities WHERE Type = 'Medical Center'`
   - Expected result: 1 (Westside Medical Center)

4. **"What medical services cost less than $100?"**
   - Keywords: `["medical", "services", "cost", "100"]`
   - Expected SQL: `SELECT * FROM MedicalServicesCatalog WHERE BasePrice < 100`
   - Expected result: Multiple lab services

#### Medium Complexity (Joins)

5. **"Which facilities offer blood tests?"**
   - Keywords: `["facilities", "blood", "tests"]`
   - Expected SQL: `SELECT f.Name, s.ServiceName FROM HealthcareFacilities f JOIN FacilityServices fs ON f.FacilityID = fs.FacilityID JOIN MedicalServicesCatalog s ON fs.ServiceID = s.ServiceID WHERE s.ServiceName LIKE '%Blood%'`
   - Expected result: Multiple facilities

6. **"Show inventory items at Metro General Hospital"**
   - Keywords: `["inventory", "Metro General Hospital"]`
   - Expected SQL: `SELECT i.* FROM MedicalInventory i JOIN HealthcareFacilities f ON i.FacilityID = f.FacilityID WHERE f.Name LIKE '%Metro General%'`
   - Expected result: Multiple inventory items

7. **"What imaging services are available in Houston (TX)?"**
   - Keywords: `["imaging", "services", "Houston", "TX"]`
   - Expected SQL: `SELECT f.Name, s.ServiceName FROM HealthcareFacilities f JOIN FacilityServices fs ON f.FacilityID = fs.FacilityID JOIN MedicalServicesCatalog s ON fs.ServiceID = s.ServiceID WHERE f.City = 'Houston' AND s.Department = 'Radiology'`
   - Expected result: Services at Central Imaging Center

#### Complex Queries (Multiple Joins + Aggregation)

8. **"Which insurance providers cover laboratory services?"**
   - Keywords: `["insurance", "providers", "laboratory"]`
   - Expected SQL: `SELECT DISTINCT ip.ProviderName FROM InsuranceProviders ip JOIN InsuranceCoverage ic ON ip.ProviderID = ic.ProviderID JOIN MedicalServicesCatalog s ON ic.ServiceID = s.ServiceID WHERE s.Department = 'Laboratory'`
   - Expected result: Multiple providers

9. **"Show all facilities in Texas (TX) with their services"**
   - Keywords: `["facilities", "Texas", "TX", "services"]`
   - Expected SQL: `SELECT f.Name, s.ServiceName FROM HealthcareFacilities f JOIN FacilityServices fs ON f.FacilityID = fs.FacilityID JOIN MedicalServicesCatalog s ON fs.ServiceID = s.ServiceID WHERE f.State = 'TX'`
   - Expected result: 2 facilities (Central Imaging Center, Eastside Specialty Clinic)

10. **"Which facilities have inventory expiring before 2026?"
   - Keywords: ["facilities", "inventory", "expiring", "2026"]
   - Expected SQL: SELECT f.Name, i.ItemName, i.ExpiryDate FROM HealthcareFacilities f JOIN MedicalInventory i ON f.FacilityID = i.FacilityID WHERE i.ExpiryDate < '2026-01-01' ORDER BY i.ExpiryDate ASC 
   - Expected result: 4 items from 3 facilities (Blood Collection Tubes at Riverside Lab, X-Ray Film at Central Imaging, Contrast Dye at Central Imaging, Disposable Gloves at Metro General)


# In-case of any deployment issues
- Contact: **The one who shall not be named**