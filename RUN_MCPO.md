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
  "Always call a tool first. Tools have enough information to answer any user question. If tool cannot answer before you do. If any tool response returns an sql_query or code, display it in a properly formatted manner."
- Configure model to: gpt-4o-mini
- Set Tool:
- Go ahead an test.

## ACTIVE DIRECTORY

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