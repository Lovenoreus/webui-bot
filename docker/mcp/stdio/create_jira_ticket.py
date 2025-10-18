# tools/create_jira_ticket.py
import os
import requests
from typing import Tuple


def create_jira_ticket(
        conversation_id: str = "",
        conversation_topic: str = "",
        description: str = "",
        location: str = "",
        queue: str = "",
        priority: str = "",
        department: str = "",
        name: str = "",
        category: str = "",
        escalated_to: str = "",  # NEW
        escalated_to_email: str = ""  # NEW
) -> dict:  # Changed return type from str to dict
    """
    Creates a support ticket in Jira using the provided details.

    Args:
        conversation_id (str): Thread identifier to get stored fields
        conversation_topic (str): Short summary of the issue.
        description (str): Detailed explanation of the issue (should include "Problem Analysis" section).
        location (str): Where the issue occurred.
        queue (str): Which team/department is responsible.
        priority (str): Priority level of the issue.
        department (str): The user's department.
        name (str): Name of the person reporting.
        category (str): Ticket category.
        escalated_to (str): Name of escalator if ticket was escalated.  # NEW
        escalated_to_email (str): Email of escalator if ticket was escalated.  # NEW

    Returns:
        dict: Response with ticket key and status.
    """

    # Use provided values or fall back to stored values
    final_values = {
        'conversation_topic': conversation_topic,
        'description': description,
        'location': location,
        'queue': queue,
        'priority': priority,
        'department': department,
        'name': name,
        'category': category,
        'escalated_to': escalated_to,  # NEW
        'escalated_to_email': escalated_to_email  # NEW
    }

    print(f'📩 Creating Jira ticket for topic: "{final_values["conversation_topic"]}"')

    # Load config
    JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
    JIRA_EMAIL = os.getenv("JIRA_EMAIL")
    JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")

    print(f'🔧 Jira config - Domain: {JIRA_DOMAIN}, Email: {JIRA_EMAIL}, Token Set: {bool(JIRA_API_TOKEN)}')

    if not all([JIRA_API_TOKEN, JIRA_EMAIL, JIRA_DOMAIN]):
        msg = "❌ Jira configuration is incomplete. Please set environment variables properly."
        print(msg)
        return {"success": False, "error": msg}

    PROJECT_KEY = "HEAL"
    ISSUE_TYPE = "Task"
    CREATE_ISSUE_URL = f"https://{JIRA_DOMAIN}/rest/api/3/issue"
    TRANSITION_URL = f"https://{JIRA_DOMAIN}/rest/api/3/issue/{{}}/transitions"

    HEADERS = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # Format description
    content_blocks = []

    cleaned_description = final_values['description'].replace("Problem Analysis: ", "", 1) if final_values[
        'description'] else "No description provided"

    # Add escalation notice at the top if escalated  # NEW
    if final_values['escalated_to']:
        content_blocks.extend([
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "⚠️ ESCALATED TICKET", "marks": [{"type": "strong"}, {"type": "em"}]}
                ]
            },
            {
                "type": "paragraph",
                "content": [
                    {"type": "text",
                     "text": f"This ticket was created by {final_values['name']} but requires review and approval by "},
                    {"type": "text", "text": f"{final_values['escalated_to']}", "marks": [{"type": "strong"}]},
                    {"type": "text", "text": f" ({final_values['escalated_to_email']}) "},
                    {"type": "text", "text": "due to insufficient permissions."}
                ]
            },
            {
                "type": "rule"  # Horizontal line separator
            }
        ])

    content_blocks.extend([
        {
            "type": "paragraph",
            "content": [{"type": "text", "text": "Summary:", "marks": [{"type": "strong"}]}]
        },
        {
            "type": "paragraph",
            "content": [{"type": "text", "text": cleaned_description}]
        }
    ])

    def add_block(label: str, value: str):
        if value:  # Only add if value exists
            content_blocks.extend([
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": f"{label}:", "marks": [{"type": "strong"}]}]
                },
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": str(value)}]
                }
            ])

    add_block("Location", final_values['location'])
    add_block("Assigned Queue", final_values['queue'])
    add_block("Priority", final_values['priority'])
    add_block("Department", final_values['department'])
    add_block("Name", final_values['name'])
    add_block("Category", final_values['category'])
    add_block("Conversation Topic", final_values['conversation_topic'])

    content_blocks.append({
        "type": "paragraph",
        "content": [{"type": "text", "text": "Call Ended", "marks": [{"type": "strong"}]}]
    })

    # Build labels - add escalation label if escalated  # NEW
    labels = ["Tickets"]
    if final_values['escalated_to']:
        labels.append("Escalated")
        labels.append("Pending-Approval")

    payload = {
        "fields": {
            "project": {"key": PROJECT_KEY},
            "summary": final_values['conversation_topic'],
            "description": {
                "type": "doc",
                "version": 1,
                "content": content_blocks
            },
            "issuetype": {"name": ISSUE_TYPE},
            "labels": labels,  # Updated
            "components": [{"name": "Tickets"}]
        }
    }

    try:
        response = requests.post(
            CREATE_ISSUE_URL,
            headers=HEADERS,
            auth=(JIRA_EMAIL, JIRA_API_TOKEN),
            json=payload
        )

        if response.status_code == 201:
            issue_key = response.json()["key"]
            print(f"✅ Ticket {issue_key} successfully created!")

            # Transition the ticket
            transition_payload = {"transition": {"id": 51}}
            transition_response = requests.post(
                TRANSITION_URL.format(issue_key),
                headers=HEADERS,
                auth=(JIRA_EMAIL, JIRA_API_TOKEN),
                json=transition_payload
            )

            if transition_response.status_code == 204:
                print(f"✅ Ticket {issue_key} transitioned to 'Tickets'.")
            else:
                print(f"⚠️ Created ticket, but transition failed: {transition_response.json()}")

            # Return structured response  # NEW
            return {
                "success": True,
                "key": issue_key,
                "jira_key": issue_key,
                "message": f"✅ Ticket {issue_key} has been created successfully!",
                "escalated": bool(final_values['escalated_to']),
                "escalated_to": final_values['escalated_to'] if final_values['escalated_to'] else None
            }

        else:
            error_msg = response.json() if response.content else "Unknown error"
            return {
                "success": False,
                "error": f"Failed to create ticket: {error_msg}"
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Exception occurred: {str(e)}"
        }
