#!/usr/bin/env python3
"""
Chat History Data Extraction Script for Open WebUI

This script helps you access and view chat history data from the Open WebUI database.
It can display all chats, chats for specific users, or individual chat details.
"""

import sqlite3
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

def get_database_path():
    """Get the path to the Open WebUI database."""
    # Check if DATABASE_URL environment variable is set
    database_url = os.environ.get("DATABASE_URL")
    if database_url and database_url.startswith("sqlite:///"):
        return database_url.replace("sqlite:///", "")
    
    # Default path based on the project structure
    current_dir = Path(__file__).parent
    data_dir = current_dir / "backend" / "data"
    
    # Check both possible locations
    possible_paths = [
        data_dir / "webui.db",
        current_dir / "backend" / "open_webui" / "data" / "webui.db",
        Path.home() / ".open-webui" / "data" / "webui.db",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return str(data_dir / "webui.db")

def connect_to_database(db_path: str):
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This allows us to access columns by name
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def format_timestamp(timestamp: int) -> str:
    """Convert Unix timestamp to readable format."""
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "Unknown"

def get_all_users(conn) -> List[Dict[str, Any]]:
    """Get all users from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, role, created_at FROM user ORDER BY created_at DESC")
    users = []
    for row in cursor.fetchall():
        users.append({
            'id': row['id'],
            'name': row['name'],
            'email': row['email'],
            'role': row['role'],
            'created_at': format_timestamp(row['created_at'])
        })
    return users

def get_user_chats(conn, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get chats for a specific user or all chats if user_id is None."""
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute("""
            SELECT id, user_id, title, created_at, updated_at, archived, pinned 
            FROM chat 
            WHERE user_id = ? 
            ORDER BY updated_at DESC
        """, (user_id,))
    else:
        cursor.execute("""
            SELECT id, user_id, title, created_at, updated_at, archived, pinned 
            FROM chat 
            ORDER BY updated_at DESC
        """)
    
    chats = []
    for row in cursor.fetchall():
        chats.append({
            'id': row['id'],
            'user_id': row['user_id'],
            'title': row['title'],
            'created_at': format_timestamp(row['created_at']),
            'updated_at': format_timestamp(row['updated_at']),
            'archived': bool(row['archived']),
            'pinned': bool(row['pinned'])
        })
    return chats

def get_chat_details(conn, chat_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific chat including messages."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat WHERE id = ?", (chat_id,))
    row = cursor.fetchone()
    
    if not row:
        return None
    
    chat_data = json.loads(row['chat']) if row['chat'] else {}
    
    return {
        'id': row['id'],
        'user_id': row['user_id'],
        'title': row['title'],
        'created_at': format_timestamp(row['created_at']),
        'updated_at': format_timestamp(row['updated_at']),
        'archived': bool(row['archived']),
        'pinned': bool(row['pinned']),
        'chat_data': chat_data,
        'messages': chat_data.get('history', {}).get('messages', {})
    }

def extract_messages_from_chat(chat_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and format messages from chat data."""
    messages = chat_data.get('history', {}).get('messages', {})
    formatted_messages = []
    
    for msg_id, message in messages.items():
        formatted_messages.append({
            'id': msg_id,
            'role': message.get('role', 'unknown'),
            'content': message.get('content', ''),
            'timestamp': message.get('timestamp', 0)
        })
    
    # Sort by timestamp if available
    formatted_messages.sort(key=lambda x: x['timestamp'])
    return formatted_messages

def main():
    """Main function to demonstrate usage."""
    db_path = get_database_path()
    print(f"Looking for database at: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        print("Make sure Open WebUI has been run at least once to create the database.")
        return
    
    conn = connect_to_database(db_path)
    if not conn:
        return
    
    try:
        print("\n=== Open WebUI Chat Data Extraction ===\n")
        
        # Get all users
        print("1. All Users:")
        users = get_all_users(conn)
        for user in users:
            print(f"  ID: {user['id']}")
            print(f"  Name: {user['name']}")
            print(f"  Email: {user['email']}")
            print(f"  Role: {user['role']}")
            print(f"  Created: {user['created_at']}")
            print()
        
        if not users:
            print("  No users found in database.")
            return
        
        # Get all chats
        print("2. All Chats:")
        chats = get_user_chats(conn)
        for chat in chats[:10]:  # Show first 10 chats
            print(f"  ID: {chat['id']}")
            print(f"  User: {chat['user_id']}")
            print(f"  Title: {chat['title']}")
            print(f"  Updated: {chat['updated_at']}")
            print(f"  Archived: {chat['archived']}, Pinned: {chat['pinned']}")
            print()
        
        if len(chats) > 10:
            print(f"  ... and {len(chats) - 10} more chats")
        
        # Show detailed view of first chat if available
        if chats:
            print(f"3. Detailed view of first chat:")
            first_chat = get_chat_details(conn, chats[0]['id'])
            if first_chat:
                print(f"  Chat ID: {first_chat['id']}")
                print(f"  Title: {first_chat['title']}")
                print(f"  User: {first_chat['user_id']}")
                print(f"  Messages:")
                
                messages = extract_messages_from_chat(first_chat)
                for i, msg in enumerate(messages[:5]):  # Show first 5 messages
                    print(f"    Message {i+1}:")
                    print(f"      Role: {msg['role']}")
                    print(f"      Content: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                    print()
                
                if len(messages) > 5:
                    print(f"    ... and {len(messages) - 5} more messages")
    
    finally:
        conn.close()

def get_user_chat_history(user_id: str):
    """Get chat history for a specific user."""
    db_path = get_database_path()
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return None
    
    conn = connect_to_database(db_path)
    if not conn:
        return None
    
    try:
        chats = get_user_chats(conn, user_id)
        detailed_chats = []
        
        for chat in chats:
            chat_details = get_chat_details(conn, chat['id'])
            if chat_details:
                detailed_chats.append(chat_details)
        
        return detailed_chats
    finally:
        conn.close()

if __name__ == "__main__":
    main()