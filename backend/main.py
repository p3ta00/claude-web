"""
Claude Web - SDK Edition
Uses Claude Code SDK for faster responses (no subprocess spawn per message)
"""

import os
import json
import asyncio
import subprocess
import hashlib
import base64
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiosqlite
import re

# Claude Code SDK
from claude_code_sdk import query, ClaudeCodeOptions

def smart_compress(content: str, max_len: int = 2000) -> str:
    """Compress old messages but preserve critical dev/CTF info"""
    if len(content) <= max_len:
        return content

    # Patterns to always preserve (credentials, IPs, paths, code)
    preserved = []

    # Extract and preserve IPs
    ips = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', content)
    if ips:
        preserved.append(f"[IPs: {', '.join(set(ips))}]")

    # Extract credentials patterns
    creds = re.findall(r'(?:password|passwd|pwd|pass|user|username|login|cred)[:\s=]+[^\s\n]{3,50}', content, re.I)
    if creds:
        preserved.append(f"[Creds found: {'; '.join(set(creds[:5]))}]")

    # Extract file paths
    paths = re.findall(r'(?:/[\w\-./]+|C:\\[\w\-\\./]+)', content)
    if paths:
        unique_paths = list(set(paths))[:10]
        preserved.append(f"[Paths: {', '.join(unique_paths)}]")

    # Extract ports
    ports = re.findall(r'\b(?:port\s*)?(\d{2,5})(?:/tcp|/udp)?\b', content, re.I)
    if ports:
        preserved.append(f"[Ports: {', '.join(set(ports))}]")

    # Extract code blocks (keep first 500 chars of each)
    code_blocks = re.findall(r'```[\w]*\n(.*?)```', content, re.S)
    if code_blocks:
        for i, block in enumerate(code_blocks[:3]):
            if len(block) > 300:
                preserved.append(f"[Code block {i+1}: {block[:300]}...]")
            else:
                preserved.append(f"[Code: {block}]")

    # Extract URLs
    urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
    if urls:
        preserved.append(f"[URLs: {', '.join(set(urls)[:5])}]")

    # Extract hashes
    hashes = re.findall(r'\b[a-fA-F0-9]{32,64}\b', content)
    if hashes:
        preserved.append(f"[Hashes: {', '.join(set(hashes)[:3])}]")

    # Build compressed version
    preserved_str = "\n".join(preserved)
    remaining = max_len - len(preserved_str) - 100

    if remaining > 200:
        # Add truncated original content
        compressed = content[:remaining] + f"...\n[COMPRESSED - {len(content)} chars total]\n{preserved_str}"
    else:
        compressed = f"[COMPRESSED MESSAGE - {len(content)} chars]\n{preserved_str}"

    return compressed

# Configuration
DB_PATH = Path(__file__).parent / "data" / "conversations.db"
CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude")

# Ensure data directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """When you create or write a file, ALWAYS show the full code/content in your response using markdown code blocks. Never just say "Done" or "Created" without showing what you wrote."""


async def init_db():
    """Initialize SQLite database"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                folder TEXT DEFAULT NULL,
                pinned INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Add columns if they don't exist (for existing databases)
        try:
            await db.execute("ALTER TABLE conversations ADD COLUMN folder TEXT DEFAULT NULL")
        except:
            pass
        try:
            await db.execute("ALTER TABLE conversations ADD COLUMN pinned INTEGER DEFAULT 0")
        except:
            pass
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        # Knowledge/Memory system for learning from conversations
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                title TEXT,
                content TEXT,
                tags TEXT,
                source_conversation_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_conversation_id) REFERENCES conversations(id)
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)
        """)
        await db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Claude Web - CLI Wrapper", lifespan=lifespan)

# Try to import optional image generation module (gitignored for privacy)
try:
    from backend.imagegen import router as imagegen_router
    app.include_router(imagegen_router)
    IMAGEGEN_AVAILABLE = True
except ImportError:
    try:
        # Fallback for when running from backend directory
        from imagegen import router as imagegen_router
        app.include_router(imagegen_router)
        IMAGEGEN_AVAILABLE = True
    except ImportError:
        IMAGEGEN_AVAILABLE = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class MessageCreate(BaseModel):
    content: str


# API Routes
@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")


@app.get("/api/conversations")
async def list_conversations(folder: Optional[str] = None):
    """List all conversations, pinned first"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if folder:
            async with db.execute(
                "SELECT * FROM conversations WHERE folder = ? ORDER BY pinned DESC, updated_at DESC",
                (folder,)
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM conversations ORDER BY pinned DESC, updated_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(row) for row in rows]


@app.get("/api/conversations/search")
async def search_conversations(q: str):
    """Search across all conversations and messages"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        search_term = f"%{q}%"

        # Search in conversation titles and message content
        async with db.execute("""
            SELECT DISTINCT c.*,
                   (SELECT content FROM messages m
                    WHERE m.conversation_id = c.id AND m.content LIKE ?
                    LIMIT 1) as matching_content
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.title LIKE ? OR m.content LIKE ?
            ORDER BY c.pinned DESC, c.updated_at DESC
            LIMIT 50
        """, (search_term, search_term, search_term)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


@app.get("/api/folders")
async def list_folders():
    """Get all unique folders"""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT DISTINCT folder FROM conversations WHERE folder IS NOT NULL ORDER BY folder"
        ) as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]


@app.patch("/api/conversations/{conv_id}/pin")
async def toggle_pin_conversation(conv_id: str):
    """Toggle pin status of a conversation"""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT pinned FROM conversations WHERE id = ?", (conv_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Conversation not found")
            new_status = 0 if row[0] else 1

        await db.execute("UPDATE conversations SET pinned = ? WHERE id = ?", (new_status, conv_id))
        await db.commit()
    return {"pinned": bool(new_status)}


@app.patch("/api/conversations/{conv_id}/folder")
async def set_conversation_folder(conv_id: str, folder: Optional[str] = None):
    """Set folder for a conversation"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE conversations SET folder = ? WHERE id = ?", (folder, conv_id))
        await db.commit()
    return {"folder": folder}


@app.post("/api/conversations")
async def create_conversation(data: ConversationCreate):
    """Create new conversation"""
    conv_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]
    title = data.title or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conv_id, title)
        )
        await db.commit()

    return {"id": conv_id, "title": title}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Get conversation with messages"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ) as cursor:
            conv = await cursor.fetchone()
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")

        async with db.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conv_id,)
        ) as cursor:
            messages = await cursor.fetchall()

    return {
        "conversation": dict(conv),
        "messages": [dict(m) for m in messages]
    }


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete conversation"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        await db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        await db.commit()
    return {"status": "deleted"}


class ConversationUpdate(BaseModel):
    title: str


@app.patch("/api/conversations/{conv_id}")
async def update_conversation(conv_id: str, data: ConversationUpdate):
    """Rename conversation"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (data.title, conv_id)
        )
        await db.commit()
    return {"status": "updated", "title": data.title}


@app.post("/api/conversations/{conv_id}/auto-title")
async def auto_title_conversation(conv_id: str):
    """Use Claude to generate a title based on conversation content"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at LIMIT 6",
            (conv_id,)
        ) as cursor:
            messages = await cursor.fetchall()

    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found or empty")

    # Build conversation preview
    conv_preview = "\n".join([f"{m['role'].upper()}: {m['content'][:500]}" for m in messages])

    title_prompt = f"""Based on this conversation, generate a short, descriptive title (max 40 characters).
Return ONLY the title, nothing else. No quotes, no explanation.

CONVERSATION:
{conv_preview[:2000]}"""

    try:
        process = await asyncio.create_subprocess_exec(
            CLAUDE_CMD,
            "--print",
            "-p", title_prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        title = stdout.decode().strip()[:50]  # Limit to 50 chars

        if title:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (title, conv_id)
                )
                await db.commit()
            return {"title": title}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate title")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-title failed: {str(e)}")


# ==================== MEMORY/KNOWLEDGE SYSTEM ====================

class MemoryCreate(BaseModel):
    category: str  # e.g., "ctf", "technique", "exploit", "enumeration", "tool"
    title: str
    content: str
    tags: Optional[str] = None  # comma-separated tags
    source_conversation_id: Optional[str] = None


class MemoryUpdate(BaseModel):
    category: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[str] = None


@app.get("/api/memories")
async def list_memories(category: Optional[str] = None, search: Optional[str] = None):
    """List all memories, optionally filtered by category or search term"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        if search:
            # Search in title, content, and tags
            query = """
                SELECT * FROM memories
                WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?
                ORDER BY created_at DESC
            """
            search_term = f"%{search}%"
            async with db.execute(query, (search_term, search_term, search_term)) as cursor:
                rows = await cursor.fetchall()
        elif category:
            async with db.execute(
                "SELECT * FROM memories WHERE category = ? ORDER BY created_at DESC",
                (category,)
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM memories ORDER BY created_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()

        return [dict(row) for row in rows]


@app.post("/api/memories")
async def create_memory(data: MemoryCreate):
    """Create a new memory/knowledge entry"""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO memories (category, title, content, tags, source_conversation_id)
               VALUES (?, ?, ?, ?, ?)""",
            (data.category, data.title, data.content, data.tags, data.source_conversation_id)
        )
        memory_id = cursor.lastrowid
        await db.commit()

    return {"id": memory_id, "status": "created"}


@app.get("/api/memories/{memory_id}")
async def get_memory(memory_id: int):
    """Get a specific memory by ID"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Memory not found")
            return dict(row)


@app.patch("/api/memories/{memory_id}")
async def update_memory(memory_id: int, data: MemoryUpdate):
    """Update a memory entry"""
    updates = []
    values = []

    if data.category is not None:
        updates.append("category = ?")
        values.append(data.category)
    if data.title is not None:
        updates.append("title = ?")
        values.append(data.title)
    if data.content is not None:
        updates.append("content = ?")
        values.append(data.content)
    if data.tags is not None:
        updates.append("tags = ?")
        values.append(data.tags)

    if not updates:
        return {"status": "no changes"}

    values.append(memory_id)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
            values
        )
        await db.commit()

    return {"status": "updated"}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: int):
    """Delete a memory entry"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        await db.commit()
    return {"status": "deleted"}


@app.get("/api/memories/categories/list")
async def list_memory_categories():
    """Get all unique categories"""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT DISTINCT category FROM memories ORDER BY category"
        ) as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]


@app.post("/api/memories/extract/{conv_id}")
async def extract_memories_from_conversation(conv_id: str):
    """Use Claude to extract key learnings from a conversation"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conv_id,)
        ) as cursor:
            messages = await cursor.fetchall()

    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found or empty")

    # Build conversation text
    conv_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

    # Use Claude to extract key insights
    extract_prompt = f"""Analyze this conversation and extract key technical learnings that would be valuable to remember for future CTF challenges or security work.

For each insight, provide:
1. A category (one of: technique, exploit, enumeration, tool, configuration, vulnerability, privilege-escalation, lateral-movement, persistence, credential)
2. A short title (max 50 chars)
3. The key information/technique (be specific, include commands if relevant)
4. Relevant tags (comma-separated)

Format your response as JSON array:
[{{"category": "...", "title": "...", "content": "...", "tags": "..."}}]

Only extract genuinely useful technical insights. If there's nothing worth saving, return an empty array [].

CONVERSATION:
{conv_text[:8000]}"""  # Limit to avoid token limits

    try:
        process = await asyncio.create_subprocess_exec(
            CLAUDE_CMD,
            "--print",
            "-p", extract_prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        response_text = stdout.decode().strip()

        # Try to parse JSON from response
        import re
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group())

            # Save each extracted memory
            saved_count = 0
            async with aiosqlite.connect(DB_PATH) as db:
                for item in extracted:
                    if all(k in item for k in ['category', 'title', 'content']):
                        await db.execute(
                            """INSERT INTO memories (category, title, content, tags, source_conversation_id)
                               VALUES (?, ?, ?, ?, ?)""",
                            (item['category'], item['title'], item['content'],
                             item.get('tags', ''), conv_id)
                        )
                        saved_count += 1
                await db.commit()

            return {"status": "extracted", "count": saved_count, "memories": extracted}
        else:
            return {"status": "no_insights", "count": 0}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


async def get_relevant_memories(query: str, limit: int = 5) -> list:
    """Get memories relevant to a query (simple keyword matching for now)"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Extract keywords from query
        keywords = [w.lower() for w in query.split() if len(w) > 3]

        if not keywords:
            return []

        # Build search query
        conditions = []
        params = []
        for kw in keywords[:5]:  # Limit to 5 keywords
            conditions.append("(LOWER(title) LIKE ? OR LOWER(content) LIKE ? OR LOWER(tags) LIKE ?)")
            params.extend([f"%{kw}%", f"%{kw}%", f"%{kw}%"])

        query_sql = f"""
            SELECT * FROM memories
            WHERE {' OR '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        async with db.execute(query_sql, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


# ==================== END MEMORY SYSTEM ====================


@app.get("/api/auth/status")
async def auth_status():
    """Check if Claude CLI is available"""
    try:
        result = subprocess.run(
            [CLAUDE_CMD, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return {
            "authenticated": result.returncode == 0,
            "subscription_type": "cli-wrapper",
            "version": result.stdout.strip() if result.returncode == 0 else None
        }
    except Exception as e:
        return {
            "authenticated": False,
            "subscription_type": None,
            "error": str(e)
        }


@app.get("/api/modes")
async def get_available_modes():
    """Check which modes are available"""
    return {
        "claude": True,
        "imagegen": IMAGEGEN_AVAILABLE
    }


def conv_id_to_uuid(conv_id: str) -> str:
    """Convert conversation ID to a valid UUID for Claude CLI session"""
    # Create a deterministic UUID from the conversation ID
    hash_bytes = hashlib.md5(conv_id.encode()).digest()
    return str(uuid.UUID(bytes=hash_bytes))


@app.websocket("/ws/{conv_id}")
async def websocket_chat(websocket: WebSocket, conv_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            images_data = data.get("images", [])

            # Support legacy single image field
            if not images_data and data.get("image"):
                images_data = [data.get("image")]

            if not user_message and not images_data:
                continue

            # Handle images if present
            image_paths = []
            for image_data in images_data:
                try:
                    if "," in image_data:
                        image_data = image_data.split(",")[1]
                    image_bytes = base64.b64decode(image_data)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        f.write(image_bytes)
                        image_paths.append(f.name)
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Failed to process image: {str(e)}"
                    })

            # Save user message to DB
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    (conv_id, "user", user_message + (f" [{len(image_paths)} image(s) attached]" if image_paths else ""))
                )
                await db.execute(
                    "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (conv_id,)
                )
                await db.commit()

            # Check if conversation has prior messages (for --resume)
            has_prior_messages = False
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (conv_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    has_prior_messages = row[0] > 1  # More than just the current message

            # Build prompt
            prompt_to_send = user_message
            if image_paths:
                paths_str = "\n".join([f"- {p}" for p in image_paths])
                prompt_to_send = f"I'm sharing {len(image_paths)} image(s) with you. Please analyze the images at:\n{paths_str}\n\n{user_message}"

            # Use subprocess with streaming - use --resume for conversation continuity

            # Create deterministic session ID from conversation ID
            session_id = conv_id_to_uuid(conv_id)

            cmd_args = [
                CLAUDE_CMD,
                "--print",
                "--output-format", "stream-json",
                "--verbose",
                "--dangerously-skip-permissions",
                "--model", "opus",
            ]

            # Use --resume for continuing conversations
            if has_prior_messages:
                cmd_args.extend(["--resume", session_id])
            else:
                cmd_args.extend(["--session-id", session_id])

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # Send prompt
                process.stdin.write(prompt_to_send.encode())
                await process.stdin.drain()
                process.stdin.close()

                full_response = ""

                # Stream output
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode().strip()
                    if not line_str:
                        continue

                    try:
                        event = json.loads(line_str)
                        event_type = event.get("type", "")

                        if event_type == "assistant":
                            message = event.get("message", {})
                            for block in message.get("content", []):
                                if block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text:
                                        full_response += text
                                        await websocket.send_json({"type": "text", "content": text})
                                elif block.get("type") == "tool_use":
                                    await websocket.send_json({
                                        "type": "tool_call",
                                        "tool": block.get("name", "unknown"),
                                        "input": block.get("input", {})
                                    })

                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    full_response += text
                                    await websocket.send_json({"type": "text", "content": text})

                    except json.JSONDecodeError:
                        text = line.decode().strip()
                        if text:
                            full_response += text + "\n"
                            await websocket.send_json({"type": "text", "content": text + "\n"})

                await process.wait()

                # Send completion
                await websocket.send_json({"type": "done"})

                # Save assistant response
                if full_response:
                    async with aiosqlite.connect(DB_PATH) as db:
                        await db.execute(
                            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                            (conv_id, "assistant", full_response)
                        )
                        await db.commit()

                # Cleanup temp image files
                for img_path in image_paths:
                    try:
                        os.unlink(img_path)
                    except:
                        pass

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                })
                await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


# Mount static files
static_path = Path(__file__).parent.parent / "frontend" / "static"
static_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
