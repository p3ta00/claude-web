"""
Claude Web - CLI Wrapper Edition
Routes through Claude Code CLI to use your subscription
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

# Configuration
DB_PATH = Path(__file__).parent / "data" / "conversations.db"
CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude")

# Ensure data directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are Claude, a helpful AI assistant. You have full access to the host system and can execute commands, read/write files, and help with various tasks.

Key capabilities:
- Execute shell commands
- Read and write files anywhere on the system
- Help with coding, writing, analysis, and more
- Assist with technical and creative tasks

Guidelines:
- Be helpful, harmless, and honest
- Explain your reasoning when helpful
- Ask clarifying questions when needed"""


async def init_db():
    """Initialize SQLite database"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
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
        await db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Claude Web", lifespan=lifespan)

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
async def list_conversations():
    """List all conversations"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


@app.post("/api/conversations")
async def create_conversation(data: ConversationCreate):
    """Create new conversation"""
    conv_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]
    title = data.title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"

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
            "version": result.stdout.strip() if result.returncode == 0 else None
        }
    except Exception as e:
        return {
            "authenticated": False,
            "error": str(e)
        }


def conv_id_to_uuid(conv_id: str) -> str:
    """Convert conversation ID to a valid UUID for Claude CLI session"""
    hash_bytes = hashlib.md5(conv_id.encode()).digest()
    return str(uuid.UUID(bytes=hash_bytes))


@app.websocket("/ws/{conv_id}")
async def websocket_chat(websocket: WebSocket, conv_id: str):
    """WebSocket endpoint for real-time chat via Claude CLI"""
    await websocket.accept()

    session_uuid = conv_id_to_uuid(conv_id)

    has_history = False
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT COUNT(*) as count FROM messages WHERE conversation_id = ?",
            (conv_id,)
        ) as cursor:
            row = await cursor.fetchone()
            has_history = row["count"] > 0

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            images_data = data.get("images", [])

            if not images_data and data.get("image"):
                images_data = [data.get("image")]

            if not user_message and not images_data:
                continue

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

            try:
                cmd_args = [
                    CLAUDE_CMD,
                    "--print",
                    "--output-format", "stream-json",
                    "--verbose",
                    "--dangerously-skip-permissions",
                    "--append-system-prompt", SYSTEM_PROMPT,
                ]

                if has_history:
                    cmd_args.extend(["--resume", session_uuid])
                else:
                    cmd_args.extend(["--session-id", session_uuid])

                process = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=10*1024*1024
                )

                prompt_to_send = user_message
                if image_paths:
                    paths_str = "\n".join([f"- {p}" for p in image_paths])
                    prompt_to_send = f"I'm sharing {len(image_paths)} image(s) with you. Please analyze the images at:\n{paths_str}\n\n{user_message}"

                process.stdin.write(prompt_to_send.encode())
                await process.stdin.drain()
                process.stdin.close()

                full_response = ""

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
                            content = message.get("content", [])
                            for block in content:
                                if block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text:
                                        full_response += text
                                        await websocket.send_json({
                                            "type": "text",
                                            "content": text
                                        })
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
                                    await websocket.send_json({
                                        "type": "text",
                                        "content": text
                                    })

                        elif event_type == "result":
                            result_text = event.get("result", "")
                            if result_text:
                                await websocket.send_json({
                                    "type": "tool_result",
                                    "tool": "command",
                                    "result": result_text[:5000]
                                })

                        elif event_type == "error":
                            await websocket.send_json({
                                "type": "error",
                                "message": event.get("error", {}).get("message", "Unknown error")
                            })

                    except json.JSONDecodeError:
                        text = line.decode().strip()
                        if text:
                            full_response += text + "\n"
                            await websocket.send_json({
                                "type": "text",
                                "content": text + "\n"
                            })

                await process.wait()

                stderr_output = await process.stderr.read()
                if stderr_output and not full_response:
                    stderr_text = stderr_output.decode().strip()
                    if stderr_text:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"CLI stderr: {stderr_text[:500]}"
                        })

                await websocket.send_json({"type": "done"})

                if full_response:
                    async with aiosqlite.connect(DB_PATH) as db:
                        await db.execute(
                            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                            (conv_id, "assistant", full_response)
                        )
                        await db.commit()

                has_history = True

                for img_path in image_paths:
                    try:
                        os.unlink(img_path)
                    except:
                        pass

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"CLI error: {str(e)}"
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
