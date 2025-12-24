# Claude Web

A mobile-friendly web interface for Claude Code CLI. Uses your existing Claude subscription - **no API costs**.

```
  ██████╗██╗      █████╗ ██╗   ██╗██████╗ ███████╗    ██╗    ██╗███████╗██████╗
 ██╔════╝██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝    ██║    ██║██╔════╝██╔══██╗
 ██║     ██║     ███████║██║   ██║██║  ██║█████╗      ██║ █╗ ██║█████╗  ██████╔╝
 ██║     ██║     ██╔══██║██║   ██║██║  ██║██╔══╝      ██║███╗██║██╔══╝  ██╔══██╗
 ╚██████╗███████╗██║  ██║╚██████╔╝██████╔╝███████╗    ╚███╔███╔╝███████╗██████╔╝
  ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝     ╚══╝╚══╝ ╚══════╝╚═════╝
```

## Features

- **No API Costs** - Routes through Claude Code CLI, uses your Pro/Max subscription
- **Mobile-Friendly** - Clean, responsive UI that works great on phones
- **Full System Access** - Execute commands, read/write files, make web requests
- **Chat History** - Persistent conversations with rename/delete support
- **Image Support** - Upload multiple images or paste from clipboard (Ctrl+V)
- **Real-time Streaming** - See responses as they're generated
- **Tool Visualization** - Watch tool calls execute in real-time
- **Syntax Highlighting** - Beautiful code blocks with copy button

## Requirements

- **Claude Code CLI** - Authenticated with your Anthropic account
- **Python 3.8+**
- **Node.js** (for installing Claude Code CLI)
- **Claude Pro or Max subscription** (for CLI access)

## Quick Start

### 1. Install Claude Code CLI

```bash
npm install -g @anthropic-ai/claude-code
```

### 2. Authenticate Claude Code

```bash
claude
```

This opens your browser to log in with your Anthropic account.

### 3. Setup Claude Web

```bash
git clone https://github.com/p3ta00/claude-web.git
cd claude-web
chmod +x setup.sh run.sh
./setup.sh
```

### 4. Run

```bash
./run.sh
```

Open http://localhost:8888 in your browser.

## How It Works

```
Your Browser → Claude Web UI → Claude Code CLI → Your Subscription → Response
```

Instead of calling the Anthropic API directly (which requires separate billing), Claude Web pipes your messages through the Claude Code CLI. Since Claude Code uses your Pro/Max subscription, there are no additional API charges.

## Usage

- **New Chat** - Click "+ New" or just start typing
- **Send Message** - Type and press Enter or click the send button
- **Attach Images** - Click the paperclip icon or paste with Ctrl+V
- **Rename Chat** - Hover over a chat and click the pencil icon
- **Delete Chat** - Hover over a chat and click the trash icon
- **Mobile** - Tap the hamburger menu to access chat history

## Security Note

This tool runs with `--dangerously-skip-permissions` to enable full system access. Only use it on systems you own or have authorization to access.

## License

MIT
