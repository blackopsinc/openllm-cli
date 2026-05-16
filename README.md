# openllm-cli

An interactive LLM CLI with filesystem access, autonomous tool use, and skills integration. Defaults to Ollama (local, no API key). Also supports Anthropic, OpenAI, OpenRouter, and LM Studio.

## Features

- **Interactive REPL** — multi-turn conversation with history, slash commands, and inline file/command injection
- **Auto mode** — fully autonomous: the LLM reads/writes files and runs shell commands to complete any task
- **Skills mode** — controlled execution driven by a `SKILL.md` file in your working directory
- **Pipe mode** — single-shot stdin → stdout for scripting and shell pipelines
- **Streaming** — real-time token output in chat mode; tool actions streamed live in auto mode
- **Multi-provider** — one binary, five backends
- **Small-model friendly** — tolerant XML parsing handles common output quirks from local models

## Installation

**Prerequisites:** Go 1.22+

```bash
# Build for current platform
make build

# Install to ~/.local/bin (no sudo)
make install-user

# Install system-wide
make install

# Cross-compile for all platforms
make all
```

Binaries land in `bin/`. The `make install-user` target adds the binary to `~/.local/bin` — make sure that's on your `$PATH`.

## Quick Start

### Interactive REPL (default — Ollama, no API key needed)

```bash
ollama serve                       # start Ollama if not already running
ollama pull gemma3:4b              # pull a model (one-time)

source env/environment_ollama      # sets LLM_PROVIDER=ollama + model
openllm-cli                        # starts the REPL
```

### Auto mode — any task

```bash
openllm-cli -a
> do research on the top 5 Go web frameworks and write a comparison to frameworks.md
> create a REST API in Go that reads from a CSV file
> find all TODO comments in this repo and list them
```

The agent uses shell commands, reads and writes files, and reports what it did — no prompts, no confirmation steps.

### Pipe mode

```bash
echo "summarise this" | openllm-cli
cat app.log | openllm-cli
git diff | openllm-cli
```

## Flags

| Flag | Description |
|------|-------------|
| `-i`, `--interactive` | Force interactive REPL |
| `-a`, `--auto` | Auto mode: fully autonomous tool use, no confirmations |
| `-s`, `--skills` | Skills mode: read `SKILL.md` for allowed commands/paths |
| `-h`, `--help` | Show help |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` · `anthropic` · `openai` · `openrouter` · `lmstudio` |
| `LLM_MODEL` | Provider default | Model name |
| `LLM_SYSTEM_PROMPT` | — | System prompt |
| `LLM_STREAM` | `false` | Enable streaming (`1`, `true`, `yes`, `on`) |
| `LLM_MAX_TOKENS` | `8096` | Max tokens to generate |
| `LLM_TIMEOUT` | `120` / `300` streaming | LLM request timeout (seconds) |
| `LLM_SHELL_TIMEOUT` | `60` | Shell command timeout (seconds) |
| `LLM_AUTO_APPROVE` | `false` | Skip tool confirmations in non-auto modes |
| `LLM_SKILLS_MODE` | `false` | Enable skills mode by default |
| `LLM_VERBOSE` | `false` | Debug logging (shows requests, model responses, parsing details) |
| `LLM_INTERACTIVE` | `false` | Force interactive REPL even when stdin is piped |
| `OPENROUTER_API_KEY` | — | Required for `openrouter` |
| `OPENAI_API_KEY` | — | Required for `openai` |
| `ANTHROPIC_API_KEY` | — | Required for `anthropic` |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama endpoint |
| `LM_STUDIO_URL` | `http://localhost:1234/v1/chat/completions` | LM Studio endpoint |

### Default models

| Provider | Default model |
|----------|---------------|
| **Ollama** *(default)* | `gemma3:4b` |
| Anthropic | `claude-sonnet-4-20250514` |
| OpenAI | `gpt-4o-mini` |
| OpenRouter | `openai/gpt-4o-mini` |
| LM Studio | `local-model` |

## Provider Setup

### Ollama — default, no API key (recommended)

```bash
ollama serve
ollama pull gemma3:4b              # or any model from https://ollama.com/library
source env/environment_ollama
openllm-cli
```

### Anthropic

```bash
source env/environment_anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
openllm-cli
```

### OpenAI

```bash
source env/environment_openai
export OPENAI_API_KEY="sk-..."
openllm-cli
```

### OpenRouter

```bash
source env/environment_openrouter
export OPENROUTER_API_KEY="sk-or-..."
openllm-cli
```

### LM Studio (local, no key needed)

```bash
# In LM Studio: load a model, then enable the local server under the Developer tab
source env/environment_lmstudio
openllm-cli
```

## Slash Commands (REPL)

### Session

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/exit` · `/quit` · `/q` | Quit |
| `/clear` · `/reset` | Clear conversation history |
| `/history` | Show message history |

### Config

| Command | Description |
|---------|-------------|
| `/model [name]` | Get or set model |
| `/system [text]` | Get or set system prompt |
| `/stream` | Toggle streaming |
| `/auto` | Toggle auto mode (fully autonomous) |
| `/skills` | Toggle skills mode |
| `/approve` | Toggle auto-approve for tool actions |
| `/maxtokens [n]` | Get or set max_tokens |

### Filesystem & Shell

| Command | Description |
|---------|-------------|
| `/read <path>` | Read file/dir; optionally send to LLM |
| `/write <path>` | Write last LLM reply (or typed content) to file |
| `/ls [path]` | List directory |
| `/pwd` | Show working directory |
| `/cd <path>` | Change working directory |
| `/run <cmd>` | Run shell command; optionally send output to LLM |

## Inline Syntax

Works in any message — the token is expanded before the LLM sees it:

```
@path/to/file          inject file contents into the prompt
`command`              run command and inject output into the prompt
```

Examples:
```
> review @main.go and find any bugs
> `git diff HEAD~1` explain what changed
> `go test ./...` what do these failures mean?
```

## Auto Mode

Auto mode (`-a`) lets the model complete tasks by reading/writing files and running shell commands. It works for any kind of task — research, file manipulation, code, data processing, system inspection:

```
> do research on quantum computing and write a summary to research.md
> create a REST API in Go that reads from data.csv
> find all .log files older than 7 days and delete them
> what Go packages does this project import?
```

### What you see

Raw XML tool calls are never shown. Instead, each action is displayed as a clean line:

```
 📖  read ./main.go
 ✏️   write ./output.md  (1.2 KB)
 ⚙️   go build -o ./app .
 ✓ Done  Built and ran the app. Output was: Hello, world!
```

Shell commands stream output to the terminal in real time. When streaming is enabled (`LLM_STREAM=1`), the model's reasoning is buffered silently in auto mode to avoid showing raw tool-call XML — only the formatted tool actions appear.

### Approach

The agent picks the right approach for the task:
- **Research / fetch data** — `curl`, `grep`, `jq` via shell
- **Read or summarise files** — `read_file`, then answer
- **Write or edit files** — `write_file`
- **System tasks** — shell commands
- **Write and run code** — only when explicitly asked

### Loop protection

If the model gets stuck repeating the same error (e.g. a malformed tool call), the loop breaks after 3 identical results and reports what went wrong. This prevents runaway inference on low-capability models.

## Non-Auto Mode

In regular chat mode, if the model outputs a tool call (some models do this regardless of instructions), the raw XML is stripped from the display. You'll see a hint instead:

```
[Model wants to use tools — type /auto or restart with -a to enable auto mode]
```

Any prose the model wrote before the tool call is shown normally.

## Skills Mode

Create a `SKILL.md` in your working directory to control what the agent can do:

```markdown
ALLOWED_COMMANDS: go,git,ls,cat,echo
ALLOWED_PATHS: .,./src,/tmp
DISALLOWED_PATHS: ~/.ssh,/etc
AUTO_EXECUTE: true

## Instructions
You are a Go development assistant. Help the user build and test Go code.
```

Start with `openllm-cli -s` or toggle with `/skills` in the REPL. When `AUTO_EXECUTE: true` is set, allowed commands run without confirmation.

## Small Model Compatibility

The tool parser handles common output quirks from smaller local models:

- **Missing inner tags** — `<run_shell>ls -la</run_shell>` works the same as `<run_shell><command>ls -la</command></run_shell>`
- **Code-fenced tool calls** — tool calls wrapped in ` ```xml ` blocks are extracted automatically
- **Whitespace in tag names** — `< run_shell >` is normalised to `<run_shell>`
- **Helpful error feedback** — when a required field is missing, the error returned to the model includes the correct format so it can self-correct

For best results with small models, use a non-streaming setup (`LLM_STREAM=0`) and set `LLM_VERBOSE=true` to see exactly what the model is outputting if something goes wrong.

## Debugging

Set `LLM_VERBOSE=true` to log:
- Every HTTP request (URL, model, message count)
- Streaming JSON parse errors
- Tool call parsing decisions

```bash
LLM_VERBOSE=true openllm-cli -a
```

## Security

- API keys are read from environment variables only — never from command-line arguments
- API keys are never logged or stored
- Auto mode executes shell commands with the permissions of the current user — run in a sandboxed directory for untrusted tasks
- Skills mode restricts commands and paths to an explicit allowlist

## License

GPL v3 — see LICENSE.
