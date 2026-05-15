# openllm-cli

An interactive LLM CLI with filesystem access, autonomous tool use, and skills integration. Defaults to Ollama (local, no API key). Also supports Anthropic, OpenAI, OpenRouter, and LM Studio.

## Features

- **Interactive REPL** — multi-turn conversation with history, slash commands, and inline file/command injection
- **Auto mode** — fully autonomous: the LLM reads/writes files and runs shell commands without prompting
- **Skills mode** — controlled execution driven by a `SKILL.md` file in your working directory
- **Pipe mode** — single-shot stdin → stdout for scripting and shell pipelines
- **Streaming** — real-time token output and live shell command streaming in auto mode
- **Multi-provider** — one binary, five backends

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

### Auto mode — create a Go app

```bash
openllm-cli -a
> create a CLI tool in Go that prints a Fibonacci sequence, compile it, and run it
```

The agent will write the source file(s) to your working directory, run `go build`, execute the binary, and show you the output — no prompts, no confirmation steps.

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
| `LLM_VERBOSE` | `false` | Debug logging |
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

## Auto Mode — Go Development

In auto mode (`-a`), the agent follows this workflow automatically when asked to build a Go program:

1. Check whether `go.mod` exists in the working directory
2. Run `go mod init app` if missing
3. Write the Go source file(s) to the working directory
4. Run `go build -o ./app .`
5. Run `./app` and show the output
6. Report what was built

Shell output streams to your terminal in real time during steps 4 and 5.

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

## Security

- API keys are read from environment variables only — never from command-line arguments
- API keys are never logged or stored
- Auto mode executes shell commands with the permissions of the current user — run in a sandboxed directory for untrusted tasks
- Skills mode restricts commands and paths to an explicit allowlist

## License

GPL v3 — see LICENSE.
