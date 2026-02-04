# OpenLLM CLI

A command-line interface for sending prompts to LLM APIs via stdin. Supports OpenRouter, ChatGPT (OpenAI), Ollama, and LM Studio. Perfect for piping data from other commands.

## Features

- 🚀 **Piped Input**: Read from stdin for seamless integration with other commands
- 🔒 **Environment-Based Configuration**: All settings via environment variables
- 📡 **Streaming Support**: Real-time streaming responses
- 🔍 **Verbose/Debug Mode**: Detailed logging for troubleshooting
- 🏠 **Local & Cloud Support**: Works with both local (Ollama, LM Studio) and cloud (OpenRouter, ChatGPT) providers

## Installation

### Prerequisites

- Go 1.21 or later
- For cloud providers: API keys (OpenRouter or OpenAI)
- For local providers: Install and run [Ollama](https://ollama.ai/) or [LM Studio](https://lmstudio.ai/)

### Building

```bash
# Build for current platform
make build

# Build for all platforms
make all

# Build for specific platform
make linux-amd64
make darwin-arm64
make windows-amd64
```

Binaries are placed in `bin/` directory. Install with:
```bash
make install          # System-wide (requires sudo)
make install-user     # User directory (~/.local/bin)
```

## Quick Start

### Basic Usage

The tool reads from stdin. Set your provider and API key, then pipe data:

```bash
# OpenRouter (default)
export OPENROUTER_API_KEY="your-key"
echo "Hello, world!" | ./openllm-cli

# ChatGPT
export OPENAI_API_KEY="your-key"
export LLM_PROVIDER="chatgpt"
echo "Hello, world!" | ./openllm-cli

# Ollama (local, no API key needed)
export LLM_PROVIDER="ollama"
export LLM_MODEL="llama2"
echo "Hello, world!" | ./openllm-cli

# LM Studio (local, no API key needed)
export LLM_PROVIDER="lmstudio"
echo "Hello, world!" | ./openllm-cli
```

### Real-World Examples

```bash
# Analyze process list
ps aux | ./openllm-cli

# Summarize log file
tail -n 100 app.log | ./openllm-cli

# Analyze disk usage
df -h | ./openllm-cli

# With pre-prompt for context
export LLM_PRE_PROMPT="Analyze and identify any suspicious processes:"
ps aux | ./openllm-cli

# Streaming mode (real-time output)
export LLM_STREAM="true"
cat document.txt | ./openllm-cli

# Verbose mode (debug logging)
export LLM_VERBOSE="true"
echo "test" | ./openllm-cli
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | No | `openrouter` | Provider: `openrouter`, `chatgpt`, `ollama`, `lmstudio` |
| `OPENROUTER_API_KEY` | Yes* | - | OpenRouter API key |
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key |
| `LLM_MODEL` | No | Provider-specific | Model name (see defaults below) |
| `LLM_PRE_PROMPT` | No | - | Text to prepend to stdin input |
| `LLM_STREAM` | No | `false` | Enable streaming (`1`, `true`, `yes`, `on`) |
| `LLM_VERBOSE` | No | `false` | Enable debug logging (`1`, `true`, `yes`, `on`) |
| `LLM_TIMEOUT` | No | 60/300 | Timeout in seconds (60 non-streaming, 300 streaming) |
| `OLLAMA_URL` | No | `http://localhost:11434/api/chat` | Ollama API URL |
| `LM_STUDIO_URL` | No | `http://localhost:1234/v1/chat/completions` | LM Studio API URL |

*Required only for cloud providers (OpenRouter, ChatGPT)

### Default Models

- **OpenRouter**: `openai/gpt-oss-20b:free`
- **ChatGPT**: `gpt-3.5-turbo`
- **Ollama**: `llama2`
- **LM Studio**: `local-model`

### Setting Environment Variables

**Linux/macOS:**
```bash
export OPENROUTER_API_KEY="your-key"
export LLM_PROVIDER="openrouter"
export LLM_MODEL="openai/gpt-4"
```

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="your-key"
$env:LLM_PROVIDER="openrouter"
$env:LLM_MODEL="openai/gpt-4"
```

**Windows (Command Prompt):**
```cmd
set OPENROUTER_API_KEY=your-key
set LLM_PROVIDER=openrouter
set LLM_MODEL=openai/gpt-4
```

## Provider Details

### OpenRouter
- Requires API key
- Supports all OpenRouter models
- Uses OpenAI-compatible API
- Streaming: Server-Sent Events (SSE)

### ChatGPT (OpenAI)
- Requires OpenAI API key
- Supports all OpenAI models
- Uses OpenAI native API
- Streaming: Server-Sent Events (SSE)

### Ollama
- No API key required
- Ensure Ollama is running: `ollama serve`
- Install models: `ollama pull llama2`
- Streaming: Newline-delimited JSON
- Default URL: `http://localhost:11434/api/chat`

### LM Studio
- No API key required
- Enable local server in LM Studio (Developer tab)
- Load a model before using
- Uses OpenAI-compatible API
- Streaming: Server-Sent Events (SSE)
- Default URL: `http://localhost:1234/v1/chat/completions`
- **Note**: Can be slow with large models. Increase timeout if needed:
  ```bash
  export LLM_TIMEOUT=600  # 10 minutes
  ```

## Error Handling

The CLI provides clear error messages for:
- Missing API keys (for cloud providers)
- Invalid provider names
- Empty input
- Network errors (connection refused usually means service isn't running)
- API errors
- Timeout errors (increase `LLM_TIMEOUT` for slow models)

## Security

- API keys are only read from environment variables (never command-line arguments)
- API keys are never logged or stored
- Use environment variables for API keys in production
- Local providers (Ollama, LM Studio) run entirely on your machine

## License

GPL v3 - see LICENSE file for details.

## Support

For issues and questions:
- [OpenRouter docs](https://openrouter.ai/docs)
- [OpenAI API docs](https://platform.openai.com/docs)
- [Ollama docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [LM Studio docs](https://lmstudio.ai/docs)
- Open an issue on GitHub
