// openllm-cli: an interactive LLM CLI with file I/O, shell execution, and auto mode.
//
// Usage:
//   openllm-cli [flags]
//   echo "prompt" | openllm-cli          (single-shot pipe mode)
//
// Flags:
//   -i, --interactive   start interactive REPL (default when no stdin pipe)
//   -a, --auto          auto mode: LLM can read/write files and run commands autonomously
//   -s, --skills        enable skills mode: read SKILL.md/Skills.md and follow directions
//   -h, --help          show help
package main

import (
        "bufio"
        "bytes"
        "context"
        "encoding/json"
        "errors"
        "fmt"
        "io"
        "log"
        "net/http"
        "os"
        "os/exec"
        "os/signal"
        "path/filepath"
        "regexp"
        "runtime"
        "strconv"
        "strings"
        "sync"
        "time"
        "unicode"

        "golang.org/x/term"
)

// version is set at build time via -ldflags "-X main.version=..."
var version = "dev"

// ============================================================
// Constants
// ============================================================

const (
        // Environment variable names
        envAPIKey       = "OPENROUTER_API_KEY"
        envOpenAIKey    = "OPENAI_API_KEY"
        envAnthropicKey = "ANTHROPIC_API_KEY"
        envProvider     = "LLM_PROVIDER"
        envModel        = "LLM_MODEL"
        envSystemPrompt = "LLM_SYSTEM_PROMPT"
        envStream       = "LLM_STREAM"
        envVerbose      = "LLM_VERBOSE"
        envTimeout      = "LLM_TIMEOUT"
        envShellTimeout = "LLM_SHELL_TIMEOUT"
        envMaxTokens    = "LLM_MAX_TOKENS"
        envAutoApprove  = "LLM_AUTO_APPROVE" // skip confirmations in auto mode
        envOllamaURL    = "OLLAMA_URL"
        envLMStudioURL  = "LM_STUDIO_URL"
        envSkillsMode   = "LLM_SKILLS_MODE"
        envInteractive  = "LLM_INTERACTIVE"

        // Defaults
        defaultLLMTimeout    = 120 * time.Second
        defaultStreamTimeout = 300 * time.Second
        defaultShellTimeout  = 60 * time.Second
        defaultMaxTokens     = 8096
        maxFileReadSize      = 2 * 1024 * 1024 // 2 MB
        maxShellOutputChars  = 12 * 1024        // 12 KB — prevents HTML blobs from overflowing context

        // Model defaults per provider
        defaultOpenRouterModel = "openai/gpt-4o-mini"
        defaultOpenAIModel     = "gpt-4o-mini"
        defaultAnthropicModel  = "claude-sonnet-4-20250514"
        defaultOllamaModel     = "gemma3:4b"
        defaultLMStudioModel   = "local-model"

        // API endpoints
        openRouterURL      = "https://openrouter.ai/api/v1/chat/completions"
        openAIURL          = "https://api.openai.com/v1/chat/completions"
        anthropicURL       = "https://api.anthropic.com/v1/messages"
        defaultOllamaURL   = "http://localhost:11434/api/chat"
        defaultLMStudioURL = "http://localhost:1234/v1/chat/completions"
        anthropicVersion   = "2023-06-01"
        userAgent          = "openllm-cli/2.0"

        // ANSI colors
        colorReset   = "\033[0m"
        colorBold    = "\033[1m"
        colorDim     = "\033[2m"
        colorCyan    = "\033[36m"
        colorGreen   = "\033[32m"
        colorYellow  = "\033[33m"
        colorRed     = "\033[31m"
        colorBlue    = "\033[34m"
        colorMagenta = "\033[35m"

        // Tool names
        toolReadFile  = "read_file"
        toolWriteFile = "write_file"
        toolRunShell  = "run_shell"
        toolDone      = "task_done"
)

// autoSystemPrompt is designed to work across all model families and sizes —
// from mistral:7b and phi3:mini up to GPT-4o and Claude Opus.
const autoSystemPrompt = `You are an autonomous AI agent. You have access to the local filesystem and shell.
Files are created in the current working directory.

## TOOL FORMAT

To take an action, output one of these XML blocks — raw, never inside code fences:

READ A FILE
<read_file><path>./path/to/file</path></read_file>

WRITE A FILE
<write_file><path>./path/to/file</path><content>
full file contents go here
</content></write_file>

RUN A SHELL COMMAND
<run_shell><command>command to run</command></run_shell>

FINISH
<task_done><summary>Brief summary of what was done and the output.</summary></task_done>

## RULES

Rule 1: Output EXACTLY ONE tool call per response. Never two in one response. If the user asks a direct question that needs no file or shell access, answer in plain text with no tool call and no task_done.
Rule 2: WAIT for the tool result before writing your next response. Never invent results.
Rule 3: Do NOT wrap tool calls in code fences or markdown blocks.
Rule 4: After every result, reason briefly then emit the next tool call or task_done.
Rule 5: Every task ends with task_done.
Rule 6: Shell output is capped at ~12 KB. For web pages or large files, pipe through grep/head/sed to extract only what you need. Example: curl -s URL | grep -i "keyword" | head -40
Rule 7: Use the simplest tool for the job. Research → curl/grep. File tasks → read_file/write_file. System info → run_shell. Do NOT write programs to solve tasks that shell commands can handle directly.

ONE CALL. WAIT. ONE CALL. WAIT. Repeat until done.

## APPROACH

Match your approach to the task:
- Research / fetch data: use curl, grep, jq, awk directly in run_shell
- Read or summarise files: use read_file, then task_done with your analysis
- Write or edit files: use write_file
- System tasks: use run_shell with the appropriate system command
- Only write and compile code when the user explicitly asks for a program`

// skillsSystemPrompt is added when skills mode is enabled
var skillsSystemPrompt = `

SKILLS MODE ACTIVE: You have access to a skills configuration that defines allowed commands and file access patterns.
When executing commands or reading files, they will be automatically validated against the skills configuration.
You can use inline commands with backticks ` + "`command`" + ` and file references with @path/to/file in your responses.`

// ============================================================
// Provider
// ============================================================

type Provider string

const (
        ProviderOpenRouter Provider = "openrouter"
        ProviderOpenAI     Provider = "openai"
        ProviderAnthropic  Provider = "anthropic"
        ProviderOllama     Provider = "ollama"
        ProviderLMStudio   Provider = "lmstudio"
)

// ============================================================
// Config
// ============================================================

type Config struct {
        Provider     Provider
        APIKey       string
        Model        string
        OllamaURL    string
        LMStudioURL  string
        SystemPrompt string
        Stream       bool
        Verbose      bool
        LLMTimeout   time.Duration
        ShellTimeout time.Duration
        MaxTokens    int
        AutoApprove  bool // skip y/N prompts for tool use
        SkillsMode   bool // enable skills processing
}

func loadConfig() *Config {
        providerStr := envOr(envProvider, string(ProviderOllama))
        provider := Provider(strings.ToLower(strings.TrimSpace(providerStr)))

        validProviders := map[Provider]bool{
                ProviderOpenRouter: true,
                ProviderOpenAI:    true,
                ProviderAnthropic:  true,
                ProviderOllama:     true,
                ProviderLMStudio:   true,
        }
        if !validProviders[provider] {
                fatalf("Unknown provider %q. Choose from: openrouter, openai, anthropic, ollama, lmstudio", provider)
        }

        apiKey := resolveAPIKey(provider)
        model := resolveModel(provider)

        maxTokens := defaultMaxTokens
        if v, err := strconv.Atoi(os.Getenv(envMaxTokens)); err == nil && v > 0 {
                maxTokens = v
        }

        stream := isTruthy(envStream)
        llmTimeout := defaultLLMTimeout
        if stream {
                llmTimeout = defaultStreamTimeout
        }
        if v, err := strconv.Atoi(os.Getenv(envTimeout)); err == nil && v > 0 {
                llmTimeout = time.Duration(v) * time.Second
        }

        shellTimeout := defaultShellTimeout
        if v, err := strconv.Atoi(os.Getenv(envShellTimeout)); err == nil && v > 0 {
                shellTimeout = time.Duration(v) * time.Second
        }

        return &Config{
                Provider:     provider,
                APIKey:       apiKey,
                Model:        model,
                OllamaURL:    envOr(envOllamaURL, defaultOllamaURL),
                LMStudioURL:  envOr(envLMStudioURL, defaultLMStudioURL),
                SystemPrompt: os.Getenv(envSystemPrompt),
                Stream:       stream,
                Verbose:      isTruthy(envVerbose),
                LLMTimeout:   llmTimeout,
                ShellTimeout: shellTimeout,
                MaxTokens:    maxTokens,
                AutoApprove:  isTruthy(envAutoApprove),
                SkillsMode:   isTruthy(envSkillsMode),
        }
}

func resolveAPIKey(p Provider) string {
        switch p {
        case ProviderOpenAI:
                key := strings.TrimSpace(os.Getenv(envOpenAIKey))
                if key == "" {
                        fatalf("OpenAI requires %s to be set", envOpenAIKey)
                }
                return key
        case ProviderOpenRouter:
                key := strings.TrimSpace(os.Getenv(envAPIKey))
                if key == "" {
                        fatalf("OpenRouter requires %s to be set", envAPIKey)
                }
                return key
        case ProviderAnthropic:
                key := strings.TrimSpace(os.Getenv(envAnthropicKey))
                if key == "" {
                        fatalf("Anthropic requires %s to be set", envAnthropicKey)
                }
                return key
        }
        return "" // Ollama, LM Studio don't need keys
}

func resolveModel(p Provider) string {
        if m := strings.TrimSpace(os.Getenv(envModel)); m != "" {
                return m
        }
        switch p {
        case ProviderOpenAI:
                return defaultOpenAIModel
        case ProviderOpenRouter:
                return defaultOpenRouterModel
        case ProviderAnthropic:
                return defaultAnthropicModel
        case ProviderOllama:
                return defaultOllamaModel
        case ProviderLMStudio:
                return defaultLMStudioModel
        default:
                return defaultOllamaModel
        }
}

// ============================================================
// Skills System
// ============================================================

// SkillsConfig holds parsed skills configuration
type SkillsConfig struct {
        Content          string
        AllowedCommands  []string
        AllowedPaths     []string
        DisallowedPaths  []string
        Instructions     string
        AutoExecute      bool
}

// loadSkillsConfig reads and parses SKILL.md or Skills.md
func loadSkillsConfig(workingDir string) (*SkillsConfig, error) {
        // Try different possible filenames
        filenames := []string{"SKILL.md", "Skills.md", "skills.md", "SKILLS.md"}

        var skillsContent string
        var foundFile string

        for _, filename := range filenames {
                path := filepath.Join(workingDir, filename)
                if content, err := os.ReadFile(path); err == nil {
                        skillsContent = string(content)
                        foundFile = filename
                        break
                }
        }

        if skillsContent == "" {
                return nil, fmt.Errorf("no skills file found (looked for: %s)", strings.Join(filenames, ", "))
        }

        infof("Loaded skills from %s", foundFile)

        config := &SkillsConfig{
                Content: skillsContent,
        }

        // Parse special directives from the skills file
        config.parseDirectives(skillsContent)

        return config, nil
}

// parseDirectives extracts configuration from skills markdown
func (sc *SkillsConfig) parseDirectives(content string) {
        lines := strings.Split(content, "\n")

        var instructionsLines []string
        inInstructions := false

        for _, line := range lines {
                line = strings.TrimSpace(line)

                // Look for configuration blocks
                if strings.HasPrefix(line, "## Instructions") || strings.HasPrefix(line, "# Instructions") {
                        inInstructions = true
                        continue
                } else if strings.HasPrefix(line, "## ") || strings.HasPrefix(line, "# ") {
                        inInstructions = false
                }

                if inInstructions && line != "" {
                        instructionsLines = append(instructionsLines, line)
                }

                // Parse allowed commands
                if strings.HasPrefix(line, "ALLOWED_COMMANDS:") {
                        commands := strings.TrimPrefix(line, "ALLOWED_COMMANDS:")
                        commands = strings.TrimSpace(commands)
                        if commands != "" {
                                sc.AllowedCommands = strings.Split(commands, ",")
                                for i := range sc.AllowedCommands {
                                        sc.AllowedCommands[i] = strings.TrimSpace(sc.AllowedCommands[i])
                                }
                        }
                }

                // Parse allowed paths
                if strings.HasPrefix(line, "ALLOWED_PATHS:") {
                        paths := strings.TrimPrefix(line, "ALLOWED_PATHS:")
                        paths = strings.TrimSpace(paths)
                        if paths != "" {
                                sc.AllowedPaths = strings.Split(paths, ",")
                                for i := range sc.AllowedPaths {
                                        sc.AllowedPaths[i] = strings.TrimSpace(sc.AllowedPaths[i])
                                }
                        }
                }

                // Parse disallowed paths
                if strings.HasPrefix(line, "DISALLOWED_PATHS:") {
                        paths := strings.TrimPrefix(line, "DISALLOWED_PATHS:")
                        paths = strings.TrimSpace(paths)
                        if paths != "" {
                                sc.DisallowedPaths = strings.Split(paths, ",")
                                for i := range sc.DisallowedPaths {
                                        sc.DisallowedPaths[i] = strings.TrimSpace(sc.DisallowedPaths[i])
                                }
                        }
                }

                // Parse auto-execute flag
                if strings.Contains(line, "AUTO_EXECUTE:") && strings.Contains(line, "true") {
                        sc.AutoExecute = true
                }
        }

        if len(instructionsLines) > 0 {
                sc.Instructions = strings.Join(instructionsLines, "\n")
        }
}

// isCommandAllowed checks if a command is allowed based on skills configuration
func (sc *SkillsConfig) isCommandAllowed(cmd string) bool {
        if len(sc.AllowedCommands) == 0 {
                // Default safe commands if none specified
                defaultAllowed := []string{"curl", "id", "whoami", "date", "pwd", "ls", "cat", "head", "tail", "grep", "echo", "which", "uname"}
                for _, allowed := range defaultAllowed {
                        if strings.HasPrefix(cmd, allowed+" ") || cmd == allowed {
                                return true
                        }
                }
                return false
        }

        for _, allowed := range sc.AllowedCommands {
                if strings.HasPrefix(cmd, allowed+" ") || cmd == allowed {
                        return true
                }
        }
        return false
}

// isPathAllowed checks if a path is allowed for reading
func (sc *SkillsConfig) isPathAllowed(path string) bool {
        absPath, err := filepath.Abs(path)
        if err != nil {
                return false
        }

        // Check disallowed paths first
        for _, disallowed := range sc.DisallowedPaths {
                if matched, _ := filepath.Match(disallowed, absPath); matched {
                        return false
                }
                if strings.HasPrefix(absPath, disallowed) {
                        return false
                }
        }

        // If no allowed paths specified, allow current directory and subdirectories
        if len(sc.AllowedPaths) == 0 {
                cwd, _ := os.Getwd()
                return strings.HasPrefix(absPath, cwd)
        }

        // Check allowed paths
        for _, allowed := range sc.AllowedPaths {
                allowedAbs, _ := filepath.Abs(allowed)
                if strings.HasPrefix(absPath, allowedAbs) {
                        return true
                }
        }

        return false
}

// ============================================================
// Wire types — shared between providers
// ============================================================

// ChatMessage is a single turn in a conversation.
type ChatMessage struct {
        Role    string `json:"role"`
        Content string `json:"content"`
}

// LLMResponse is the parsed, provider-normalised reply from any backend.
type LLMResponse struct {
        Text       string
        ToolCalls  []ToolCall // populated in auto mode
        StopReason string
}

// ============================================================
// Tool system (auto mode)
// ============================================================

// ToolCall is an action the LLM wants to take.
type ToolCall struct {
        Name  string
        Input map[string]string
}

// parseToolCall scans the LLM's text for an XML tool call.
// Tolerant of small-model quirks: code fences wrapping the XML and
// whitespace inside tag names. Returns nil if none found.
func parseToolCall(text string) *ToolCall {
        if tc := parseToolCallRaw(text); tc != nil {
                return tc
        }
        // Small models often wrap tool calls in ```xml...``` blocks.
        if inner := extractCodeFenceContents(text); inner != "" {
                return parseToolCallRaw(inner)
        }
        return nil
}

func parseToolCallRaw(text string) *ToolCall {
        text = normalizeTagWhitespace(text)
        tools := []string{toolReadFile, toolWriteFile, toolRunShell, toolDone}

        // Pick the tool whose opening tag appears earliest in the text, so that if a
        // model emits two tool calls we always take the first one regardless of which
        // tool type it is (the iteration order of 'tools' must not determine priority).
        bestStart := -1
        var best *ToolCall

        for _, name := range tools {
                open := "<" + name + ">"
                close := "</" + name + ">"
                start := strings.Index(text, open)
                end := strings.Index(text, close)
                if start == -1 || end == -1 || end <= start {
                        continue
                }
                if bestStart != -1 && start >= bestStart {
                        continue
                }
                inner := text[start+len(open) : end]
                tc := &ToolCall{Name: name, Input: map[string]string{}}
                for _, field := range []string{"path", "content", "command", "summary"} {
                        if val := extractTag(inner, field); val != "" {
                                tc.Input[field] = val
                        }
                }
                // Fallback for small models that omit the inner tag:
                // <run_shell>ls -la</run_shell> instead of <run_shell><command>ls -la</command></run_shell>
                // <task_done>summary text</task_done> instead of <task_done><summary>...</summary></task_done>
                innerTrimmed := strings.TrimSpace(inner)
                if name == toolRunShell && tc.Input["command"] == "" && innerTrimmed != "" {
                        tc.Input["command"] = innerTrimmed
                }
                if name == toolDone && tc.Input["summary"] == "" && innerTrimmed != "" {
                        tc.Input["summary"] = innerTrimmed
                }
                bestStart = start
                best = tc
        }
        return best
}

var tagWhitespaceRe = regexp.MustCompile(`<(\s*/?\s*)(\w+)(\s*)>`)

func normalizeTagWhitespace(text string) string {
        return tagWhitespaceRe.ReplaceAllStringFunc(text, func(m string) string {
                parts := tagWhitespaceRe.FindStringSubmatch(m)
                if len(parts) < 3 {
                        return m
                }
                prefix := strings.TrimSpace(parts[1])
                return "<" + prefix + parts[2] + ">"
        })
}

var codeFenceRe = regexp.MustCompile("(?s)```(?:xml|bash|sh|shell|json)?\n?(.*?)```")

func extractCodeFenceContents(text string) string {
        matches := codeFenceRe.FindAllStringSubmatch(text, -1)
        if len(matches) == 0 {
                return ""
        }
        var parts []string
        for _, m := range matches {
                if len(m) > 1 {
                        parts = append(parts, strings.TrimSpace(m[1]))
                }
        }
        return strings.Join(parts, "\n")
}

// stripToolCallXML removes the tool-call XML block from text, leaving only prose.
func stripToolCallXML(text string) string {
        tools := []string{toolReadFile, toolWriteFile, toolRunShell, toolDone}
        for _, name := range tools {
                open := "<" + name + ">"
                close := "</" + name + ">"
                start := strings.Index(text, open)
                end := strings.Index(text, close)
                if start == -1 || end == -1 {
                        continue
                }
                return strings.TrimSpace(text[:start] + text[end+len(close):])
        }
        return text
}

func extractTag(s, tag string) string {
        open := "<" + tag + ">"
        close := "</" + tag + ">"
        start := strings.Index(s, open)
        end := strings.Index(s, close)
        if start == -1 || end == -1 || end <= start {
                return ""
        }
        return strings.TrimSpace(s[start+len(open) : end])
}

// ============================================================
// Session — holds conversation state
// ============================================================

type Session struct {
        cfg          *Config
        history      []ChatMessage
        cwd          string // tracked working directory
        autoMode     bool
        skillsMode   bool
        skillsConfig *SkillsConfig
        cancelMu     sync.Mutex
        cancelReq    context.CancelFunc // non-nil while an LLM request is in flight
}

func newSession(cfg *Config, auto bool, skills bool) *Session {
        cwd, _ := os.Getwd()
        s := &Session{cfg: cfg, cwd: cwd, autoMode: auto, skillsMode: skills}

        // Load skills configuration if skills mode is enabled
        if skills {
                if skillsConfig, err := loadSkillsConfig(cwd); err != nil {
                        infof("Skills mode disabled: %v", err)
                        s.skillsMode = false
                } else {
                        s.skillsConfig = skillsConfig
                        infof("Skills mode enabled with %d allowed commands", len(skillsConfig.AllowedCommands))
                }
        }

        return s
}

func (s *Session) addUser(content string)     { s.history = append(s.history, ChatMessage{"user", content}) }
func (s *Session) addAssistant(content string) { s.history = append(s.history, ChatMessage{"assistant", content}) }
func (s *Session) clearHistory()              { s.history = nil }

// send calls the LLM with the current history and returns the response.
// Ctrl+C (SIGINT) cancels the in-flight request via signal.NotifyContext.
func (s *Session) send() (*LLMResponse, error) {
        ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
        defer cancel()
        s.cancelMu.Lock()
        s.cancelReq = cancel
        s.cancelMu.Unlock()
        defer func() {
                s.cancelMu.Lock()
                s.cancelReq = nil
                s.cancelMu.Unlock()
        }()
        return callLLM(ctx, s.cfg, s.history, s.autoMode, s.skillsMode, s.skillsConfig)
}

// ============================================================
// Tool execution
// ============================================================

// executeTool runs a tool call and returns a result string to feed back to the LLM.
func (s *Session) executeTool(tc *ToolCall) (string, error) {
        switch tc.Name {

        case toolReadFile:
                path := tc.Input["path"]
                if path == "" {
                        return "", fmt.Errorf("read_file: missing <path>")
                }

                // Skills mode validation
                if s.skillsMode && s.skillsConfig != nil {
                        absPath := filepath.Join(s.cwd, path)
                        if !s.skillsConfig.isPathAllowed(absPath) {
                                return fmt.Sprintf("[read_file error: path %s not allowed by skills configuration]", path), nil
                        }
                }

                content, err := readPath(filepath.Join(s.cwd, path))
                if err != nil {
                        return fmt.Sprintf("[read_file error: %v]", err), nil
                }
                return fmt.Sprintf("[read_file: %s]\n```\n%s\n```", path, content), nil

        case toolWriteFile:
                path := tc.Input["path"]
                content := tc.Input["content"]
                if path == "" {
                        return "[write_file error: missing <path> tag. Correct format: <write_file><path>./filename.ext</path><content>\nfile contents\n</content></write_file>]", nil
                }
                if strings.TrimSpace(content) == "" {
                        return fmt.Sprintf("[write_file error: missing <content> tag or empty content for %s. Include the full file content between <content>...</content> tags]", path), nil
                }

                // Skills mode validation
                if s.skillsMode && s.skillsConfig != nil {
                        absPath := filepath.Join(s.cwd, path)
                        if !s.skillsConfig.isPathAllowed(absPath) {
                                return fmt.Sprintf("[write_file error: path %s not allowed by skills configuration]", path), nil
                        }
                }

                abs := filepath.Join(s.cwd, path)
                if err := writeFileAtomic(abs, content); err != nil {
                        return fmt.Sprintf("[write_file error: %v]", err), nil
                }
                return fmt.Sprintf("[write_file: wrote %d bytes to %s]", len(content), path), nil

        case toolRunShell:
                cmd := tc.Input["command"]
                if cmd == "" {
                        return "[run_shell error: missing <command> tag. Correct format: <run_shell><command>your command here</command></run_shell>]", nil
                }

                // Skills mode validation
                if s.skillsMode && s.skillsConfig != nil {
                        if !s.skillsConfig.isCommandAllowed(cmd) {
                                return fmt.Sprintf("[run_shell error: command '%s' not allowed by skills configuration]", cmd), nil
                        }
                }

                var out string
                var err error
                if s.autoMode {
                        out, err = runShellLive(cmd, s.cwd, s.cfg.ShellTimeout)
                } else {
                        out, err = runShell(cmd, s.cwd, s.cfg.ShellTimeout)
                }
                result := out
                if err != nil {
                        result += "\n[exit error: " + err.Error() + "]"
                }
                if result == "" {
                        result = "(no output)"
                }
                if len(result) > maxShellOutputChars {
                        result = result[:maxShellOutputChars] + fmt.Sprintf("\n[...output truncated: %d chars total. Use grep/head/tail/jq to filter first.]", len(result))
                }
                return fmt.Sprintf("[run_shell: `%s`]\n```\n%s\n```", cmd, result), nil

        case toolDone:
                summary := tc.Input["summary"]
                return fmt.Sprintf("[task_done: %s]", summary), nil
        }

        return "", fmt.Errorf("unknown tool: %s", tc.Name)
}

// confirm asks the user to approve a tool call.
// Auto mode and auto-approve both bypass confirmation entirely.
func (s *Session) confirm(tc *ToolCall) bool {
        if s.autoMode || s.cfg.AutoApprove {
                return true
        }

        // In skills mode with auto_execute, auto-approve safe commands
        if s.skillsMode && s.skillsConfig != nil && s.skillsConfig.AutoExecute {
                if tc.Name == toolRunShell {
                        cmd := tc.Input["command"]
                        if s.skillsConfig.isCommandAllowed(cmd) {
                                return true
                        }
                }
                if tc.Name == toolReadFile {
                        path := tc.Input["path"]
                        absPath := filepath.Join(s.cwd, path)
                        if s.skillsConfig.isPathAllowed(absPath) {
                                return true
                        }
                }
        }

        var desc string
        switch tc.Name {
        case toolReadFile:
                desc = fmt.Sprintf("read %s", tc.Input["path"])
        case toolWriteFile:
                desc = fmt.Sprintf("write %s (%d bytes)", tc.Input["path"], len(tc.Input["content"]))
        case toolRunShell:
                desc = fmt.Sprintf("run: %s", tc.Input["command"])
        case toolDone:
                return true
        default:
                desc = tc.Name
        }
        fmt.Printf("\n%s%s  Allow: %s?%s  %s[y/N]%s ", colorBold, colorYellow, desc, colorReset, colorDim, colorReset)
        var ans string
        fmt.Scanln(&ans)
        return strings.ToLower(strings.TrimSpace(ans)) == "y"
}

// ============================================================
// Auto mode loop
// ============================================================

// runAutoTask runs a single task through the tool-use loop.
// The user's message is already in s.history when this is called.
func (s *Session) runAutoTask() error {
        const maxRounds = 2024
        const maxConsecutiveErrors = 3
        lastResult := ""
        consecutiveErrors := 0

        for round := 0; round < maxRounds; round++ {
                resp, err := s.send()
                if err != nil {
                        return fmt.Errorf("LLM error: %w", err)
                }

                // Parse tool call first so we know what to display
                tc := parseToolCall(resp.Text)

                // task_done — print summary only, skip showing raw XML response
                if tc != nil && tc.Name == toolDone {
                        s.addAssistant(resp.Text)
                        fmt.Printf("\n%s✓ Done%s  %s\n\n", colorGreen+colorBold, colorReset, tc.Input["summary"])
                        return nil
                }

                // Process skills inline commands and file references before printing
                processedText := resp.Text
                if s.skillsMode {
                        processedText = s.processSkillsInline(resp.Text)
                }

                // Strip the tool call XML from display text to avoid raw XML in output
                displayText := processedText
                if tc != nil {
                        displayText = stripToolCallXML(processedText)
                }
                if strings.TrimSpace(displayText) != "" {
                        printAssistant(s.cfg, displayText)
                }
                s.addAssistant(processedText)

                if tc == nil {
                        debugf(s.cfg, "no tool call detected in response (round %d)", round+1)
                        return nil
                }

                // Show the tool call to the user and confirm
                printToolCall(tc)
                if !s.confirm(tc) {
                        s.addUser("[User declined this action. Stop and ask what to do instead.]")
                        continue
                }

                // Execute and feed the result back
                result, err := s.executeTool(tc)
                if err != nil {
                        result = fmt.Sprintf("[tool error: %v]", err)
                }

                // Break if the model is stuck repeating the same error
                if result == lastResult {
                        consecutiveErrors++
                        if consecutiveErrors >= maxConsecutiveErrors {
                                return fmt.Errorf("model stuck: same result %d times in a row: %s", consecutiveErrors, result)
                        }
                } else {
                        consecutiveErrors = 0
                        lastResult = result
                }

                // In auto mode, shell output was already streamed live — skip duplicate print.
                if !s.autoMode || tc.Name != toolRunShell {
                        printToolResult(result)
                }
                s.addUser(result)
        }

        return fmt.Errorf("reached maximum tool rounds (%d) without completing", maxRounds)
}

// processSkillsInline processes inline commands and file references in skills mode
func (s *Session) processSkillsInline(text string) string {
        if s.skillsConfig == nil {
                return text
        }

        // Process backtick commands
        text = s.processInlineCommands(text)

        // Process @file references
        text = s.processInlineFileRefs(text)

        return text
}

// processInlineCommands executes commands in backticks if allowed
func (s *Session) processInlineCommands(text string) string {
        // Regex to find `command` patterns
        re := regexp.MustCompile("`([^`]+)`")

        return re.ReplaceAllStringFunc(text, func(match string) string {
                cmd := match[1 : len(match)-1] // Remove backticks

                if !s.skillsConfig.isCommandAllowed(cmd) {
                        return fmt.Sprintf("[command '%s' not allowed]", cmd)
                }

                out, err := runShell(cmd, s.cwd, s.cfg.ShellTimeout)
                if err != nil {
                        out += " [error: " + err.Error() + "]"
                }

                fmt.Printf("%s[executed: `%s`]%s\n", colorDim, cmd, colorReset)

                if out == "" {
                        return "(no output)"
                }
                return out
        })
}

// processInlineFileRefs reads files referenced with @path if allowed
func (s *Session) processInlineFileRefs(text string) string {
        // Regex to find @path patterns
        re := regexp.MustCompile(`@([^\s]+)`)

        return re.ReplaceAllStringFunc(text, func(match string) string {
                path := match[1:] // Remove @
                absPath := filepath.Join(s.cwd, path)

                if !s.skillsConfig.isPathAllowed(absPath) {
                        return fmt.Sprintf("[file '%s' not allowed]", path)
                }

                content, err := readPath(absPath)
                if err != nil {
                        return fmt.Sprintf("[file error: %v]", err)
                }

                fmt.Printf("%s[read: @%s]%s\n", colorDim, path, colorReset)

                return fmt.Sprintf("[File: %s]\n```\n%s\n```", path, content)
        })
}

// ============================================================
// Terminal line editor — cross-platform, no build tags
// ============================================================

var errInterrupt = fmt.Errorf("interrupt")

type lineReader struct {
        history  []string
        histFile string
        maxHist  int
}

func newLineReader(histFile string) *lineReader {
        lr := &lineReader{histFile: histFile, maxHist: 500}
        lr.loadHistory()
        return lr
}

func (lr *lineReader) close() { lr.saveHistory() }

func (lr *lineReader) addToHistory(line string) {
        if line == "" {
                return
        }
        if len(lr.history) > 0 && lr.history[len(lr.history)-1] == line {
                return
        }
        lr.history = append(lr.history, line)
        if len(lr.history) > lr.maxHist {
                lr.history = lr.history[len(lr.history)-lr.maxHist:]
        }
}

func (lr *lineReader) loadHistory() {
        if lr.histFile == "" {
                return
        }
        data, err := os.ReadFile(lr.histFile)
        if err != nil {
                return
        }
        for _, line := range strings.Split(string(data), "\n") {
                if line = strings.TrimRight(line, "\r"); line != "" {
                        lr.history = append(lr.history, line)
                }
        }
        if len(lr.history) > lr.maxHist {
                lr.history = lr.history[len(lr.history)-lr.maxHist:]
        }
}

func (lr *lineReader) saveHistory() {
        if lr.histFile == "" || len(lr.history) == 0 {
                return
        }
        _ = os.WriteFile(lr.histFile, []byte(strings.Join(lr.history, "\n")+"\n"), 0600)
}

// readLine displays prompt and reads a line with full line editing and history.
// Returns (line, errInterrupt) on Ctrl+C, ("", io.EOF) on Ctrl+D with empty buffer.
func (lr *lineReader) readLine(prompt string) (string, error) {
        fd := int(os.Stdin.Fd())
        state, err := term.MakeRaw(fd)
        if err != nil {
                // not a terminal or raw mode unsupported — plain buffered read
                fmt.Print(prompt)
                r := bufio.NewReader(os.Stdin)
                line, err2 := r.ReadString('\n')
                return strings.TrimRight(line, "\r\n"), err2
        }
        defer term.Restore(fd, state) //nolint:errcheck

        fmt.Print(prompt)

        var (
                buf        []rune
                cursor     int
                histIdx    = len(lr.history)
                savedInput string
        )

        redraw := func() {
                fmt.Printf("\r%s%s\033[K", prompt, string(buf))
                if tail := len(buf) - cursor; tail > 0 {
                        fmt.Printf("\033[%dD", tail)
                }
        }

        b1 := make([]byte, 1)
        readByte := func() (byte, error) {
                _, err := os.Stdin.Read(b1)
                return b1[0], err
        }

        for {
                b, err := readByte()
                if err != nil {
                        fmt.Print("\r\n")
                        return string(buf), err
                }

                switch b {
                case '\r', '\n':
                        fmt.Print("\r\n")
                        return string(buf), nil

                case 3: // Ctrl+C
                        fmt.Print("^C\r\n")
                        return string(buf), errInterrupt

                case 4: // Ctrl+D — forward delete or EOF on empty buffer
                        if len(buf) == 0 {
                                fmt.Print("\r\n")
                                return "", io.EOF
                        }
                        if cursor < len(buf) {
                                buf = append(buf[:cursor], buf[cursor+1:]...)
                                redraw()
                        }

                case 127, 8: // Backspace / DEL
                        if cursor > 0 {
                                cursor--
                                buf = append(buf[:cursor], buf[cursor+1:]...)
                                redraw()
                        }

                case 1: // Ctrl+A — beginning of line
                        cursor = 0
                        redraw()

                case 5: // Ctrl+E — end of line
                        cursor = len(buf)
                        redraw()

                case 11: // Ctrl+K — kill to end
                        buf = buf[:cursor]
                        redraw()

                case 21: // Ctrl+U — kill to beginning
                        buf = buf[cursor:]
                        cursor = 0
                        redraw()

                case 23: // Ctrl+W — kill word backward
                        end := cursor
                        for cursor > 0 && buf[cursor-1] == ' ' {
                                cursor--
                        }
                        for cursor > 0 && buf[cursor-1] != ' ' {
                                cursor--
                        }
                        buf = append(buf[:cursor], buf[end:]...)
                        redraw()

                case 27: // ESC — arrow keys and other sequences
                        b2, err := readByte()
                        if err != nil || b2 != '[' {
                                continue
                        }
                        b3, err := readByte()
                        if err != nil {
                                continue
                        }
                        switch b3 {
                        case 'A': // Up — history previous
                                if histIdx > 0 {
                                        if histIdx == len(lr.history) {
                                                savedInput = string(buf)
                                        }
                                        histIdx--
                                        buf = []rune(lr.history[histIdx])
                                        cursor = len(buf)
                                        redraw()
                                }
                        case 'B': // Down — history next
                                if histIdx < len(lr.history) {
                                        histIdx++
                                        if histIdx == len(lr.history) {
                                                buf = []rune(savedInput)
                                        } else {
                                                buf = []rune(lr.history[histIdx])
                                        }
                                        cursor = len(buf)
                                        redraw()
                                }
                        case 'C': // Right
                                if cursor < len(buf) {
                                        cursor++
                                        fmt.Print("\033[C")
                                }
                        case 'D': // Left
                                if cursor > 0 {
                                        cursor--
                                        fmt.Print("\033[D")
                                }
                        case 'H': // Home
                                cursor = 0
                                redraw()
                        case 'F': // End
                                cursor = len(buf)
                                redraw()
                        case '1', '3', '4': // Extended: ESC [ n ~
                                b4, err := readByte()
                                if err != nil || b4 != '~' {
                                        continue
                                }
                                switch b3 {
                                case '1': // Home (xterm variant)
                                        cursor = 0
                                        redraw()
                                case '3': // Delete
                                        if cursor < len(buf) {
                                                buf = append(buf[:cursor], buf[cursor+1:]...)
                                                redraw()
                                        }
                                case '4': // End (xterm variant)
                                        cursor = len(buf)
                                        redraw()
                                }
                        }

                default:
                        if b < 32 {
                                continue // ignore other control characters
                        }
                        // Decode UTF-8 multi-byte sequences
                        var r rune
                        if b < 0x80 {
                                r = rune(b)
                        } else {
                                var needed int
                                switch {
                                case b&0xE0 == 0xC0:
                                        needed = 1
                                case b&0xF0 == 0xE0:
                                        needed = 2
                                case b&0xF8 == 0xF0:
                                        needed = 3
                                default:
                                        continue // invalid UTF-8 lead byte
                                }
                                rest := make([]byte, needed)
                                if _, err := io.ReadFull(os.Stdin, rest); err != nil {
                                        continue
                                }
                                rr := []rune(string(append([]byte{b}, rest...)))
                                if len(rr) == 0 {
                                        continue
                                }
                                r = rr[0]
                        }
                        buf = append(buf, 0)
                        copy(buf[cursor+1:], buf[cursor:])
                        buf[cursor] = r
                        cursor++
                        if cursor == len(buf) {
                                fmt.Printf("%c", r)
                        } else {
                                redraw()
                        }
                }
        }
}

// ============================================================
// Interactive REPL
// ============================================================

func promptStr(s *Session) string {
        mode := ""
        if s.autoMode && s.skillsMode {
                mode = colorMagenta + " [auto+skills]" + colorReset
        } else if s.autoMode {
                mode = colorMagenta + " [auto]" + colorReset
        } else if s.skillsMode {
                mode = colorMagenta + " [skills]" + colorReset
        }
        return fmt.Sprintf("%s%sopenllm-cli%s%s %s›%s ", colorBold, colorCyan, colorReset, mode, colorDim, colorReset)
}

func runInteractive(cfg *Config, autoMode, skillsMode bool) {
        s := newSession(cfg, autoMode, skillsMode)
        printBanner(cfg, autoMode, skillsMode)

        histFile := ""
        if home, err := os.UserHomeDir(); err == nil {
                histFile = home + "/.openllm-cli_history"
        }
        lr := newLineReader(histFile)
        defer lr.close()

        for {
                line, err := lr.readLine(promptStr(s))
                if errors.Is(err, errInterrupt) {
                        if strings.TrimSpace(line) == "" {
                                fmt.Printf("%sBye!%s\n", colorDim, colorReset)
                                return
                        }
                        // Ctrl+C mid-line — clear and continue
                        continue
                }
                if errors.Is(err, io.EOF) {
                        fmt.Printf("%sBye!%s\n", colorDim, colorReset)
                        return
                }
                if err != nil {
                        printError(err.Error())
                        return
                }

                line = strings.TrimSpace(line)
                if line == "" {
                        continue
                }

                if strings.HasPrefix(line, "/") {
                        if quit := s.handleSlashCommand(line); quit {
                                break
                        }
                        continue
                }

                lr.addToHistory(line)

                expanded, err := expandInline(line, cfg)
                if err != nil {
                        printError(err.Error())
                        continue
                }

                s.addUser(expanded)

                if s.autoMode {
                        if err := s.runAutoTask(); err != nil {
                                if errors.Is(err, context.Canceled) {
                                        fmt.Printf("\n%s[interrupted]%s\n", colorYellow, colorReset)
                                        s.history = s.history[:len(s.history)-1]
                                } else {
                                        printError(err.Error())
                                }
                        }
                } else {
                        resp, err := s.send()
                        if err != nil {
                                if errors.Is(err, context.Canceled) {
                                        fmt.Printf("\n%s[interrupted]%s\n", colorYellow, colorReset)
                                } else {
                                        printError(err.Error())
                                }
                                s.history = s.history[:len(s.history)-1]
                                continue
                        }

                        responseText := resp.Text
                        if s.skillsMode {
                                responseText = s.processSkillsInline(resp.Text)
                        }
                        s.printChatResponse(responseText)
                        s.addAssistant(responseText)
                }
                fmt.Println()
        }
}

// ============================================================
// Slash commands
// ============================================================

func (s *Session) handleSlashCommand(line string) (quit bool) {
        parts := strings.SplitN(line, " ", 2)
        cmd := strings.ToLower(parts[0])
        arg := ""
        if len(parts) > 1 {
                arg = strings.TrimSpace(parts[1])
        }

        switch cmd {

        // ---- session ----
        case "/exit", "/quit", "/q":
                fmt.Printf("%sBye!%s\n", colorDim, colorReset)
                return true

        case "/clear", "/reset":
                s.clearHistory()
                infof("Conversation cleared.")

        case "/history":
                s.printHistory()

        // ---- config ----
        case "/model":
                if arg == "" {
                        infof("Model: %s", s.cfg.Model)
                } else {
                        s.cfg.Model = arg
                        infof("Model set to %s", s.cfg.Model)
                }

        case "/system":
                if arg == "" {
                        if s.cfg.SystemPrompt == "" {
                                infof("No system prompt set.")
                        } else {
                                infof("System prompt: %s", s.cfg.SystemPrompt)
                        }
                } else {
                        s.cfg.SystemPrompt = arg
                        infof("System prompt updated.")
                }

        case "/stream":
                s.cfg.Stream = !s.cfg.Stream
                infof("Streaming: %v", s.cfg.Stream)

        case "/auto":
                s.autoMode = !s.autoMode
                infof("Auto mode: %v", s.autoMode)

        case "/skills":
                s.skillsMode = !s.skillsMode
                if s.skillsMode && s.skillsConfig == nil {
                        // Try to load skills config
                        if skillsConfig, err := loadSkillsConfig(s.cwd); err != nil {
                                infof("Cannot enable skills mode: %v", err)
                                s.skillsMode = false
                        } else {
                                s.skillsConfig = skillsConfig
                                infof("Skills mode enabled")
                        }
                } else {
                        infof("Skills mode: %v", s.skillsMode)
                }

        case "/approve":
                s.cfg.AutoApprove = !s.cfg.AutoApprove
                infof("Auto-approve: %v", s.cfg.AutoApprove)

        case "/maxtokens":
                if arg == "" {
                        infof("max_tokens: %d", s.cfg.MaxTokens)
                } else if v, err := strconv.Atoi(arg); err == nil && v > 0 {
                        s.cfg.MaxTokens = v
                        infof("max_tokens set to %d", v)
                } else {
                        printError("Invalid value: " + arg)
                }

        // ---- filesystem ----
        case "/read":
                if arg == "" {
                        printError("Usage: /read <path>")
                        break
                }
                s.cmdRead(arg)

        case "/write":
                if arg == "" {
                        printError("Usage: /write <path>")
                        break
                }
                s.cmdWrite(arg)

        case "/ls":
                dir := s.cwd
                if arg != "" {
                        dir = filepath.Join(s.cwd, arg)
                }
                listing, err := dirListing(dir)
                if err != nil {
                        printError(err.Error())
                } else {
                        fmt.Printf("%s%s%s\n", colorDim, listing, colorReset)
                }

        case "/pwd":
                fmt.Printf("%s%s%s\n", colorCyan, s.cwd, colorReset)

        case "/cd":
                if arg == "" {
                        printError("Usage: /cd <path>")
                        break
                }
                target := filepath.Join(s.cwd, arg)
                if info, err := os.Stat(target); err != nil || !info.IsDir() {
                        printError(fmt.Sprintf("Not a directory: %s", target))
                } else {
                        s.cwd = filepath.Clean(target)
                        infof("cwd: %s", s.cwd)
                }

        // ---- shell ----
        case "/run", "/shell", "/exec":
                if arg == "" {
                        printError("Usage: /run <command>")
                        break
                }
                s.cmdRun(arg)

        case "/help":
                printCommandHelp()

        default:
                printError(fmt.Sprintf("Unknown command: %s  (type /help)", cmd))
        }

        return false
}

// cmdRead reads a file/dir and optionally sends it to the LLM.
func (s *Session) cmdRead(path string) {
        abs := filepath.Join(s.cwd, path)

        // Skills mode validation
        if s.skillsMode && s.skillsConfig != nil {
                if !s.skillsConfig.isPathAllowed(abs) {
                        printError(fmt.Sprintf("Path %s not allowed by skills configuration", path))
                        return
                }
        }

        content, err := readPath(abs)
        if err != nil {
                printError(err.Error())
                return
        }
        fmt.Printf("%s── %s ──%s\n%s\n%s────%s\n", colorDim, path, colorReset, content, colorDim, colorReset)

        if askYN("Send to LLM?") {
                prompt := askLine("Add a message (or Enter to just send the file):")
                msg := fmt.Sprintf("[File: %s]\n```\n%s\n```", path, content)
                if prompt != "" {
                        msg = prompt + "\n\n" + msg
                }
                s.addUser(msg)
                s.dispatchLLM()
        }
}

// cmdWrite writes content to a file. Uses last assistant reply if no content typed.
func (s *Session) cmdWrite(path string) {
        abs := filepath.Join(s.cwd, path)

        // Skills mode validation
        if s.skillsMode && s.skillsConfig != nil {
                if !s.skillsConfig.isPathAllowed(abs) {
                        printError(fmt.Sprintf("Path %s not allowed by skills configuration", path))
                        return
                }
        }

        // Default to last assistant message
        content := s.lastAssistantReply()
        if content == "" {
                fmt.Printf("%sEnter content (end with a single '.' on its own line):%s\n", colorDim, colorReset)
                var lines []string
                for {
                        l := stdinReadLine()
                        if l == "." {
                                break
                        }
                        lines = append(lines, l)
                }
                content = strings.Join(lines, "\n")
        }

        if !s.cfg.AutoApprove {
                if !askYN(fmt.Sprintf("Write %d bytes to %s?", len(content), path)) {
                        infof("Cancelled.")
                        return
                }
        }
        if err := writeFileAtomic(abs, content); err != nil {
                printError(err.Error())
        } else {
                infof("Written: %s (%d bytes)", path, len(content))
        }
}

// cmdRun executes a shell command and optionally sends output to the LLM.
func (s *Session) cmdRun(cmd string) {
        // Skills mode validation
        if s.skillsMode && s.skillsConfig != nil {
                if !s.skillsConfig.isCommandAllowed(cmd) {
                        printError(fmt.Sprintf("Command '%s' not allowed by skills configuration", cmd))
                        return
                }
        }

        if !s.cfg.AutoApprove {
                fmt.Printf("%s%s⚠  Run: %s%s\n", colorBold, colorYellow, colorReset, cmd)
                if !askYN("Confirm?") {
                        infof("Cancelled.")
                        return
                }
        }
        out, err := runShell(cmd, s.cwd, s.cfg.ShellTimeout)
        if err != nil {
                fmt.Printf("%s[exit error: %v]%s\n", colorRed, err, colorReset)
        }
        if out == "" {
                out = "(no output)"
        }
        fmt.Printf("%s── output ──%s\n%s\n%s────%s\n", colorDim, colorReset, out, colorDim, colorReset)

        if askYN("Send output to LLM?") {
                question := askLine("What should I do with this? (or Enter to just send):")
                msg := fmt.Sprintf("[Command: `%s`]\n```\n%s\n```", cmd, out)
                if question != "" {
                        msg = question + "\n\n" + msg
                }
                s.addUser(msg)
                s.dispatchLLM()
        }
}

// dispatchLLM sends the current history to the LLM (standard or auto mode) and handles the reply.
func (s *Session) dispatchLLM() {
        if s.autoMode {
                if err := s.runAutoTask(); err != nil {
                        printError(err.Error())
                }
        } else {
                resp, err := s.send()
                if err != nil {
                        printError(err.Error())
                        s.history = s.history[:len(s.history)-1]
                        return
                }

                responseText := resp.Text
                if s.skillsMode {
                        responseText = s.processSkillsInline(resp.Text)
                }
                s.printChatResponse(responseText)
                s.addAssistant(responseText)
        }
        fmt.Println()
}

// printChatResponse handles displaying an LLM response in non-auto mode.
// It strips any tool-call XML the model may have emitted and shows a hint
// pointing the user toward auto mode when the entire response was a tool call.
func (s *Session) printChatResponse(text string) {
        displayText := stripToolCallXML(text)
        if strings.TrimSpace(displayText) == "" {
                if parseToolCall(text) != nil {
                        fmt.Printf("%s[Model wants to use tools — type /auto or restart with -a to enable auto mode]%s\n", colorYellow, colorReset)
                }
                return
        }
        printAssistant(s.cfg, displayText)
}

func (s *Session) lastAssistantReply() string {
        for i := len(s.history) - 1; i >= 0; i-- {
                if s.history[i].Role == "assistant" {
                        return s.history[i].Content
                }
        }
        return ""
}

func (s *Session) printHistory() {
        if len(s.history) == 0 {
                infof("No history yet.")
                return
        }
        fmt.Println()
        for i, m := range s.history {
                color := colorCyan
                if m.Role == "assistant" {
                        color = colorGreen
                }
                preview := m.Content
                if len(preview) > 100 {
                        preview = preview[:100] + "…"
                }
                fmt.Printf("  %s%d [%s]%s %s\n", color, i+1, m.Role, colorReset, preview)
        }
        fmt.Println()
}

// ============================================================
// Inline expansion: @file and `backtick` in free-form messages
// ============================================================

// expandInline resolves @path and `cmd` tokens in a message before sending.
// This lets the user write natural prompts like "review @main.go" or
// "`ps aux` — what are all these processes?"
func expandInline(input string, cfg *Config) (string, error) {
        // Pass 1: backtick commands
        var err error
        input, err = expandBackticks(input, cfg.ShellTimeout)
        if err != nil {
                return "", err
        }
        // Pass 2: @file references
        input, err = expandAtFiles(input)
        return input, err
}

func expandBackticks(input string, timeout time.Duration) (string, error) {
        var sb strings.Builder
        i := 0
        for i < len(input) {
                if input[i] != '`' {
                        sb.WriteByte(input[i])
                        i++
                        continue
                }
                j := i + 1
                for j < len(input) && input[j] != '`' {
                        j++
                }
                if j >= len(input) { // unmatched backtick — leave as-is
                        sb.WriteByte('`')
                        i++
                        continue
                }
                cmd := input[i+1 : j]
                out, _ := runShell(cmd, "", timeout)
                fmt.Printf("%s[ran: `%s`]%s\n", colorDim, cmd, colorReset)
                sb.WriteString(fmt.Sprintf("[Command: `%s`]\n```\n%s\n```", cmd, out))
                i = j + 1
        }
        return sb.String(), nil
}

func expandAtFiles(input string) (string, error) {
        var sb strings.Builder
        i := 0
        for i < len(input) {
                if input[i] != '@' {
                        sb.WriteByte(input[i])
                        i++
                        continue
                }
                j := i + 1
                for j < len(input) && !unicode.IsSpace(rune(input[j])) {
                        j++
                }
                token := input[i+1 : j]
                if token == "" {
                        sb.WriteByte('@')
                        i++
                        continue
                }
                content, err := readPath(token)
                if err != nil {
                        return "", fmt.Errorf("@%s: %w", token, err)
                }
                fmt.Printf("%s[injected: @%s]%s\n", colorDim, token, colorReset)
                sb.WriteString(fmt.Sprintf("[File: %s]\n```\n%s\n```", token, content))
                i = j
        }
        return sb.String(), nil
}

// ============================================================
// Filesystem helpers
// ============================================================

func readPath(path string) (string, error) {
        path = filepath.Clean(path)
        info, err := os.Stat(path)
        if err != nil {
                return "", err
        }
        if info.IsDir() {
                return dirListing(path)
        }
        if info.Size() > maxFileReadSize {
                return "", fmt.Errorf("file too large (%d bytes; max %d)", info.Size(), maxFileReadSize)
        }
        data, err := os.ReadFile(path)
        return string(data), err
}

func dirListing(dir string) (string, error) {
        entries, err := os.ReadDir(dir)
        if err != nil {
                return "", err
        }
        var sb strings.Builder
        sb.WriteString(dir + "/\n")
        for _, e := range entries {
                name := e.Name()
                if e.IsDir() {
                        sb.WriteString(fmt.Sprintf("  %-30s  <dir>\n", name+"/"))
                        subs, _ := os.ReadDir(filepath.Join(dir, name))
                        for _, se := range subs {
                                sub := se.Name()
                                if se.IsDir() {
                                        sub += "/"
                                }
                                sb.WriteString(fmt.Sprintf("    %s\n", sub))
                        }
                } else {
                        info, _ := e.Info()
                        size := int64(0)
                        if info != nil {
                                size = info.Size()
                        }
                        sb.WriteString(fmt.Sprintf("  %-30s  %d bytes\n", name, size))
                }
        }
        return sb.String(), nil
}

func writeFileAtomic(path, content string) error {
        if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
                return err
        }
        return os.WriteFile(path, []byte(content), 0644)
}

// ============================================================
// Shell execution
// ============================================================

func runShell(cmd, cwd string, timeout time.Duration) (string, error) {
        ctx, cancel := context.WithTimeout(context.Background(), timeout)
        defer cancel()

        var c *exec.Cmd
        if runtime.GOOS == "windows" {
                c = exec.CommandContext(ctx, "cmd", "/C", cmd) //nolint:gosec
        } else {
                c = exec.CommandContext(ctx, "sh", "-c", cmd) //nolint:gosec
        }
        if cwd != "" {
                c.Dir = cwd
        }

        var buf bytes.Buffer
        c.Stdout = &buf
        c.Stderr = &buf
        err := c.Run()

        if ctx.Err() == context.DeadlineExceeded {
                return buf.String(), fmt.Errorf("timed out after %v", timeout)
        }
        // Non-zero exit is not a fatal error — return output + the error for context
        return strings.TrimRight(buf.String(), "\n"), err
}

// runShellLive streams command output to the terminal in real time (auto mode).
func runShellLive(cmd, cwd string, timeout time.Duration) (string, error) {
        ctx, cancel := context.WithTimeout(context.Background(), timeout)
        defer cancel()

        var c *exec.Cmd
        if runtime.GOOS == "windows" {
                c = exec.CommandContext(ctx, "cmd", "/C", cmd) //nolint:gosec
        } else {
                c = exec.CommandContext(ctx, "sh", "-c", cmd) //nolint:gosec
        }
        if cwd != "" {
                c.Dir = cwd
        }

        var buf bytes.Buffer
        c.Stdout = io.MultiWriter(os.Stdout, &buf)
        c.Stderr = io.MultiWriter(os.Stderr, &buf)

        fmt.Printf("%s", colorDim)
        err := c.Run()
        fmt.Printf("%s", colorReset)

        if ctx.Err() == context.DeadlineExceeded {
                return buf.String(), fmt.Errorf("timed out after %v", timeout)
        }
        return strings.TrimRight(buf.String(), "\n"), err
}

// ============================================================
// LLM client — unified interface across all providers
// ============================================================

func callLLM(ctx context.Context, cfg *Config, history []ChatMessage, autoMode, skillsMode bool, skillsConfig *SkillsConfig) (*LLMResponse, error) {
        switch cfg.Provider {
        case ProviderAnthropic:
                return callAnthropic(ctx, cfg, history, autoMode, skillsMode, skillsConfig)
        default:
                return callOpenAICompat(ctx, cfg, history, autoMode, skillsMode, skillsConfig)
        }
}

// buildSystemPrompt returns the effective system prompt, merging auto-mode and skills instructions.
func buildSystemPrompt(cfg *Config, autoMode, skillsMode bool, skillsConfig *SkillsConfig) string {
        base := cfg.SystemPrompt

        if autoMode {
                if base != "" {
                        base = base + "\n\n" + autoSystemPrompt
                } else {
                        base = autoSystemPrompt
                }
        }

        if skillsMode && skillsConfig != nil {
                base = base + skillsSystemPrompt
                if skillsConfig.Instructions != "" {
                        base = base + "\n\nSKILLS INSTRUCTIONS:\n" + skillsConfig.Instructions
                }
        }

        return base
}

// buildOpenAIMessages converts history + system prompt into an OpenAI messages array.
func buildOpenAIMessages(cfg *Config, history []ChatMessage, autoMode, skillsMode bool, skillsConfig *SkillsConfig) []ChatMessage {
        sys := buildSystemPrompt(cfg, autoMode, skillsMode, skillsConfig)
        if sys == "" {
                return history
        }
        msgs := make([]ChatMessage, 0, len(history)+1)
        msgs = append(msgs, ChatMessage{Role: "system", Content: sys})
        msgs = append(msgs, history...)
        return msgs
}

// ---- OpenAI-compatible (OpenRouter, OpenAI, Ollama, LM Studio) ----

type openAIRequest struct {
        Model    string        `json:"model"`
        Messages []ChatMessage `json:"messages"`
        Stream   bool          `json:"stream,omitempty"`
}

type openAIResponse struct {
        Choices []struct {
                Delta   struct{ Content string `json:"content"` } `json:"delta"`
                Message struct{ Content string `json:"content"` } `json:"message"`
                FinishReason string `json:"finish_reason"`
        } `json:"choices"`
        Error *struct {
                Message string `json:"message"`
                Type    string `json:"type"`
        } `json:"error,omitempty"`
}

type ollamaResponse struct {
        Message struct {
                Role    string `json:"role"`
                Content string `json:"content"`
        } `json:"message"`
        Done  bool   `json:"done"`
        Error string `json:"error,omitempty"`
}

func callOpenAICompat(ctx context.Context, cfg *Config, history []ChatMessage, autoMode, skillsMode bool, skillsConfig *SkillsConfig) (*LLMResponse, error) {
        if cfg.Provider == ProviderOllama {
                return callOllama(ctx, cfg, history, autoMode, skillsMode, skillsConfig)
        }

        msgs := buildOpenAIMessages(cfg, history, autoMode, skillsMode, skillsConfig)
        if len(history) > 0 && history[0].Role == "system" {
                msgs = history
        }

        apiURL := cfg.apiURL()
        body := openAIRequest{Model: cfg.Model, Messages: msgs, Stream: cfg.Stream}

        return doOpenAIRequest(ctx, cfg, apiURL, body, autoMode)
}

func doOpenAIRequest(ctx context.Context, cfg *Config, apiURL string, body openAIRequest, autoMode bool) (*LLMResponse, error) {
        data, err := json.Marshal(body)
        if err != nil {
                return nil, err
        }

        debugf(cfg, "POST %s  model=%s  messages=%d", apiURL, body.Model, len(body.Messages))

        ctx, cancel := context.WithTimeout(ctx, cfg.LLMTimeout)
        defer cancel()

        req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewReader(data))
        if err != nil {
                return nil, err
        }
        req.Header.Set("Content-Type", "application/json")
        req.Header.Set("User-Agent", userAgent)
        if cfg.APIKey != "" {
                req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
        }
        if cfg.Provider == ProviderOpenRouter {
                req.Header.Set("X-Title", "openllm-cli")
        }

        resp, err := http.DefaultClient.Do(req)
        if err != nil {
                return nil, err
        }
        defer resp.Body.Close()

        if !cfg.Stream {
                return parseOpenAIResponse(resp)
        }
        return streamOpenAIResponse(cfg, resp, autoMode)
}

func parseOpenAIResponse(resp *http.Response) (*LLMResponse, error) {
        raw, err := io.ReadAll(resp.Body)
        if err != nil {
                return nil, err
        }
        if resp.StatusCode != http.StatusOK {
                var e openAIResponse
                if json.Unmarshal(raw, &e) == nil && e.Error != nil {
                        return nil, fmt.Errorf("API error %d (%s): %s", resp.StatusCode, e.Error.Type, e.Error.Message)
                }
                return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, raw)
        }
        var r openAIResponse
        if err := json.Unmarshal(raw, &r); err != nil {
                return nil, err
        }
        if r.Error != nil {
                return nil, fmt.Errorf("API error (%s): %s", r.Error.Type, r.Error.Message)
        }
        if len(r.Choices) == 0 {
                return nil, fmt.Errorf("no choices in response")
        }
        return &LLMResponse{Text: r.Choices[0].Message.Content}, nil
}

func streamOpenAIResponse(cfg *Config, resp *http.Response, autoMode bool) (*LLMResponse, error) {
        if resp.StatusCode != http.StatusOK {
                raw, _ := io.ReadAll(resp.Body)
                return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, raw)
        }

        var full strings.Builder
        scanner := bufio.NewScanner(resp.Body)
        if !autoMode {
                printStreamHeader(cfg)
        }

        for scanner.Scan() {
                line := scanner.Text()
                if !strings.HasPrefix(line, "data: ") {
                        continue
                }
                data := line[6:]
                if data == "[DONE]" {
                        break
                }
                var chunk openAIResponse
                if err := json.Unmarshal([]byte(data), &chunk); err != nil {
                        debugf(cfg, "stream: bad JSON chunk: %v", err)
                        continue
                }
                if chunk.Error != nil {
                        return nil, fmt.Errorf("stream error (%s): %s", chunk.Error.Type, chunk.Error.Message)
                }
                if len(chunk.Choices) > 0 {
                        tok := chunk.Choices[0].Delta.Content
                        if tok != "" {
                                if !autoMode {
                                        fmt.Print(tok)
                                }
                                full.WriteString(tok)
                        }
                        if chunk.Choices[0].FinishReason != "" {
                                break
                        }
                }
        }
        if !autoMode {
                fmt.Println()
        }
        return &LLMResponse{Text: full.String()}, scanner.Err()
}

// ---- Ollama (uses same format but different streaming protocol) ----

func callOllama(ctx context.Context, cfg *Config, history []ChatMessage, autoMode, skillsMode bool, skillsConfig *SkillsConfig) (*LLMResponse, error) {
        msgs := buildOpenAIMessages(cfg, history, autoMode, skillsMode, skillsConfig)
        if len(history) > 0 && history[0].Role == "system" {
                msgs = history
        }

        body := map[string]interface{}{
                "model":    cfg.Model,
                "messages": msgs,
                "stream":   cfg.Stream,
        }
        data, _ := json.Marshal(body)

        debugf(cfg, "POST %s  model=%s", cfg.OllamaURL, cfg.Model)

        ctx, cancel := context.WithTimeout(ctx, cfg.LLMTimeout)
        defer cancel()

        req, err := http.NewRequestWithContext(ctx, "POST", cfg.OllamaURL, bytes.NewReader(data))
        if err != nil {
                return nil, err
        }
        req.Header.Set("Content-Type", "application/json")

        resp, err := http.DefaultClient.Do(req)
        if err != nil {
                return nil, err
        }
        defer resp.Body.Close()

        if !cfg.Stream {
                raw, _ := io.ReadAll(resp.Body)
                var r ollamaResponse
                if err := json.Unmarshal(raw, &r); err != nil {
                        return nil, err
                }
                if r.Error != "" {
                        return nil, fmt.Errorf("Ollama error: %s", r.Error)
                }
                return &LLMResponse{Text: r.Message.Content}, nil
        }

        // Streaming: newline-delimited JSON
        var full strings.Builder
        scanner := bufio.NewScanner(resp.Body)
        if !autoMode {
                printStreamHeader(cfg)
        }
        for scanner.Scan() {
                var chunk ollamaResponse
                if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
                        debugf(cfg, "ollama stream: bad JSON chunk: %v", err)
                        continue
                }
                if chunk.Error != "" {
                        return nil, fmt.Errorf("Ollama stream error: %s", chunk.Error)
                }
                tok := chunk.Message.Content
                if tok != "" {
                        if !autoMode {
                                fmt.Print(tok)
                        }
                        full.WriteString(tok)
                }
                if chunk.Done {
                        break
                }
        }
        if !autoMode {
                fmt.Println()
        }
        return &LLMResponse{Text: full.String()}, scanner.Err()
}

// ---- Anthropic ----

type anthropicRequest struct {
        Model     string        `json:"model"`
        MaxTokens int           `json:"max_tokens"`
        System    string        `json:"system,omitempty"`
        Messages  []ChatMessage `json:"messages"`
        Stream    bool          `json:"stream,omitempty"`
}

type anthropicResponse struct {
        Content []struct {
                Type string `json:"type"`
                Text string `json:"text"`
        } `json:"content"`
        StopReason string `json:"stop_reason"`
        Error      *struct {
                Type    string `json:"type"`
                Message string `json:"message"`
        } `json:"error,omitempty"`
}

type anthropicSSE struct {
        Type  string `json:"type"`
        Delta struct {
                Type string `json:"type"`
                Text string `json:"text"`
        } `json:"delta"`
        Error *struct {
                Type    string `json:"type"`
                Message string `json:"message"`
        } `json:"error,omitempty"`
}

func callAnthropic(ctx context.Context, cfg *Config, history []ChatMessage, autoMode, skillsMode bool, skillsConfig *SkillsConfig) (*LLMResponse, error) {
        // Anthropic puts system at the top level, not as a message
        sys := buildSystemPrompt(cfg, autoMode, skillsMode, skillsConfig)
        msgs := make([]ChatMessage, 0, len(history))
        for _, m := range history {
                if m.Role == "system" {
                        // Carry system messages that were injected for auto mode into the top-level field
                        if sys == "" {
                                sys = m.Content
                        } else {
                                sys += "\n\n" + m.Content
                        }
                } else {
                        msgs = append(msgs, m)
                }
        }

        body := anthropicRequest{
                Model:     cfg.Model,
                MaxTokens: cfg.MaxTokens,
                System:    sys,
                Messages:  msgs,
                Stream:    cfg.Stream,
        }

        data, _ := json.Marshal(body)
        debugf(cfg, "POST %s  model=%s  max_tokens=%d", anthropicURL, cfg.Model, cfg.MaxTokens)

        ctx, cancel := context.WithTimeout(ctx, cfg.LLMTimeout)
        defer cancel()

        req, err := http.NewRequestWithContext(ctx, "POST", anthropicURL, bytes.NewReader(data))
        if err != nil {
                return nil, err
        }
        req.Header.Set("Content-Type", "application/json")
        req.Header.Set("User-Agent", userAgent)
        req.Header.Set("x-api-key", cfg.APIKey)
        req.Header.Set("anthropic-version", anthropicVersion)

        resp, err := http.DefaultClient.Do(req)
        if err != nil {
                return nil, err
        }
        defer resp.Body.Close()

        if !cfg.Stream {
                raw, _ := io.ReadAll(resp.Body)
                if resp.StatusCode != http.StatusOK {
                        var e anthropicResponse
                        if json.Unmarshal(raw, &e) == nil && e.Error != nil {
                                return nil, fmt.Errorf("Anthropic error %d (%s): %s", resp.StatusCode, e.Error.Type, e.Error.Message)
                        }
                        return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, raw)
                }
                var r anthropicResponse
                if err := json.Unmarshal(raw, &r); err != nil {
                        return nil, err
                }
                if r.Error != nil {
                        return nil, fmt.Errorf("Anthropic error (%s): %s", r.Error.Type, r.Error.Message)
                }
                var sb strings.Builder
                for _, block := range r.Content {
                        if block.Type == "text" {
                                sb.WriteString(block.Text)
                        }
                }
                return &LLMResponse{Text: sb.String(), StopReason: r.StopReason}, nil
        }

        // Streaming
        if resp.StatusCode != http.StatusOK {
                raw, _ := io.ReadAll(resp.Body)
                return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, raw)
        }
        var full strings.Builder
        scanner := bufio.NewScanner(resp.Body)
        if !autoMode {
                printStreamHeader(cfg)
        }
        for scanner.Scan() {
                line := scanner.Text()
                if !strings.HasPrefix(line, "data: ") {
                        continue
                }
                var event anthropicSSE
                if err := json.Unmarshal([]byte(line[6:]), &event); err != nil {
                        debugf(cfg, "anthropic stream: bad JSON chunk: %v", err)
                        continue
                }
                switch event.Type {
                case "content_block_delta":
                        if event.Delta.Text != "" {
                                if !autoMode {
                                        fmt.Print(event.Delta.Text)
                                }
                                full.WriteString(event.Delta.Text)
                        }
                case "message_stop":
                        goto done
                case "error":
                        if event.Error != nil {
                                return nil, fmt.Errorf("Anthropic stream error (%s): %s", event.Error.Type, event.Error.Message)
                        }
                }
        }
done:
        if !autoMode {
                fmt.Println()
        }
        return &LLMResponse{Text: full.String()}, scanner.Err()
}

func (cfg *Config) apiURL() string {
        switch cfg.Provider {
        case ProviderOpenAI:
                return openAIURL
        case ProviderOpenRouter:
                return openRouterURL
        case ProviderLMStudio:
                return cfg.LMStudioURL
        default:
                return openRouterURL
        }
}

// ============================================================
// Single-shot pipe mode
// ============================================================

func runPipe(cfg *Config) {
        raw, err := io.ReadAll(os.Stdin)
        if err != nil {
                fatalf("reading stdin: %v", err)
        }
        input := strings.TrimSpace(string(raw))
        if input == "" {
                fatalf("stdin is empty")
        }
        // Expand inline references even in pipe mode
        expanded, err := expandInline(input, cfg)
        if err != nil {
                fatalf("expand: %v", err)
        }

        s := newSession(cfg, false, cfg.SkillsMode)
        s.addUser(expanded)

        resp, err := s.send()
        if err != nil {
                fatalf("LLM: %v", err)
        }

        responseText := resp.Text
        if cfg.SkillsMode {
                responseText = s.processSkillsInline(resp.Text)
        }

        if !cfg.Stream {
                fmt.Println(responseText)
        }
}

// ============================================================
// main
// ============================================================

func main() {
        cfg := loadConfig()

        interactive := isTruthy(envInteractive) // can also be forced via env
        autoMode := false
        skillsMode := cfg.SkillsMode

        // Simple flag parsing — no external deps
        args := os.Args[1:]
        for _, arg := range args {
                switch arg {
                case "-i", "--interactive":
                        interactive = true
                case "-a", "--auto":
                        interactive = true
                        autoMode = true
                case "-s", "--skills":
                        interactive = true
                        skillsMode = true
                case "-h", "--help":
                        printHelp()
                        os.Exit(0)
                }
        }

        // Auto-detect interactive when stdin is a terminal (not a pipe)
        if !interactive && isTerminal() {
                interactive = true
        }

        if interactive {
                runInteractive(cfg, autoMode, skillsMode)
        } else {
                runPipe(cfg)
        }
}

// isTerminal returns true when stdin is connected to a real terminal.
func isTerminal() bool {
        fi, err := os.Stdin.Stat()
        if err != nil {
                return false
        }
        return (fi.Mode() & os.ModeCharDevice) != 0
}

// ============================================================
// UI helpers
// ============================================================

func printBanner(cfg *Config, autoMode, skillsMode bool) {
        mode := "chat"
        if autoMode {
                mode = "auto"
        }
        if skillsMode {
                if autoMode {
                        mode = "auto+skills"
                } else {
                        mode = "skills"
                }
        }
        wd, _ := os.Getwd()
        fmt.Printf("\n%s%s openllm-cli %s%s  %s%s · %s · %s%s\n",
                colorBold, colorCyan, version, colorReset,
                colorDim, cfg.Provider, cfg.Model, mode, colorReset,
        )
        fmt.Printf("%scwd: %s%s\n", colorDim, wd, colorReset)

        helpText := "/help for commands · /auto to toggle auto mode"
        if skillsMode {
                helpText += " · /skills to toggle skills mode"
        }
        helpText += " · /exit to quit"

        fmt.Printf("%s%s%s\n\n", colorDim, helpText, colorReset)
}

func printAssistant(cfg *Config, text string) {
        if cfg.Stream {
                // Already printed token by token
                return
        }
        fmt.Printf("\n%s%s%s\n%s", colorBold, colorGreen, cfg.Model, colorReset)
        fmt.Println(text)
}

func printStreamHeader(cfg *Config) {
        fmt.Printf("\n%s%s%s\n%s", colorBold, colorGreen, cfg.Model, colorReset)
}

func printToolCall(tc *ToolCall) {
        icon := map[string]string{
                toolReadFile:  "📖",
                toolWriteFile: "✏️ ",
                toolRunShell:  "⚙️ ",
                toolDone:      "✓ ",
        }[tc.Name]
        fmt.Printf("\n%s%s %s%s  ", colorBold, colorBlue, icon, colorReset)
        switch tc.Name {
        case toolReadFile:
                fmt.Printf("read %s\n", tc.Input["path"])
        case toolWriteFile:
                fmt.Printf("write %s  (%d bytes)\n", tc.Input["path"], len(tc.Input["content"]))
        case toolRunShell:
                fmt.Printf("%s%s%s\n", colorYellow, tc.Input["command"], colorReset)
        case toolDone:
                fmt.Printf("task complete\n")
        }
}

func printToolResult(result string) {
        lines := strings.Split(result, "\n")
        preview := result
        if len(lines) > 10 {
                preview = strings.Join(lines[:10], "\n") + fmt.Sprintf("\n%s… (%d more lines)%s", colorDim, len(lines)-10, colorReset)
        }
        fmt.Printf("%s%s%s\n", colorDim, preview, colorReset)
}

func printError(msg string) {
        fmt.Fprintf(os.Stderr, "%s%s error:%s %s\n", colorBold, colorRed, colorReset, msg)
}

func infof(format string, args ...interface{}) {
        fmt.Printf("%s"+format+"%s\n", append([]interface{}{colorDim}, append(args, colorReset)...)...)
}

func debugf(cfg *Config, format string, args ...interface{}) {
        if cfg.Verbose {
                log.Printf("[DEBUG] "+format, args...)
        }
}

func printCommandHelp() {
        type row struct{ cmd, desc string }
        sections := []struct {
                name string
                rows []row
        }{
                {"Session", []row{
                        {"/help", "show this help"},
                        {"/exit  /quit  /q", "quit"},
                        {"/clear  /reset", "clear conversation history"},
                        {"/history", "show message history"},
                }},
                {"Config", []row{
                        {"/model [name]", "get/set model"},
                        {"/system [text]", "get/set system prompt"},
                        {"/stream", "toggle streaming"},
                        {"/auto", "toggle auto mode (tool use)"},
                        {"/skills", "toggle skills mode (SKILL.md integration)"},
                        {"/approve", "toggle auto-approve for tool actions"},
                        {"/maxtokens [n]", "get/set max_tokens (Anthropic)"},
                }},
                {"Filesystem", []row{
                        {"/read <path>", "read file or directory; optionally send to LLM"},
                        {"/write <path>", "write last LLM reply (or typed text) to a file"},
                        {"/ls [path]", "list directory"},
                        {"/pwd", "working directory"},
                        {"/cd <path>", "change directory"},
                }},
                {"Shell", []row{
                        {"/run <cmd>", "run shell command; optionally send output to LLM"},
                        {"/shell  /exec", "aliases for /run"},
                }},
        }

        fmt.Println()
        for _, sec := range sections {
                fmt.Printf("%s%s%s\n", colorBold, sec.name, colorReset)
                for _, r := range sec.rows {
                        fmt.Printf("  %s%-22s%s  %s%s%s\n", colorYellow, r.cmd, colorReset, colorDim, r.desc, colorReset)
                }
                fmt.Println()
        }

        fmt.Printf("%sInline syntax (in any message)%s\n", colorBold, colorReset)
        fmt.Printf("  %s@path/to/file%s   inject file or directory into prompt\n", colorMagenta, colorReset)
        fmt.Printf("  %s`cmd`%s           run command and inject output into prompt\n", colorMagenta, colorReset)
        fmt.Println()

        fmt.Printf("%sSkills mode%s — controlled command execution and file access:\n", colorBold, colorReset)
        fmt.Printf("  Start with %s-s%s / %s--skills%s, or type %s/skills%s to toggle.\n", colorYellow, colorReset, colorYellow, colorReset, colorYellow, colorReset)
        fmt.Printf("  Reads SKILL.md/Skills.md for allowed commands and paths.\n")
        fmt.Printf("  AI can use inline `commands` and @file references safely.\n\n")

        fmt.Printf("%sAuto mode%s — the LLM can use tools autonomously:\n", colorBold, colorReset)
        fmt.Printf("  Start with %s-a%s / %s--auto%s, or type %s/auto%s to toggle.\n", colorYellow, colorReset, colorYellow, colorReset, colorYellow, colorReset)
        fmt.Printf("  Tools: read_file · write_file · run_shell\n")
        fmt.Printf("  All tool actions run without confirmation in auto mode.\n")
        fmt.Printf("  Set %sLLM_AUTO_APPROVE=1%s to skip confirmations in non-auto modes.\n\n", colorYellow, colorReset)

        fmt.Printf("%sExamples%s\n", colorBold, colorReset)
        fmt.Printf("  openllm-cli > review @./main.go and list any bugs\n")
        fmt.Printf("  openllm-cli > `ps aux` what are all these processes?\n")
        fmt.Printf("  openllm-cli -a > read package.json and update the version to 2.0.0\n")
        fmt.Printf("  openllm-cli -s > what system info can you gather?\n\n")
}

func printHelp() {
        fmt.Printf(`%s%sopenllm-cli%s

Interactive LLM CLI with filesystem access, auto mode, and skills integration.

%sUsage:%s
  openllm-cli                  interactive REPL (auto-detected)
  openllm-cli -i               force interactive mode
  openllm-cli -a               interactive mode with auto tool use
  openllm-cli -s               interactive mode with skills integration
  echo "prompt" | openllm-cli  single-shot pipe mode

%sFlags:%s
  -i, --interactive   interactive REPL
  -a, --auto          auto mode (LLM can read/write files and run commands)
  -s, --skills        skills mode (read SKILL.md and allow inline commands/files)
  -h, --help          show this help

%sEnvironment:%s
  LLM_PROVIDER         ollama* | openrouter | openai | anthropic | lmstudio
  LLM_MODEL            model name (provider default used if unset)
  LLM_STREAM           1/true to enable streaming tokens
  LLM_SYSTEM_PROMPT    system prompt
  LLM_MAX_TOKENS       max tokens to generate (default %d)
  LLM_TIMEOUT          LLM request timeout seconds (default 120 / 300 streaming)
  LLM_SHELL_TIMEOUT    shell command timeout seconds (default 60)
  LLM_AUTO_APPROVE     1/true to skip tool-use confirmations
  LLM_SKILLS_MODE      1/true to enable skills mode by default
  LLM_VERBOSE          1/true for debug logging

  OPENROUTER_API_KEY   required for openrouter
  OPENAI_API_KEY       required for openai
  ANTHROPIC_API_KEY    required for anthropic
  OLLAMA_URL           ollama endpoint (default %s)
  LM_STUDIO_URL        lm studio endpoint (default %s)

%sModel defaults:%s
  openrouter   %s
  openai       %s
  anthropic    %s
  ollama       %s
  lmstudio     %s

%sSkills configuration (SKILL.md/Skills.md):%s
  ALLOWED_COMMANDS: curl,id,whoami,date
  ALLOWED_PATHS: .,./config,/tmp
  DISALLOWED_PATHS: ~/.ssh,/etc
  AUTO_EXECUTE: true

  ## Instructions
  You are a system assistant that can gather information safely.
`,
                colorBold, colorCyan, colorReset,
                colorBold, colorReset,
                colorBold, colorReset,
                colorBold, colorReset,
                defaultMaxTokens,
                defaultOllamaURL,
                defaultLMStudioURL,
                colorBold, colorReset,
                defaultOpenRouterModel,
                defaultOpenAIModel,
                defaultAnthropicModel,
                defaultOllamaModel,
                defaultLMStudioModel,
                colorBold, colorReset,
        )
}

// ============================================================
// Small utilities
// ============================================================

func isTruthy(key string) bool {
        v := strings.ToLower(strings.TrimSpace(os.Getenv(key)))
        return v == "1" || v == "true" || v == "yes" || v == "on"
}

func envOr(key, fallback string) string {
        if v := strings.TrimSpace(os.Getenv(key)); v != "" {
                return v
        }
        return fallback
}

func fatalf(format string, args ...interface{}) {
        fmt.Fprintf(os.Stderr, colorRed+"error: "+colorReset+format+"\n", args...)
        os.Exit(1)
}

// stdinReadLine reads one line from stdin in cooked mode (used for sub-prompts).
func stdinReadLine() string {
        r := bufio.NewReader(os.Stdin)
        line, _ := r.ReadString('\n')
        return strings.TrimRight(line, "\r\n")
}

func askYN(prompt string) bool {
        fmt.Printf("%s%s [y/N]%s ", colorDim, prompt, colorReset)
        return strings.ToLower(stdinReadLine()) == "y"
}

func askLine(prompt string) string {
        fmt.Printf("%s%s%s ", colorDim, prompt, colorReset)
        return stdinReadLine()
}
