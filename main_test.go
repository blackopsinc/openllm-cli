package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestParseToolCallIgnoresClosingTagBeforeOpeningTag(t *testing.T) {
	tc := parseToolCall("</run_shell> prose <run_shell><command>pwd</command></run_shell>")
	if tc == nil || tc.Name != toolRunShell || tc.Input["command"] != "pwd" {
		t.Fatalf("parseToolCall() = %#v, want run_shell pwd", tc)
	}
}

func TestParseToolCallDecodesEscapedContentCloseTag(t *testing.T) {
	tc := parseToolCall("<write_file><path>x.txt</path><content>before<&#47;content>after</content></write_file>")
	if tc == nil {
		t.Fatal("parseToolCall() returned nil")
	}
	if got := tc.Input["content"]; got != "before</content>after" {
		t.Fatalf("content = %q, want decoded closing tag", got)
	}
}

func TestResolveToolPathConfinesWorkspace(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "inside.txt"), []byte("ok"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := resolveToolPath(root, "inside.txt", false); err != nil {
		t.Fatalf("inside path rejected: %v", err)
	}
	if _, err := resolveToolPath(root, "../outside.txt", true); err == nil {
		t.Fatal("parent traversal was accepted")
	}
	if _, err := resolveToolPath(root, filepath.Join(outside, "outside.txt"), true); err == nil {
		t.Fatal("absolute path was accepted")
	}

	link := filepath.Join(root, "outside-link")
	if err := os.Symlink(outside, link); err != nil {
		t.Logf("symlink checks skipped: %v", err)
		return
	}
	if _, err := resolveToolPath(root, "outside-link", false); err == nil {
		t.Fatal("read through an escaping symlink was accepted")
	}
	if _, err := resolveToolPath(root, filepath.Join("outside-link", "new.txt"), true); err == nil {
		t.Fatal("write through an escaping symlink was accepted")
	}
}

func TestExpandInlineUsesSessionWorkingDirectory(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte("from session cwd"), 0600); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{ShellTimeout: time.Second}
	expanded, err := expandInline("review @note.txt", cfg, dir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(expanded, "from session cwd") {
		t.Fatalf("expanded text does not contain file contents: %q", expanded)
	}

	if runtime.GOOS != "windows" {
		expanded, err = expandInline("`pwd`", cfg, dir)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(expanded, dir) {
			t.Fatalf("command did not run in session cwd: %q", expanded)
		}
	}
}

func TestExpandAtFilesLeavesEmailAddressesAlone(t *testing.T) {
	input := "contact dev@example.com"
	got, err := expandAtFiles(input, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if got != input {
		t.Fatalf("expandAtFiles() = %q, want %q", got, input)
	}
}

func TestWriteFileAtomicPreservesModeAndCleansTemporaryFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.txt")
	if err := os.WriteFile(path, []byte("old"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := writeFileAtomic(path, "new"); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "new" {
		t.Fatalf("content = %q, want new", data)
	}
	if runtime.GOOS != "windows" {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if got := info.Mode().Perm(); got != 0600 {
			t.Fatalf("mode = %o, want 600", got)
		}
	}
	matches, err := filepath.Glob(filepath.Join(dir, ".data.txt.tmp-*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 0 {
		t.Fatalf("temporary files left behind: %v", matches)
	}
}

func TestDirectoryListingIsBounded(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < maxDirectoryEntries+10; i++ {
		name := filepath.Join(dir, fmt.Sprintf("file-%04d", i))
		if err := os.WriteFile(name, nil, 0600); err != nil {
			t.Fatal(err)
		}
	}
	listing, err := dirListing(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(listing, "listing truncated after 500 entries") {
		t.Fatalf("listing was not truncated: %q", listing[len(listing)-100:])
	}
}

func TestCappedBufferBoundsStoredOutput(t *testing.T) {
	buffer := newCappedBuffer(4)
	if n, err := buffer.Write([]byte("abcdefgh")); err != nil || n != 8 {
		t.Fatalf("Write() = %d, %v", n, err)
	}
	got := buffer.String()
	if !strings.HasPrefix(got, "abcd") || !strings.Contains(got, "8 bytes total") {
		t.Fatalf("String() = %q", got)
	}
	if buffer.buf.Len() != 4 {
		t.Fatalf("stored bytes = %d, want 4", buffer.buf.Len())
	}
}

func TestStreamScannerAcceptsLargeEvents(t *testing.T) {
	line := strings.Repeat("x", 128*1024)
	scanner := newStreamScanner(strings.NewReader(line + "\n"))
	if !scanner.Scan() || scanner.Text() != line || scanner.Err() != nil {
		t.Fatalf("large stream event was not scanned: %v", scanner.Err())
	}
}

func TestOpenAIStreamAcceptsDataWithoutSpace(t *testing.T) {
	body := "data:{\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"\"}]}\n" +
		"data:{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n"
	resp := &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(body))}
	result, err := streamOpenAIResponse(&Config{}, resp, true)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || result.StopReason != "stop" {
		t.Fatalf("stream result = %#v", result)
	}
}

func TestInputBrokerStopsBeforeNextPrompt(t *testing.T) {
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	broker := newInputBroker(reader)
	s := &Session{input: broker}
	done := make(chan struct{})
	_, stopped := s.startAutoInterruptReader(done)
	close(done)
	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Fatal("interrupt reader did not stop")
	}

	go func() { _, _ = io.WriteString(writer, "next prompt\n") }()
	line, err := broker.readLine()
	if err != nil {
		t.Fatal(err)
	}
	if line != "next prompt" {
		t.Fatalf("next line = %q, want preserved prompt input", line)
	}
}

func TestRunAutoTaskRecoversFromMissingToolCall(t *testing.T) {
	responses := []struct {
		text       string
		stopReason string
	}{
		{"<read_file><path>input.txt</path></read_file>", "stop"},
		{"I have enough information to finish.", "length"},
		{"<task_done><status>success</status><summary>finished</summary></task_done>", "stop"},
	}

	var requests atomic.Int32
	var receivedMaxTokens atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request openAIRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
		}
		receivedMaxTokens.Store(int32(request.MaxTokens))
		i := int(requests.Add(1)) - 1
		if i >= len(responses) {
			t.Errorf("unexpected request %d", i+1)
			http.Error(w, "too many requests", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message":       map[string]string{"content": responses[i].text},
				"finish_reason": responses[i].stopReason,
			}},
		})
	}))
	defer server.Close()

	dir := t.TempDir()
	if err := writeFileAtomic(filepath.Join(dir, "input.txt"), "test data"); err != nil {
		t.Fatal(err)
	}
	s := &Session{
		cfg: &Config{
			Provider:     ProviderLMStudio,
			Model:        "test-model",
			LMStudioURL:  server.URL,
			LLMTimeout:   time.Second,
			ShellTimeout: time.Second,
			MaxTokens:    321,
		},
		cwd:      dir,
		autoMode: true,
	}
	s.addUser("read input.txt")

	if err := s.runAutoTask(); err != nil {
		t.Fatalf("runAutoTask returned an error: %v", err)
	}
	if got := requests.Load(); got != 3 {
		t.Fatalf("request count = %d, want 3", got)
	}
	if got := receivedMaxTokens.Load(); got != 321 {
		t.Fatalf("max_tokens = %d, want 321", got)
	}

	var correction string
	for _, message := range s.history {
		if message.Role == "user" && strings.Contains(message.Content, "Protocol correction") {
			correction = message.Content
			break
		}
	}
	if correction == "" {
		t.Fatal("history does not contain a protocol correction")
	}
	if !strings.Contains(correction, "truncated") {
		t.Fatalf("correction does not explain truncation: %s", correction)
	}
}

func TestRunAutoTaskStopsAfterMissingToolCallRetries(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		i := requests.Add(1)
		text := "still no tool call"
		if i == 1 {
			text = "<read_file><path>input.txt</path></read_file>"
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"choices":[{"message":{"content":%q},"finish_reason":"stop"}]}`, text)
	}))
	defer server.Close()

	dir := t.TempDir()
	if err := writeFileAtomic(filepath.Join(dir, "input.txt"), "test data"); err != nil {
		t.Fatal(err)
	}
	s := &Session{
		cfg: &Config{
			Provider:     ProviderLMStudio,
			Model:        "test-model",
			LMStudioURL:  server.URL,
			LLMTimeout:   time.Second,
			ShellTimeout: time.Second,
		},
		cwd:      dir,
		autoMode: true,
	}
	s.addUser("read input.txt")

	err := s.runAutoTask()
	if err == nil || !strings.Contains(err.Error(), "after 2 recovery attempts") {
		t.Fatalf("error = %v, want exhausted recovery error", err)
	}
	if got := requests.Load(); got != 4 {
		t.Fatalf("request count = %d, want 4", got)
	}
}
