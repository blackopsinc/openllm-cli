BINARY_NAME := openllm-cli
INSTALL_DIR := /usr/local/bin
VERSION     ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS     := -ldflags="-s -w -X main.version=$(VERSION)"
BUILDFLAGS  := -trimpath $(LDFLAGS)

# Detect native OS for the local build target
NATIVE_OS := $(shell go env GOOS)
ifeq ($(NATIVE_OS),windows)
    BIN_EXT := .exe
else
    BIN_EXT :=
endif

.PHONY: all build clean install install-user fmt vet test test-coverage help \
        linux-amd64 linux-arm64 darwin-amd64 darwin-arm64 windows-amd64

# Default: cross-compile for all supported platforms
all: linux-amd64 linux-arm64 darwin-amd64 darwin-arm64 windows-amd64

# Build for the current platform
build: bin
	@echo "Building $(BINARY_NAME) $(VERSION) for $(NATIVE_OS)/$(shell go env GOARCH)..."
	@go build $(BUILDFLAGS) -o bin/$(BINARY_NAME)$(BIN_EXT) .
ifneq ($(NATIVE_OS),windows)
	@chmod +x bin/$(BINARY_NAME)$(BIN_EXT)
endif
	@echo "Done: bin/$(BINARY_NAME)$(BIN_EXT)"

# Cross-compile targets — CGO_ENABLED=0 ensures no C toolchain is required
linux-amd64: bin
	@echo "Building linux/amd64..."
	@CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build $(BUILDFLAGS) -o bin/$(BINARY_NAME)-linux-amd64 .

linux-arm64: bin
	@echo "Building linux/arm64..."
	@CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build $(BUILDFLAGS) -o bin/$(BINARY_NAME)-linux-arm64 .

darwin-amd64: bin
	@echo "Building darwin/amd64 (Intel Mac)..."
	@CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build $(BUILDFLAGS) -o bin/$(BINARY_NAME)-darwin-amd64 .

darwin-arm64: bin
	@echo "Building darwin/arm64 (Apple Silicon)..."
	@CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build $(BUILDFLAGS) -o bin/$(BINARY_NAME)-darwin-arm64 .

windows-amd64: bin
	@echo "Building windows/amd64..."
	@CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build $(BUILDFLAGS) -o bin/$(BINARY_NAME)-windows-amd64.exe .

bin:
	@mkdir -p bin

clean:
	@go clean
	@rm -rf bin/
	@echo "Clean complete"

# Unix/macOS install targets
install: build
	@echo "Installing to $(INSTALL_DIR)..."
	@sudo cp bin/$(BINARY_NAME)$(BIN_EXT) $(INSTALL_DIR)/$(BINARY_NAME)
	@sudo chmod +x $(INSTALL_DIR)/$(BINARY_NAME)
	@echo "Installed: $(INSTALL_DIR)/$(BINARY_NAME)"

install-user: build
	@mkdir -p $$HOME/.local/bin
	@cp bin/$(BINARY_NAME)$(BIN_EXT) $$HOME/.local/bin/$(BINARY_NAME)
	@chmod +x $$HOME/.local/bin/$(BINARY_NAME)
	@echo "Installed: $$HOME/.local/bin/$(BINARY_NAME)"
	@echo "Ensure $$HOME/.local/bin is on your PATH."

fmt:
	@go fmt ./...

vet:
	@go vet ./...

test:
	@go test -v ./...

test-coverage:
	@go test -v -coverprofile=coverage.out ./...
	@go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

help:
	@echo "openllm-cli $(VERSION)"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "  build            Build for current platform"
	@echo "  all              Cross-compile for all platforms (output: bin/)"
	@echo ""
	@echo "  linux-amd64      Linux x86-64"
	@echo "  linux-arm64      Linux ARM64 (Raspberry Pi, cloud ARM)"
	@echo "  darwin-amd64     macOS Intel"
	@echo "  darwin-arm64     macOS Apple Silicon"
	@echo "  windows-amd64    Windows 64-bit"
	@echo ""
	@echo "  install          Install to $(INSTALL_DIR) (requires sudo, Unix/macOS only)"
	@echo "  install-user     Install to ~/.local/bin (Unix/macOS only)"
	@echo "  clean            Remove build artifacts"
	@echo "  fmt              Run go fmt"
	@echo "  vet              Run go vet"
	@echo "  test             Run tests"
	@echo "  test-coverage    Run tests with HTML coverage report"
	@echo ""
	@echo "  Windows install: copy bin/$(BINARY_NAME)-windows-amd64.exe to a directory on your PATH"
