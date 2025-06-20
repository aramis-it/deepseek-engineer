# Multi-LLM Engineer 🤖

## Overview

Multi-LLM Engineer is an enhanced version of DeepSeek Engineer that supports multiple Large Language Model providers. It provides a unified interface for AI-powered coding assistance across different LLM providers including DeepSeek, OpenAI, Anthropic Claude, Google Gemini, and local Ollama models.

## 🚀 Key Features

### 🔄 **Universal LLM Support**
- **DeepSeek**: Advanced reasoning with Chain-of-Thought capabilities
- **OpenAI**: GPT-4o with function calling
- **Anthropic Claude**: Claude-3.5-Sonnet with tool use
- **Google Gemini**: Gemini-1.5-Pro (integration in progress)
- **Ollama**: Local models with OpenAI-compatible API

### 🛠️ **Function Calling Tools**
All providers support these operations (where technically possible):

#### `read_file(file_path: str)`
- Read single file content with automatic path normalization
- Built-in error handling for missing or inaccessible files

#### `read_multiple_files(file_paths: List[str])`
- Batch read multiple files efficiently
- Formatted output with clear file separators

#### `create_file(file_path: str, content: str)`
- Create new files or overwrite existing ones
- Automatic directory creation and safety checks

#### `create_multiple_files(files: List[Dict])`
- Create multiple files in a single operation
- Perfect for scaffolding projects or creating related files

#### `edit_file(file_path: str, original_snippet: str, new_snippet: str)`
- Precise snippet-based file editing
- Safe replacement with exact matching

#### `switch_provider(provider: str)`
- Dynamically switch between LLM providers during conversation
- Maintains conversation context across providers

### 🎨 **Rich Terminal Interface**
- **Provider-aware feedback** showing current LLM in use
- **Real-time streaming** with provider-specific features
- **Color-coded responses** for different providers
- **Progress indicators** for long operations

### 🛡️ **Security & Safety**
- **Path normalization** and validation
- **Directory traversal protection**
- **File size limits** (5MB per file)
- **API key isolation** per provider

## Getting Started

### Prerequisites
1. **Python 3.11+**: Required for optimal performance
2. **API Keys**: At least one LLM provider API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd deepseek-engineer
   ```

2. **Set up environment**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your API keys
   nano .env  # or use your preferred editor
   ```

3. **Install dependencies**:

   #### Option 1: All providers (recommended):
   ```bash
   pip install -r requirements-all-providers.txt
   ```

   #### Option 2: Core only, then add providers as needed:
   ```bash
   # Core dependencies only
   pip install -r requirements-multi.txt
   
   # Add specific providers
   pip install anthropic>=0.34.0              # For Claude
   pip install google-generativeai>=0.8.0     # For Gemini
   ```

   #### Option 3: Using uv (recommended - faster):
   ```bash
   uv venv
   uv pip install -r requirements-all-providers.txt
   ```

   #### Provider-specific SDKs:
   - **DeepSeek & OpenAI**: Use `openai` package (included in core)
   - **Anthropic Claude**: Requires `anthropic` package
   - **Google Gemini**: Requires `google-generativeai` package
   - **Ollama**: Uses OpenAI-compatible API (no additional SDK needed)

### Configuration

Edit your `.env` file with your API keys:

```env
# Choose your primary provider
LLM_PROVIDER=deepseek

# Add your API keys (only for providers you want to use)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# For local Ollama (optional)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2
```

### Usage

```bash
# Run the multi-LLM version
python3 multi-llm-engineer.py

# Or with uv
uv run multi-llm-engineer.py
```

## Usage Examples

### **Provider Switching**
```
You> switch to openai
✓ Switched to OpenAI (gpt-4o)

You> use claude
✓ Switched to Claude (claude-3-5-sonnet-20241022)

You> change to deepseek
✓ Switched to DeepSeek (deepseek-reasoner)
```

### **Cross-Provider File Operations**
```
You> Can you read main.py and create tests using OpenAI?

🤖 DeepSeek Processing...
💭 Reasoning: I need to read the main.py file first...

⚡ Executing 1 function call(s)...
→ read_file
✓ Read file 'main.py'

🔄 DeepSeek processing results...
I'll switch to OpenAI to create the tests as requested.

⚡ Executing 1 function call(s)...
→ switch_provider
✓ Switched to OpenAI

🤖 OpenAI Processing...
Now I'll create comprehensive tests based on the code structure.

⚡ Executing 1 function call(s)...
→ create_file
✓ Created file 'test_main.py'
```

### **Provider-Specific Features**
```
You> Use DeepSeek's reasoning to analyze this algorithm

🤖 DeepSeek Processing...
💭 Reasoning: This algorithm appears to be implementing a binary search...
[Detailed reasoning process shown]

🤖 DeepSeek> Based on my analysis, this algorithm has O(log n) complexity...
```

## Provider Comparison

| Provider | SDK Required | Reasoning | Function Calls | Streaming | Local |
|----------|--------------|-----------|----------------|-----------|-------|
| **DeepSeek** | `openai` | ✅ Chain-of-Thought | ✅ Full Support | ✅ Yes | ❌ No |
| **OpenAI** | `openai` | ❌ No | ✅ Full Support | ✅ Yes | ❌ No |
| **Claude** | `anthropic` | ❌ No | ✅ Basic Support | ✅ Yes | ❌ No |
| **Gemini** | `google-generativeai` | ❌ No | 🚧 In Progress | ✅ Yes | ❌ No |
| **Ollama** | `openai` | ❌ No | ✅ Full Support | ✅ Yes | ✅ Yes |

### SDK Installation:
```bash
# All providers
pip install -r requirements-all-providers.txt

# Individual SDKs
pip install openai>=1.58.1                 # DeepSeek, OpenAI, Ollama
pip install anthropic>=0.34.0              # Claude
pip install google-generativeai>=0.8.0     # Gemini
```

## Advanced Features

### **Dynamic Provider Selection**
The system automatically detects available providers based on your API keys:

```python
# Available providers are shown at startup
Available Providers: deepseek, openai, ollama
Current Provider: DeepSeek (deepseek-reasoner)
```

### **Context Preservation**
Conversation context is maintained when switching providers:

```
You> Remember this: my project uses FastAPI
🤖 DeepSeek> I'll remember that your project uses FastAPI.

You> switch to openai
✓ Switched to OpenAI

You> Create a new endpoint
🤖 OpenAI> I'll create a new FastAPI endpoint for your project...
```

### **Provider-Specific Optimizations**
- **DeepSeek**: Utilizes reasoning capabilities for complex analysis
- **OpenAI**: Optimized for fast function calling
- **Ollama**: Configured for local model efficiency

## File Operations

### **Automatic File Reading**
```
You> Review the utils.py file and suggest improvements

🤖 Processing...
⚡ Executing 1 function call(s)...
→ read_file
✓ Read file 'utils.py'

Based on my analysis of utils.py, here are my suggestions...
```

### **Batch Operations**
```
You> Create a complete Flask API structure

🤖 Processing...
⚡ Executing 1 function call(s)...
→ create_multiple_files
✓ Created 4 files: app.py, models.py, routes.py, tests.py
```

## Troubleshooting

### **Provider Not Available**
```
❌ Provider 'claude' not available. Available: deepseek, openai
```
**Solution**: Check your API key in `.env` and install required dependencies.

### **API Key Issues**
```
❌ No LLM providers configured! Please set up API keys in .env file.
```
**Solution**: Add at least one valid API key to your `.env` file.

### **Import Errors**
```
Warning: anthropic package not installed. Claude support disabled.
```
**Solution**: Install the optional dependency:
```bash
pip install anthropic
```

## Development

### **Adding New Providers**
To add support for a new LLM provider:

1. Add provider configuration in the `PROVIDERS` dictionary
2. Implement provider-specific streaming function
3. Add provider to the switch logic
4. Update documentation

### **Testing**
```bash
# Test with different providers
python3 multi-llm-engineer.py
# In the app: switch to openai, test functionality
# In the app: switch to deepseek, test reasoning
```

## Migration from Original

To migrate from the original DeepSeek Engineer:

1. **Backup your `.env`**: Your existing DeepSeek API key will work
2. **Install new dependencies**: `pip install -r requirements-multi.txt`
3. **Run new version**: `python3 multi-llm-engineer.py`
4. **Optional**: Add additional provider API keys for more options

## Contributing

Contributions are welcome! Priority areas:
- Complete Claude and Gemini integrations
- Add more local model support
- Improve provider-specific optimizations
- Add provider benchmarking tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> **Multi-LLM Engineer**: One interface, multiple AI minds. Choose the right tool for each task! 🚀