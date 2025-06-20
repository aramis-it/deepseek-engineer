#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.style import Style
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
import time

# Initialize Rich console and prompt session
console = Console()
prompt_session = PromptSession(
    style=PromptStyle.from_dict({
        'prompt': '#0066ff bold',  # Bright blue prompt
        'completion-menu.completion': 'bg:#1e3a8a fg:#ffffff',
        'completion-menu.completion.current': 'bg:#3b82f6 fg:#ffffff bold',
    })
)

# --------------------------------------------------------------------------------
# 1. LLM Provider Configuration
# --------------------------------------------------------------------------------
load_dotenv()  # Load environment variables from .env file

class LLMProvider:
    """Base class for LLM providers"""
    def __init__(self, name: str, client: Any, model: str, supports_tools: bool = True, supports_reasoning: bool = False):
        self.name = name
        self.client = client
        self.model = model
        self.supports_tools = supports_tools
        self.supports_reasoning = supports_reasoning

# Available LLM providers
PROVIDERS = {}

# DeepSeek (default)
if os.getenv("DEEPSEEK_API_KEY"):
    PROVIDERS["deepseek"] = LLMProvider(
        name="DeepSeek",
        client=OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"),
        model="deepseek-reasoner",
        supports_tools=True,
        supports_reasoning=True
    )

# OpenAI
if os.getenv("OPENAI_API_KEY"):
    PROVIDERS["openai"] = LLMProvider(
        name="OpenAI",
        client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        model="gpt-4o",
        supports_tools=True,
        supports_reasoning=False
    )

# Anthropic Claude (via OpenAI-compatible API)
if os.getenv("ANTHROPIC_API_KEY"):
    try:
        import anthropic
        PROVIDERS["claude"] = LLMProvider(
            name="Claude",
            client=anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
            model="claude-3-5-sonnet-20241022",
            supports_tools=True,
            supports_reasoning=False
        )
    except ImportError:
        console.print("[yellow]Warning: anthropic package not installed. Claude support disabled.[/yellow]")

# Google Gemini
if os.getenv("GOOGLE_API_KEY"):
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        PROVIDERS["gemini"] = LLMProvider(
            name="Gemini",
            client=genai.GenerativeModel('gemini-1.5-pro'),
            model="gemini-1.5-pro",
            supports_tools=True,
            supports_reasoning=False
        )
    except ImportError:
        console.print("[yellow]Warning: google-generativeai package not installed. Gemini support disabled.[/yellow]")

# Ollama (local)
if os.getenv("OLLAMA_BASE_URL") or os.path.exists("/usr/local/bin/ollama"):
    PROVIDERS["ollama"] = LLMProvider(
        name="Ollama",
        client=OpenAI(
            api_key="ollama",  # Ollama doesn't require API key
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        ),
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        supports_tools=True,
        supports_reasoning=False
    )

# Select current provider
current_provider_key = os.getenv("LLM_PROVIDER", "deepseek")
if current_provider_key not in PROVIDERS:
    # Fallback to first available provider
    if PROVIDERS:
        current_provider_key = list(PROVIDERS.keys())[0]
    else:
        console.print("[bold red]❌ No LLM providers configured! Please set up API keys in .env file.[/bold red]")
        sys.exit(1)

current_provider = PROVIDERS[current_provider_key]
client = current_provider.client

# --------------------------------------------------------------------------------
# 2. Define our schema using Pydantic for type safety
# --------------------------------------------------------------------------------
class FileToCreate(BaseModel):
    path: str
    content: str

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str

# --------------------------------------------------------------------------------
# 2.1. Define Function Calling Tools
# --------------------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a single file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read (relative or absolute)",
                    }
                },
                "required": ["file_path"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_multiple_files",
            "description": "Read the content of multiple files from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of file paths to read (relative or absolute)",
                    }
                },
                "required": ["file_paths"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file or overwrite an existing file with the provided content",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path where the file should be created",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    }
                },
                "required": ["file_path", "content"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_multiple_files",
            "description": "Create multiple files at once",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["path", "content"]
                        },
                        "description": "Array of files to create with their paths and content",
                    }
                },
                "required": ["files"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit an existing file by replacing a specific snippet with new content",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to edit",
                    },
                    "original_snippet": {
                        "type": "string",
                        "description": "The exact text snippet to find and replace",
                    },
                    "new_snippet": {
                        "type": "string",
                        "description": "The new text to replace the original snippet with",
                    }
                },
                "required": ["file_path", "original_snippet", "new_snippet"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "switch_provider",
            "description": "Switch to a different LLM provider",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "The provider to switch to",
                        "enum": list(PROVIDERS.keys())
                    }
                },
                "required": ["provider"]
            },
        }
    }
]

# --------------------------------------------------------------------------------
# 3. system prompt
# --------------------------------------------------------------------------------
system_PROMPT = dedent(f"""\
    You are an elite software engineer called Multi-LLM Engineer with decades of experience across all programming domains.
    Your expertise spans system design, algorithms, testing, and best practices.
    You provide thoughtful, well-structured solutions while explaining your reasoning.
    
    Current LLM Provider: {current_provider.name} ({current_provider.model})
    Available Providers: {', '.join(PROVIDERS.keys())}
    Reasoning Support: {'Yes' if current_provider.supports_reasoning else 'No'}
    Function Calling: {'Yes' if current_provider.supports_tools else 'No'}

    Core capabilities:
    1. Code Analysis & Discussion
       - Analyze code with expert-level insight
       - Explain complex concepts clearly
       - Suggest optimizations and best practices
       - Debug issues with precision

    2. File Operations (via function calls):
       - read_file: Read a single file's content
       - read_multiple_files: Read multiple files at once
       - create_file: Create or overwrite a single file
       - create_multiple_files: Create multiple files at once
       - edit_file: Make precise edits to existing files using snippet replacement
       - switch_provider: Switch between different LLM providers

    Guidelines:
    1. Provide natural, conversational responses explaining your reasoning
    2. Use function calls when you need to read or modify files
    3. For file operations:
       - Always read files first before editing them to understand the context
       - Use precise snippet matching for edits
       - Explain what changes you're making and why
       - Consider the impact of changes on the overall codebase
    4. Follow language-specific best practices
    5. Suggest tests or validation steps when appropriate
    6. Be thorough in your analysis and recommendations

    IMPORTANT: In your thinking process, if you realize that something requires a tool call, cut your thinking short and proceed directly to the tool call. Don't overthink - act efficiently when file operations are needed.

    Remember: You're a senior engineer - be thoughtful, precise, and explain your reasoning clearly.
""")

# --------------------------------------------------------------------------------
# 4. Helper functions 
# --------------------------------------------------------------------------------

def read_local_file(file_path: str) -> str:
    """Return the text content of a local file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_file(path: str, content: str):
    """Create (or overwrite) a file at 'path' with the given 'content'."""
    file_path = Path(path)
    
    # Security checks
    if any(part.startswith('~') for part in file_path.parts):
        raise ValueError("Home directory references not allowed")
    normalized_path = normalize_path(str(file_path))
    
    # Validate reasonable file size for operations
    if len(content) > 5_000_000:  # 5MB limit
        raise ValueError("File content exceeds 5MB size limit")
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(f"[bold blue]✓[/bold blue] Created/updated file at '[bright_cyan]{file_path}[/bright_cyan]'")

def normalize_path(path_str: str) -> str:
    """Return a canonical, absolute version of the path with security checks."""
    path = Path(path_str).resolve()
    
    # Prevent directory traversal attacks
    if ".." in path.parts:
        raise ValueError(f"Invalid path: {path_str} contains parent directory references")
    
    return str(path)

def switch_llm_provider(provider_key: str) -> bool:
    """Switch to a different LLM provider"""
    global current_provider, current_provider_key, client, system_PROMPT
    
    if provider_key not in PROVIDERS:
        console.print(f"[bold red]❌ Provider '{provider_key}' not available. Available: {', '.join(PROVIDERS.keys())}[/bold red]")
        return False
    
    current_provider_key = provider_key
    current_provider = PROVIDERS[provider_key]
    client = current_provider.client
    
    # Update system prompt with new provider info
    system_PROMPT = dedent(f"""\
        You are an elite software engineer called Multi-LLM Engineer with decades of experience across all programming domains.
        Your expertise spans system design, algorithms, testing, and best practices.
        You provide thoughtful, well-structured solutions while explaining your reasoning.
        
        Current LLM Provider: {current_provider.name} ({current_provider.model})
        Available Providers: {', '.join(PROVIDERS.keys())}
        Reasoning Support: {'Yes' if current_provider.supports_reasoning else 'No'}
        Function Calling: {'Yes' if current_provider.supports_tools else 'No'}

        Core capabilities:
        1. Code Analysis & Discussion
           - Analyze code with expert-level insight
           - Explain complex concepts clearly
           - Suggest optimizations and best practices
           - Debug issues with precision

        2. File Operations (via function calls):
           - read_file: Read a single file's content
           - read_multiple_files: Read multiple files at once
           - create_file: Create or overwrite a single file
           - create_multiple_files: Create multiple files at once
           - edit_file: Make precise edits to existing files using snippet replacement
           - switch_provider: Switch between different LLM providers

        Guidelines:
        1. Provide natural, conversational responses explaining your reasoning
        2. Use function calls when you need to read or modify files
        3. For file operations:
           - Always read files first before editing them to understand the context
           - Use precise snippet matching for edits
           - Explain what changes you're making and why
           - Consider the impact of changes on the overall codebase
        4. Follow language-specific best practices
        5. Suggest tests or validation steps when appropriate
        6. Be thorough in your analysis and recommendations

        IMPORTANT: In your thinking process, if you realize that something requires a tool call, cut your thinking short and proceed directly to the tool call. Don't overthink - act efficiently when file operations are needed.

        Remember: You're a senior engineer - be thoughtful, precise, and explain your reasoning clearly.
    """)
    
    # Update conversation history with new system prompt
    conversation_history[0] = {"role": "system", "content": system_PROMPT}
    
    console.print(f"[bold green]✓[/bold green] Switched to {current_provider.name} ({current_provider.model})")
    return True

# --------------------------------------------------------------------------------
# 5. Function execution
# --------------------------------------------------------------------------------

def execute_function_call_dict(tool_call_dict) -> str:
    """Execute a function call from a dictionary format and return the result as a string."""
    try:
        function_name = tool_call_dict["function"]["name"]
        arguments = json.loads(tool_call_dict["function"]["arguments"])
        
        if function_name == "read_file":
            file_path = arguments["file_path"]
            normalized_path = normalize_path(file_path)
            content = read_local_file(normalized_path)
            return f"Content of file '{normalized_path}':\n\n{content}"
            
        elif function_name == "read_multiple_files":
            file_paths = arguments["file_paths"]
            results = []
            for file_path in file_paths:
                try:
                    normalized_path = normalize_path(file_path)
                    content = read_local_file(normalized_path)
                    results.append(f"Content of file '{normalized_path}':\n\n{content}")
                except OSError as e:
                    results.append(f"Error reading '{file_path}': {e}")
            return "\n\n" + "="*50 + "\n\n".join(results)
            
        elif function_name == "create_file":
            file_path = arguments["file_path"]
            content = arguments["content"]
            create_file(file_path, content)
            return f"Successfully created file '{file_path}'"
            
        elif function_name == "create_multiple_files":
            files = arguments["files"]
            created_files = []
            for file_info in files:
                create_file(file_info["path"], file_info["content"])
                created_files.append(file_info["path"])
            return f"Successfully created {len(created_files)} files: {', '.join(created_files)}"
            
        elif function_name == "edit_file":
            file_path = arguments["file_path"]
            original_snippet = arguments["original_snippet"]
            new_snippet = arguments["new_snippet"]
            
            try:
                content = read_local_file(file_path)
                updated_content = content.replace(original_snippet, new_snippet, 1)
                create_file(file_path, updated_content)
                return f"Successfully edited file '{file_path}'"
            except FileNotFoundError:
                return f"Error: File '{file_path}' not found"
            except Exception as e:
                return f"Error editing file '{file_path}': {str(e)}"
                
        elif function_name == "switch_provider":
            provider = arguments["provider"]
            success = switch_llm_provider(provider)
            return f"Successfully switched to {provider}" if success else f"Failed to switch to {provider}"
            
        else:
            return f"Unknown function: {function_name}"
            
    except Exception as e:
        return f"Error executing {function_name}: {str(e)}"

# --------------------------------------------------------------------------------
# 6. Conversation state
# --------------------------------------------------------------------------------
conversation_history = [
    {"role": "system", "content": system_PROMPT}
]

# --------------------------------------------------------------------------------
# 7. Universal streaming function
# --------------------------------------------------------------------------------

def stream_llm_response(user_message: str):
    """Universal streaming function that works with all LLM providers"""
    # Add the user message to conversation history
    conversation_history.append({"role": "user", "content": user_message})
    
    try:
        # Handle different providers
        if current_provider_key == "claude":
            return stream_claude_response()
        elif current_provider_key == "gemini":
            return stream_gemini_response()
        else:
            # OpenAI-compatible providers (DeepSeek, OpenAI, Ollama)
            return stream_openai_compatible_response()
    except Exception as e:
        error_msg = f"{current_provider.name} API error: {str(e)}"
        console.print(f"\n[bold red]❌ {error_msg}[/bold red]")
        return {"error": error_msg}

def stream_openai_compatible_response():
    """Stream response for OpenAI-compatible APIs (DeepSeek, OpenAI, Ollama)"""
    stream_params = {
        "model": current_provider.model,
        "messages": conversation_history,
        "stream": True
    }
    
    # Add tools if supported
    if current_provider.supports_tools:
        stream_params["tools"] = tools
        
    # Add max tokens for providers that support it
    if current_provider_key in ["deepseek", "openai"]:
        stream_params["max_completion_tokens"] = 64000

    stream = current_provider.client.chat.completions.create(**stream_params)

    console.print(f"\n[bold bright_blue]🤖 {current_provider.name} Processing...[/bold bright_blue]")
    reasoning_started = False
    final_content = ""
    tool_calls = []

    for chunk in stream:
        # Handle reasoning content if available (DeepSeek specific)
        if (current_provider.supports_reasoning and 
            hasattr(chunk.choices[0].delta, 'reasoning_content') and 
            chunk.choices[0].delta.reasoning_content):
            if not reasoning_started:
                console.print("\n[bold blue]💭 Reasoning:[/bold blue]")
                reasoning_started = True
            console.print(chunk.choices[0].delta.reasoning_content, end="")
        elif chunk.choices[0].delta.content:
            if reasoning_started:
                console.print("\n")  # Add spacing after reasoning
                console.print(f"\n[bold bright_blue]🤖 {current_provider.name}>[/bold bright_blue] ", end="")
                reasoning_started = False
            final_content += chunk.choices[0].delta.content
            console.print(chunk.choices[0].delta.content, end="")
        elif chunk.choices[0].delta.tool_calls and current_provider.supports_tools:
            # Handle tool calls
            for tool_call_delta in chunk.choices[0].delta.tool_calls:
                if tool_call_delta.index is not None:
                    # Ensure we have enough tool_calls
                    while len(tool_calls) <= tool_call_delta.index:
                        tool_calls.append({
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    
                    if tool_call_delta.id:
                        tool_calls[tool_call_delta.index]["id"] = tool_call_delta.id
                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            tool_calls[tool_call_delta.index]["function"]["name"] += tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            tool_calls[tool_call_delta.index]["function"]["arguments"] += tool_call_delta.function.arguments

    console.print()  # New line after streaming
    return handle_response_completion(final_content, tool_calls)

def stream_claude_response():
    """Stream response for Anthropic Claude"""
    try:
        # Convert OpenAI format to Claude format
        claude_messages = []
        system_message = conversation_history[0]["content"]
        
        for msg in conversation_history[1:]:  # Skip system message
            if msg["role"] == "user":
                claude_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                claude_messages.append({"role": "assistant", "content": msg["content"] or ""})
            elif msg["role"] == "tool":
                # Claude handles tool results differently
                continue
        
        # Claude tools format
        claude_tools = []
        if current_provider.supports_tools:
            for tool in tools:
                if tool["function"]["name"] != "switch_provider":  # Skip provider switching for Claude
                    claude_tools.append({
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "input_schema": tool["function"]["parameters"]
                    })
        
        stream_params = {
            "model": current_provider.model,
            "max_tokens": 4096,
            "messages": claude_messages,
            "system": system_message,
            "stream": True
        }
        
        if claude_tools:
            stream_params["tools"] = claude_tools
            
        console.print(f"\n[bold bright_blue]🤖 {current_provider.name} Processing...[/bold bright_blue]")
        
        stream = current_provider.client.messages.create(**stream_params)
        
        final_content = ""
        tool_calls = []
        
        for chunk in stream:
            if chunk.type == "content_block_start":
                if chunk.content_block.type == "text":
                    continue
                elif chunk.content_block.type == "tool_use":
                    # Start of tool use block
                    tool_calls.append({
                        "id": chunk.content_block.id,
                        "type": "function",
                        "function": {
                            "name": chunk.content_block.name,
                            "arguments": json.dumps(chunk.content_block.input)
                        }
                    })
            elif chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    final_content += chunk.delta.text
                    console.print(chunk.delta.text, end="")
        
        console.print()
        return handle_response_completion(final_content, tool_calls)
        
    except Exception as e:
        console.print(f"[red]Claude API error: {e}[/red]")
        return {"error": str(e)}

def stream_gemini_response():
    """Stream response for Google Gemini"""
    try:
        # Convert conversation to Gemini format
        gemini_history = []
        system_instruction = conversation_history[0]["content"]
        
        for msg in conversation_history[1:]:  # Skip system message
            if msg["role"] == "user":
                gemini_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant" and msg["content"]:
                gemini_history.append({"role": "model", "parts": [msg["content"]]})
            elif msg["role"] == "tool":
                # Skip tool messages for now - Gemini handles them differently
                continue
        
        console.print(f"\n[bold bright_blue]🤖 {current_provider.name} Processing...[/bold bright_blue]")
        
        # Configure the model with system instruction
        model = current_provider.client
        
        # Start chat with history
        chat = model.start_chat(
            history=gemini_history[:-1] if len(gemini_history) > 1 else []
        )
        
        # Get the last user message
        last_message = gemini_history[-1]["parts"][0] if gemini_history else ""
        
        # Send message and get streaming response
        response = chat.send_message(last_message, stream=True)
        
        final_content = ""
        for chunk in response:
            if chunk.text:
                final_content += chunk.text
                console.print(chunk.text, end="")
        
        console.print()
        
        # Note: Gemini function calling integration would need more work
        # For now, return without tool calls
        return handle_response_completion(final_content, [])
        
    except Exception as e:
        console.print(f"[red]Gemini API error: {e}[/red]")
        return {"error": str(e)}

def handle_response_completion(final_content: str, tool_calls: list):
    """Handle the completion of a response, including tool calls"""
    # Store the assistant's response in conversation history
    assistant_message = {
        "role": "assistant",
        "content": final_content if final_content else None
    }
    
    if tool_calls and current_provider.supports_tools:
        # Convert our tool_calls format to the expected format
        formatted_tool_calls = []
        for i, tc in enumerate(tool_calls):
            if tc["function"]["name"]:  # Only add if we have a function name
                # Ensure we have a valid tool call ID
                tool_id = tc["id"] if tc["id"] else f"call_{i}_{int(time.time() * 1000)}"
                
                formatted_tool_calls.append({
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                })
        
        if formatted_tool_calls:
            # Important: When there are tool calls, content should be None or empty
            if not final_content:
                assistant_message["content"] = None
                
            assistant_message["tool_calls"] = formatted_tool_calls
            conversation_history.append(assistant_message)
            
            # Execute tool calls and add results immediately
            console.print(f"\n[bold bright_cyan]⚡ Executing {len(formatted_tool_calls)} function call(s)...[/bold bright_cyan]")
            for tool_call in formatted_tool_calls:
                console.print(f"[bright_blue]→ {tool_call['function']['name']}[/bright_blue]")
                
                try:
                    result = execute_function_call_dict(tool_call)
                    
                    # Add tool result to conversation immediately
                    tool_response = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    }
                    conversation_history.append(tool_response)
                except Exception as e:
                    console.print(f"[red]Error executing {tool_call['function']['name']}: {e}[/red]")
                    # Still need to add a tool response even on error
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": f"Error: {str(e)}"
                    })
            
            # Get follow-up response after tool execution
            console.print(f"\n[bold bright_blue]🔄 {current_provider.name} processing results...[/bold bright_blue]")
            
            # Get follow-up response
            return stream_follow_up_response()
    else:
        # No tool calls, just store the regular response
        conversation_history.append(assistant_message)

    return {"success": True}

def stream_follow_up_response():
    """Get follow-up response after tool execution"""
    try:
        stream_params = {
            "model": current_provider.model,
            "messages": conversation_history,
            "stream": True
        }
        
        if current_provider.supports_tools:
            stream_params["tools"] = tools
            
        if current_provider_key in ["deepseek", "openai"]:
            stream_params["max_completion_tokens"] = 64000

        follow_up_stream = current_provider.client.chat.completions.create(**stream_params)
        
        follow_up_content = ""
        reasoning_started = False
        
        for chunk in follow_up_stream:
            # Handle reasoning content if available
            if (current_provider.supports_reasoning and 
                hasattr(chunk.choices[0].delta, 'reasoning_content') and 
                chunk.choices[0].delta.reasoning_content):
                if not reasoning_started:
                    console.print("\n[bold blue]💭 Reasoning:[/bold blue]")
                    reasoning_started = True
                console.print(chunk.choices[0].delta.reasoning_content, end="")
            elif chunk.choices[0].delta.content:
                if reasoning_started:
                    console.print("\n")
                    console.print(f"\n[bold bright_blue]🤖 {current_provider.name}>[/bold bright_blue] ", end="")
                    reasoning_started = False
                follow_up_content += chunk.choices[0].delta.content
                console.print(chunk.choices[0].delta.content, end="")
        
        console.print()
        
        # Store follow-up response
        conversation_history.append({
            "role": "assistant",
            "content": follow_up_content
        })
        
        return {"success": True}
    except Exception as e:
        console.print(f"[red]Follow-up error: {e}[/red]")
        return {"error": str(e)}

# --------------------------------------------------------------------------------
# 8. Main interactive loop
# --------------------------------------------------------------------------------

def main():
    # Create a beautiful gradient-style welcome panel
    welcome_text = f"""[bold bright_blue]🤖 Multi-LLM Engineer[/bold bright_blue] [bright_cyan]with Universal Provider Support[/bright_cyan]
[dim blue]Current Provider: {current_provider.name} ({current_provider.model})[/dim blue]
[dim blue]Available Providers: {', '.join(PROVIDERS.keys())}[/dim blue]"""
    
    console.print(Panel.fit(
        welcome_text,
        border_style="bright_blue",
        padding=(1, 2),
        title="[bold bright_cyan]🚀 AI Code Assistant[/bold bright_cyan]",
        title_align="center"
    ))
    
    # Create an elegant instruction panel
    instructions = f"""[bold bright_blue]🔄 Provider Management:[/bold bright_blue]
  • [bright_cyan]Current: {current_provider.name}[/bright_cyan] - {current_provider.model}
  • [bright_cyan]Available: {', '.join(PROVIDERS.keys())}[/bright_cyan]
  • [bright_cyan]Switch with: "switch to openai" or "use claude"[/bright_cyan]

[bold bright_blue]📁 File Operations:[/bold bright_blue]
  • [bright_cyan]/add path/to/file[/bright_cyan] - Include a single file in conversation
  • [bright_cyan]/add path/to/folder[/bright_cyan] - Include all files in a folder
  • [dim]The AI can automatically read and create files using function calls[/dim]

[bold bright_blue]🎯 Commands:[/bold bright_blue]
  • [bright_cyan]exit[/bright_cyan] or [bright_cyan]quit[/bright_cyan] - End the session
  • Just ask naturally - the AI will handle file operations automatically!"""
    
    console.print(Panel(
        instructions,
        border_style="blue",
        padding=(1, 2),
        title="[bold blue]💡 How to Use[/bold blue]",
        title_align="left"
    ))
    console.print()

    while True:
        try:
            user_input = prompt_session.prompt("🔵 You> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold yellow]👋 Exiting gracefully...[/bold yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            console.print("[bold bright_blue]👋 Goodbye! Happy coding![/bold bright_blue]")
            break

        # Handle provider switching commands
        if user_input.lower().startswith(("switch to ", "use ", "change to ")):
            provider_name = user_input.lower().replace("switch to ", "").replace("use ", "").replace("change to ", "").strip()
            if switch_llm_provider(provider_name):
                continue
            else:
                console.print(f"[yellow]Available providers: {', '.join(PROVIDERS.keys())}[/yellow]")
                continue

        response_data = stream_llm_response(user_input)
        
        if response_data.get("error"):
            console.print(f"[bold red]❌ Error: {response_data['error']}[/bold red]")

    console.print("[bold blue]✨ Session finished. Thank you for using Multi-LLM Engineer![/bold blue]")

if __name__ == "__main__":
    main()