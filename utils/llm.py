import os
import requests
import json
from dotenv import load_dotenv
from typing import Dict, Any, Tuple, List

# Load environment variables
load_dotenv()
api_key = os.environ.get("GROQ_API", "")
model_env = os.environ.get("MODEL", "llama-3.3-70b-versatile")
twin_persona = os.environ.get("TWIN", "You are a digital twin AI engine.")


class SafeLLM:
    def __init__(self, api_key: str = api_key, model_name: str = model_env):
        self.api_key = api_key or os.environ.get("GROQ_API", "")
        self.model_name = model_name or os.environ.get("MODEL", "llama-3.3-70b-versatile")
        self.system_persona = os.environ.get("TWIN", twin_persona)
        self.blocked_terms = ["system_prompt_override", "pII_leak", "secret_key"]
        
        # Internal state to store token usage from the latest completion
        self.last_usage = {"inbound_tokens": 0, "outbound_tokens": 0}

    def sanitize_input(self, user_input: str) -> Tuple[bool, str]:
        """Pre-execution safety check on incoming user queries."""
        for term in self.blocked_terms:
            if term in user_input.lower():
                return False, "Input flagged by safety filters."
        return True, user_input

    def validate_output(self, response_text: str) -> bool:
        """Post-execution safety verification."""
        if "SYSTEM_PROMPT" in response_text or "INTERNAL_API_KEY" in response_text:
            return False
        return True

    def prune_context_window(self, history: List[Dict[str, Any]], max_turns: int = 6) -> List[Dict[str, str]]:
        """Prunes conversation history to max_turns and converts to structured role messages."""
        recent_turns = history[-max_turns:] if len(history) > max_turns else history
        pruned_messages = []
        
        for turn in recent_turns:
            role = turn.get("role")
            content = turn.get("content")
            if role in ["user", "assistant"] and content:
                pruned_messages.append({"role": role, "content": content})
                
        return pruned_messages

    def get_last_token_usage(self) -> Dict[str, int]:
        """Returns the inbound and outbound token counts recorded during the latest chat call."""
        return self.last_usage

    def chat(self, user_input: str, matched_context: str, history: List[Dict[str, Any]]) -> str:
        """Executes LLM request with safety filters, RAG context, and token tracking."""
        # Reset usage tracker for the current turn
        self.last_usage = {"inbound_tokens": 0, "outbound_tokens": 0}

        # 1. Sanitize user input
        is_safe, cleaned_input = self.sanitize_input(user_input)
        if not is_safe:
            return cleaned_input

        # 2. Build system message with background facts
        messages = [
            {
                "role": "system",
                "content": f"{self.system_persona}\n\nBACKGROUND KNOWLEDGE:\n{matched_context}"
            }
        ]

        # 3. Append pruned context window turns
        messages.extend(self.prune_context_window(history))

        # 4. Append current cleaned user query
        messages.append({"role": "user", "content": cleaned_input})

        # 5. Formulate API call payload
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7
        }

        # 6. Execute request with comprehensive error handling
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Capture exact token metrics from API response
                usage_meta = response_data.get("usage", {})
                self.last_usage = {
                    "inbound_tokens": usage_meta.get("prompt_tokens", 0),
                    "outbound_tokens": usage_meta.get("completion_tokens", 0)
                }

                # Extract response message content
                reply = response_data["choices"][0]["message"]["content"]

                # Perform post-execution safety validation
                if not self.validate_output(reply):
                    return "Response redacted due to safety verification failure."

                return reply
            else:
                return f"API Error {response.status_code}: {response.text}"

        except requests.exceptions.Timeout:
            return "Error: Request timed out while connecting to the LLM service."
        except requests.exceptions.RequestException as e:
            return f"Network Error: Unable to complete request ({str(e)})"
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return f"Parsing Error: Unexpected response structure ({str(e)})"

