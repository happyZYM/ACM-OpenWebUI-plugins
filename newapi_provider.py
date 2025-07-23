"""
title: OpenRouter
version: 0.2.0
license: MIT
description: Adds support for OpenRouter, including citations, reasoning tokens, and API call reporting
author: Zhuang Yumin
author_url: https://zhuangyumin.dev
"""

import re
import requests
import json
import time
import tiktoken
from typing import List, Union, Generator, Iterator, Optional, Callable, Any, Awaitable, AsyncGenerator
from pydantic import BaseModel, Field
import asyncio


def _insert_citations(text: str, citations: list[str]) -> str:
    """
    Replace citation markers [n] in text with markdown links to the corresponding citation URLs.

    Args:
        text: The text containing citation markers like [1], [2], etc.
        citations: A list of citation URLs, where index 0 corresponds to [1] in the text

    Returns:
        Text with citation markers replaced with markdown links
    """
    # Define regex pattern for citation markers [n]
    pattern = r"\[(\d+)\]"

    def replace_citation(match_obj):
        # Extract the number from the match
        num = int(match_obj.group(1))

        # Check if there's a corresponding citation URL
        # Citations are 0-indexed in the list, but 1-indexed in the text
        if 1 <= num <= len(citations):
            url = citations[num - 1]
            # Return Markdown link: [url]([n])
            return f"[{match_obj.group(0)}]({url})"
        else:
            # If no corresponding citation, return the original marker
            return match_obj.group(0)

    # Replace all citation markers in the text
    result = re.sub(pattern, replace_citation, text)

    return result


class Pipe:
    class Valves(BaseModel):
        NEWAPI_BASE_URL: str = Field(
            default="https://example.com/v1", description="Your NewAPI base URL"
        )
        NEWAPI_API_KEY: str = Field(
            default="", description="Your OpenRouter API key"
        )
        # INCLUDE_REASONING: bool = Field(
        #     default=True,
        #     description="Request reasoning tokens from models that support it",
        # )
        MODEL_PREFIX: str = Field(
            default="", description="Optional prefix for model names in Open WebUI"
        )
        REPORT_API_URL: str = Field(
            default="",
            description="URL to report API",
        )
        REPORT_API_KEY: str = Field(
            default="",
            description="API key to report API",
        )

    def __init__(self):
        self.type = "manifold"  # Multiple models
        self.valves = self.Valves()

        # Updated pricing dictionary with more OpenAI models (per 1M tokens)
        self.pricing_dict = {
            "gpt-4o": {
                "input": 2.5,
                "output": 10
            },
            "gpt-4.1": {
                "input": 2,
                "output": 8
            }
        }

    def _calculate_tokens_and_cost(self, messages: list, response_text: str, model_id: str) -> dict:
        """Calculate token count and cost using tiktoken"""
        try:
            # Get appropriate encoding for the model
            if "gpt-4" in model_id.lower():
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model_id.lower():
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to gpt-4 encoding for other models
                encoding = tiktoken.encoding_for_model("gpt-4")
            
            # Calculate input tokens from messages
            input_tokens = 0
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    input_tokens += len(encoding.encode(content))
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            input_tokens += len(encoding.encode(item.get("text", "")))
            
            # Add tokens for message formatting (approximate)
            input_tokens += len(messages) * 4  # Approximate overhead per message
            
            # Calculate output tokens
            output_tokens = len(encoding.encode(response_text)) if response_text else 0
            
            # Get pricing for the model
            pricing = self.pricing_dict.get(model_id, {"input": 2.5, "output": 10})  # Default to gpt-4o pricing
            
            # Calculate cost (pricing is per 1M tokens)
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            total_cost = input_cost + output_cost
            
            return {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": total_cost
            }
            
        except Exception as e:
            print(f"Error calculating tokens and cost: {e}")
            # Return fallback values
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,  
                "total_tokens": 0,
                "cost": 0.0
            }

    def pipes(self) -> List[dict]:
        """Fetch available models from OpenRouter API"""
        if not self.valves.NEWAPI_API_KEY:
            return [{"id": "error", "name": "API Key not provided"}]

        try:
            headers = {"Authorization": f"Bearer {self.valves.NEWAPI_API_KEY}"}
            response = requests.get(
                f"{self.valves.NEWAPI_BASE_URL}/models", headers=headers
            )

            if response.status_code != 200:
                return [
                    {
                        "id": "error",
                        "name": f"Error fetching models: {response.status_code}",
                    }
                ]

            models_data = response.json()

            # Extract model information
            models = []
            for model in models_data.get("data", []):
                model_id = model.get("id")
                if model_id and model_id in self.pricing_dict:
                    # Use model name or ID, with optional prefix
                    model_name = model.get("name", model_id)
                    prefix = self.valves.MODEL_PREFIX
                    models.append(
                        {
                            "id": model_id,
                            "name": f"{prefix}{model_name}" if prefix else model_name,
                        }
                    )

            return models or [{"id": "error", "name": "No models found"}]

        except Exception as e:
            print(f"Error fetching models: {e}")
            return [{"id": "error", "name": f"Error: {str(e)}"}]

    async def _report_api_call_direct(self, usage_info: dict, user_email: str, model_id: str, __event_emitter__: Callable[[Any], Awaitable[None]]):
        """Report API call to upstream reporting service using direct usage information"""
        if not self.valves.REPORT_API_URL or not self.valves.REPORT_API_KEY:
            return
        
        try:
            # Extract required fields for reporting from usage info
            timestamp = int(time.time())
            input_tokens = usage_info.get("prompt_tokens", 0)
            output_tokens = usage_info.get("completion_tokens", 0)
            cost_usd = usage_info.get("cost", 0.0)
            
            # Prepare API call record
            api_call_record = {
                "timestamp": timestamp,
                "model_id": model_id,
                "user_email": user_email,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd
            }
            
            # Send to reporting API
            headers = {
                "Authorization": f"Bearer {self.valves.REPORT_API_KEY}",
                "Content-Type": "application/json"
            }
            
            report_url = f"{self.valves.REPORT_API_URL.rstrip('/')}/api/record_api_call"
            
            response = requests.post(
                report_url,
                headers=headers,
                json=api_call_record,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"Successfully reported API call for user {user_email}")
            else:
                print(f"Failed to report API call: {response.status_code}")
            
            info = f"input: {input_tokens} | output: {output_tokens} | cost: {cost_usd:.6f}"
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": info,
                        "done": True,
                    },
                }
            )
        except Exception as e:
            print(f"Error reporting API call: {e}")

    async def pipe(self, body: dict, __user__: dict, __metadata__: dict, __event_emitter__: Callable[[Any], Awaitable[None]]) -> Union[str, AsyncGenerator]:
        """Process the request and handle reasoning tokens if supported"""
        # Clone the body for OpenRouter
        payload = body.copy()

        # Print incoming body for debugging
        print(f"Original request body: {json.dumps(body)[:500]}...")

        # Extract user email and model ID for reporting
        user_email = __user__.get("email", "") if __user__ else ""
        model_id = __metadata__.get("model").get("id", "") if __metadata__ else ""

        # Make sure the model ID is properly extracted from the pipe format
        if "model" in payload and payload["model"] and "." in payload["model"]:
            # Extract the model ID from the format like "openrouter.model-id"
            payload["model"] = payload["model"].split(".", 1)[1]
            print(f"Extracted model ID: {payload['model']}")

        # # Add include_reasoning parameter if enabled
        # if self.valves.INCLUDE_REASONING:
        #     payload["include_reasoning"] = True
        
        # Add usage tracking to get token and cost information directly
        # payload["usage"] = {"include": True}

        # Set up headers
        headers = {
            "Authorization": f"Bearer {self.valves.NEWAPI_API_KEY}",
            "Content-Type": "application/json",
        }

        # Add HTTP-Referer and X-Title if provided
        # These help identify your app on OpenRouter
        if body.get("http_referer"):
            headers["HTTP-Referer"] = body["http_referer"]
        if body.get("x_title"):
            headers["X-Title"] = body["x_title"]

        # Default headers for identifying the app to OpenRouter
        if "HTTP-Referer" not in headers:
            headers["HTTP-Referer"] = "https://openwebui.com/"
        if "X-Title" not in headers:
            headers["X-Title"] = "Open WebUI via Pipe"

        url = f"{self.valves.NEWAPI_BASE_URL}/chat/completions"

        try:
            if body.get("stream", False):
                return self.stream_response(url, headers, payload, user_email, model_id, __event_emitter__)
            else:
                return await self.non_stream_response(url, headers, payload, user_email, model_id, __event_emitter__)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    async def non_stream_response(self, url, headers, payload, user_email, model_id, __event_emitter__: Callable[[Any], Awaitable[None]]):
        """Handle non-streaming responses and wrap reasoning in <think> tags if present"""
        try:
            print(
                f"Sending non-streaming request to OpenRouter: {json.dumps(payload)[:200]}..."
            )
            response = requests.post(url, headers=headers, json=payload, timeout=90)

            if response.status_code != 200:
                error_message = f"HTTP Error {response.status_code}"
                try:
                    error_data = response.json()
                    print(f"Error response: {json.dumps(error_data)}")
                    if "error" in error_data:
                        if (
                            isinstance(error_data["error"], dict)
                            and "message" in error_data["error"]
                        ):
                            error_message += f": {error_data['error']['message']}"
                        else:
                            error_message += f": {error_data['error']}"
                except Exception as e:
                    print(f"Failed to parse error response: {e}")
                    error_message += f": {response.text[:500]}"

                # Log request payload for debugging
                print(f"Request that caused error: {json.dumps(payload)}")
                raise Exception(error_message)

            res = response.json()
            print(f"OpenRouter response keys: {list(res.keys())}")

            # Check if we have choices in the response
            if not res.get("choices") or len(res["choices"]) == 0:
                return ""

            # Extract content and reasoning if present
            choice = res["choices"][0]
            message = choice.get("message", {})

            # Debug output
            print(f"Message keys: {list(message.keys())}")

            content = message.get("content", "")
            reasoning = message.get("reasoning", "")

            print(f"Found reasoning: {bool(reasoning)} ({len(reasoning) if reasoning is not None else 0} chars)")
            print(f"Found content: {bool(content)} ({len(content) if content is not None else 0} chars)")

            # Build final response text
            final_response = ""
            if reasoning and content:
                final_response = f"<think>\n{reasoning}\n</think>\n\n{content}"
            elif reasoning:  # Only reasoning, no content (unusual)
                final_response = f"<think>\n{reasoning}\n</think>\n\n"
            elif content:  # Only content, no reasoning
                final_response = content

            # Calculate usage information using tiktoken
            if user_email and model_id:
                messages = payload.get("messages", [])
                usage_info = self._calculate_tokens_and_cost(messages, final_response, model_id)
                
                try:
                    await self._report_api_call_direct(usage_info, user_email, model_id, __event_emitter__)
                except Exception as e:
                    print(f"Error reporting API call: {e}")
                    return f"Error: {e}"

            return final_response
        except Exception as e:
            print(f"Error in non_stream_response: {e}")
            return f"Error: {e}"

    async def stream_response(self, url, headers, payload, user_email, model_id, __event_emitter__: Callable[[Any], Awaitable[None]]):
        """Stream reasoning tokens in real-time with proper tag management"""
        try:
            response = requests.post(
                url, headers=headers, json=payload, stream=True, timeout=90
            )

            if response.status_code != 200:
                error_message = f"HTTP Error {response.status_code}"
                try:
                    error_data = response.json()
                    error_message += (
                        f": {error_data.get('error', {}).get('message', '')}"
                    )
                except:
                    pass
                raise Exception(error_message)

            # State tracking
            in_reasoning_state = False  # True if we've output the opening <think> tag
            latest_citations = []  # The latest citations list
            accumulated_content = ""  # Accumulate all content for token calculation
            accumulated_reasoning = ""  # Accumulate all reasoning for token calculation

            # Process the response stream
            for line in response.iter_lines():
                if not line:
                    continue

                line_text = line.decode("utf-8")
                if not line_text.startswith("data: "):
                    continue
                elif line_text == "data: [DONE]":
                    # Handle citations at the end
                    if latest_citations:
                        citation_list = [f"1. {l}" for l in latest_citations]
                        citation_list_str = "\n".join(citation_list)
                        yield f"\n\n---\nCitations:\n{citation_list_str}"
                    
                    # Calculate usage information using tiktoken and report
                    if user_email and model_id:
                        messages = payload.get("messages", [])
                        final_response = ""
                        if accumulated_reasoning and accumulated_content:
                            final_response = f"<think>\n{accumulated_reasoning}\n</think>\n\n{accumulated_content}"
                        elif accumulated_reasoning:
                            final_response = f"<think>\n{accumulated_reasoning}\n</think>\n\n"
                        elif accumulated_content:
                            final_response = accumulated_content
                        
                        usage_info = self._calculate_tokens_and_cost(messages, final_response, model_id)
                        
                        try:
                            await self._report_api_call_direct(usage_info, user_email, model_id, __event_emitter__)
                        except Exception as e:
                            print(f"Error reporting API call: {e}")
                            yield f"Error: {e}"
                    
                    # Stop processing after [DONE]
                    break

                try:
                    chunk = json.loads(line_text[6:])

                    if "choices" in chunk and chunk["choices"]:
                        choice = chunk["choices"][0]
                        citations = chunk.get("citations") or []

                        # Update the citation list
                        if citations:
                            latest_citations = citations

                        # Check for reasoning tokens
                        reasoning_text = None
                        if "delta" in choice and "reasoning" in choice["delta"]:
                            reasoning_text = choice["delta"]["reasoning"]
                        elif "message" in choice and "reasoning" in choice["message"]:
                            reasoning_text = choice["message"]["reasoning"]

                        # Check for content tokens
                        content_text = None
                        if "delta" in choice and "content" in choice["delta"]:
                            content_text = choice["delta"]["content"]
                        elif "message" in choice and "content" in choice["message"]:
                            content_text = choice["message"]["content"]

                        # Handle reasoning tokens
                        if reasoning_text:
                            # Accumulate reasoning for token calculation
                            accumulated_reasoning += reasoning_text
                            
                            # If first reasoning token, output opening tag
                            if not in_reasoning_state:
                                yield "<think>\n"
                                in_reasoning_state = True

                            # Output the reasoning token
                            yield _insert_citations(reasoning_text, citations)

                        # Handle content tokens
                        if content_text:
                            # Accumulate content for token calculation
                            accumulated_content += content_text
                            
                            # If transitioning from reasoning to content, close the thinking tag
                            if in_reasoning_state:
                                yield "\n</think>\n\n"
                                in_reasoning_state = False

                            # Output the content
                            if content_text:
                                yield _insert_citations(content_text, citations)

                except Exception as e:
                    print(f"Error processing chunk: {e}")

            # If we're still in reasoning state at the end, close the tag
            if in_reasoning_state:
                yield "\n</think>\n\n"

        except Exception as e:
            print(f"Error in stream_response: {e}")
            yield f"Error: {e}"
