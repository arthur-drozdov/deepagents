import asyncio
import json
import logging
import sys
from fastapi import FastAPI, WebSocket
import uvicorn
from contextlib import asynccontextmanager

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.config import create_model
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("python_bridge")

# We will instantiate the agent once
agent = None
backend = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, backend
    # Create model using CLI logic to respect config.toml (~/.deepagents/config.toml)
    try:
        # User explicitly wants to use their config.toml settings
        # create_model() with no args loads the default/recent model from config
        result = create_model(
            extra_kwargs={
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            }
        )
        result.apply_to_settings()
        logger.info(f"Loaded model: {result.model_name} from provider: {result.provider}")

        agent, backend = create_cli_agent(
            model=result.model,
            assistant_id="voice_vision_app",
            system_prompt="You are a helpful and concise voice assistant. Do not use markdown.",
            enable_memory=True,
            enable_skills=True,
            enable_shell=False, # Disable shell for safety in this bridge
            auto_approve=True,   # Auto-approve for voice turns
        )
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise
    yield
    # Cleanup if needed

app = FastAPI(lifespan=lifespan)

@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to Python bridge")
    try:
        while True:
            # We expect JSON payloads looking like {"text": "Hello"}
            data = await websocket.receive_text()
            payload = json.loads(data)
            user_text = payload.get("text", "")

            if not user_text:
                continue

            logger.info(f"User: {user_text}")

            # Stream the response from the agent
            # We configure max_concurrency or similar if needed, but astream works
            input_state = {"messages": [HumanMessage(content=user_text)]}
            config = {"configurable": {"thread_id": "voice_vision_session"}}

            full_response_chunks = []
            async for event in agent.astream(input_state, config=config, stream_mode="messages"):
                try:
                    msg, metadata = event

                    # Logic to skip 'thinking' or 'reasoning' chunks
                    if hasattr(msg, "additional_kwargs"):
                        if msg.additional_kwargs.get("reasoning_content") or msg.additional_kwargs.get("thought"):
                            continue

                    if hasattr(msg, "content") and msg.content:
                        content = msg.content
                        if content.strip():
                             full_response_chunks.append(content)
                except Exception as e:
                    logger.error(f"Error during astream: {e}")

            full_text = "".join(full_response_chunks).strip()

            # Aggressive Reasoning Strip:
            # If the model leaks reasoning into content, it often looks like:
            # "The user sent... I should respond... Hello! ..."
            # We look for common greeting transitions or the first occurrence of a clear greeting
            # if the text starts with "The user" or "This is".

            cleaned_text = full_text
            if full_text.startswith(("The user", "This is", "I should")):
                # Try to find the actual response starting after the first double newline or a greeting
                import re
                # Common greetings for Qwen/concise assistants
                greetings = ["Hello", "Hi", "Greetings", "How can I help"]
                found_greeting = False
                for g in greetings:
                    match = re.search(rf"\b{g}\b", full_text, re.IGNORECASE)
                    if match:
                        cleaned_text = full_text[match.start():].strip()
                        found_greeting = True
                        break

                if not found_greeting and "\n\n" in full_text:
                    # Fallback to double newline
                    cleaned_text = full_text.split("\n\n")[-1].strip()

            if cleaned_text:
                logger.info(f"Final Agent Response (Cleaned): {cleaned_text}")
                await websocket.send_json({"chunk": cleaned_text})

            logger.info("Agent response complete")
            # Indicate generation is complete for this turn
            await websocket.send_json({"done": True})

    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
    finally:
        logger.info("Client disconnected from Python bridge")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
