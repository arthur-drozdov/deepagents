import asyncio
import json
import logging
import sys
from fastapi import FastAPI, WebSocket
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.config import create_model
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("python_bridge")

# We will instantiate the agent once
agent = None
backend = None
active_vision_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, backend
    try:
        # Create model using CLI logic to respect config.toml (~/.deepagents/config.toml)
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
            system_prompt="You are a helpful and concise voice assistant. If the user asks a vision-related question, you can implicitly refer to the content in VIDEO.md, but NEVER mention the filename 'VIDEO.md' or state that you are reading from a file. Just act as if you are naturally seeing what is described. Do not use markdown.",
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VisionPayload(BaseModel):
    video_base64: str
    format: str

@app.post("/vision")
async def process_vision(payload: VisionPayload):
    global agent, active_vision_task
    if not agent:
        return {"error": "Agent not initialized"}

    try:
        if active_vision_task and not active_vision_task.done():
            logger.info("Canceling previous in-flight vision task")
            active_vision_task.cancel()

        logger.info(f"Received image snapshot (format: {payload.format}), processing with Vision Agent...")

        prompt = "Here is a recent snapshot from the user's camera. Save what you see in the image to the file VIDEO.md in the current directory. Only describe what is visibly present."

        image_msg = {
            "type": "image_url",
            "image_url": {"url": f"data:image/{payload.format};base64,{payload.video_base64}"}
        }

        input_state = {
            "messages": [
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    image_msg
                ])
            ]
        }

        # A separate thread id ensures it doesn't pollute the spoken voice memory buffer
        config = {"configurable": {"thread_id": "vision_background_session"}}

        async def run_vision():
            logger.info("Starting background vision task...")
            try:
                # We consume the generator to actually run it
                async for event in agent.astream(input_state, config=config, stream_mode="messages"):
                    pass
                logger.info("Finished background vision task and wrote VIDEO.md")
            except asyncio.CancelledError:
                logger.info("Vision task was cancelled")
            except Exception as e:
                logger.error(f"Error in background vision task: {e}")

        active_vision_task = asyncio.create_task(run_vision())

        return {"status": "processing"}
    except Exception as e:
        logger.error(f"Vision endpoint error: {e}")
        return {"error": str(e)}

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
                try:
                    await websocket.send_json({"chunk": cleaned_text})
                except Exception as e:
                    logger.error(f"WebSocket send failed (client likely disconnected), aborting generation: {e}")
                    break

            logger.info("Agent response complete")
            # Indicate generation is complete for this turn
            try:
                await websocket.send_json({"done": True})
            except:
                pass

    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
    finally:
        logger.info("Client disconnected from Python bridge")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
