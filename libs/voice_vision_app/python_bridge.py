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
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("python_bridge")

class CameraVision(BaseTool):
    name: str = "Vision"
    description: str = "Use this tool to see the most recent frame from the user's camera when they ask you what you see. It returns the image data."

    def _run(self) -> list[dict]:
        import os
        import base64
        frame_path = os.path.join(os.getcwd(), "latest_frame.jpg")
        if not os.path.exists(frame_path):
            return [{"type": "text", "text": "The camera frame is not available yet."}]

        try:
            with open(frame_path, "rb") as f:
                image_bytes = f.read()
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            return [
                {"type": "text", "text": "Here is the most recent snapshot from the user's camera:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
        except Exception as e:
            return [{"type": "text", "text": f"Error reading camera frame: {e}"}]

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

        system_prompt = (
            "You are a helpful and concise voice assistant. "
            "You have access to a tool called 'Vision' which returns the latest camera snapshot from the user. "
            "If the user asks a vision-related question (e.g. 'what do you see?'), you MUST call the Vision tool to see the image. "
            "When analyzing the image, describe what is visibly present as if you are naturally seeing it. Do not use markdown."
        )

        agent, backend = create_cli_agent(
            model=result.model,
            assistant_id="voice_vision_app",
            system_prompt=system_prompt,
            tools=[CameraVision()],
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
    try:
        # Save the latest 240p frame directly to disk so the vision tool can read it
        import base64
        import os

        frame_path = os.path.join(os.getcwd(), "latest_frame.jpg")
        image_data = base64.b64decode(payload.video_base64)

        with open(frame_path, "wb") as f:
            f.write(image_data)

        return {"status": "saved"}
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
