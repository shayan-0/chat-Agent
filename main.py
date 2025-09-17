from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from langgraph_agent import HajjUmrahAgent
from config import Config

app = FastAPI(title="Hajj & Umrah Chat Agent", version="1.0.0")

# Initialize the LangGraph agent
try:
    agent = HajjUmrahAgent()
except ValueError as e:
    print(f"Configuration error: {e}")
    print("Please set the GEMINI_API_KEY environment variable")
    agent = None

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for Hajj & Umrah guidance and platform FAQs
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized. Please check configuration.")
    
    try:
        # Get response from the LangGraph agent
        response = await agent.process_message(request.user_id, request.message)
        return ChatResponse(reply=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "hajj-umrah-chat-agent"}

if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
