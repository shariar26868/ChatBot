from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from ai_bot import get_response_from_ai_agent
from langchain_core.messages import HumanMessage

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

ALLOWED_MODEL_NAMES = [
    "llama3-70b-8192",        
    "mixtral-8x7b-32768",      
    "llama-3.3-70b-versatile", 
    "gpt-4o-mini",             
    "gpt-3.5-turbo"
]

app = FastAPI(title="LangGraph AI Agent")

@app.post("/chat")
def chat_endpoint(request: RequestState): 
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}
    
    llm_id = request.model_name
    query = [HumanMessage(content=msg) for msg in request.messages]
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    response = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)