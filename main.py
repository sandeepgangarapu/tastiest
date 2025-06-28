import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from openai import OpenAI
from supabase import create_client, Client
from typing import List, Optional
from datetime import datetime

# 1. Configuration using Pydantic's BaseSettings
# This automatically reads environment variables from a .env file
class Settings(BaseSettings):
    OPENROUTER_API_KEY: str
    SUPABASE_URL: str = "https://kyudboffpfipzxlkzxam.supabase.co"
    SUPABASE_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt5dWRib2ZmcGZpcHp4bGt6eGFtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkzMTk3OTgsImV4cCI6MjA2NDg5NTc5OH0.BzJlk8x56I5IPg4S78aVlhm2-bWk3UYr4yh3AuM9vXc"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# 2. Initialize the OpenAI client to point to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
)

# 3. Initialize the Supabase client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# 4. Initialize the FastAPI app
app = FastAPI(
    title="TSA Item Checker API",
    description="An API to check if an item is allowed in carry-on or checked baggage.",
    version="1.0.0",
)

# 5. Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# 6. Define the data models for request and response
# This ensures our API has a clear and validated structure.
class ItemRequest(BaseModel):
    item_name: str

class TSAResponse(BaseModel):
    carry_on: bool
    checked_bag: bool
    description: str

class TSACheckHistory(BaseModel):
    id: str
    item_name: str
    carry_on: bool
    checked_bag: bool
    description: str
    created_at: datetime

# 7. The System Prompt for the AI Model
# This is the most important part for getting reliable results.
# We instruct the AI on its role and the exact JSON format to return.
SYSTEM_PROMPT = """
You are an expert assistant specializing in TSA (Transportation Security Administration) regulations.
Your task is to determine if a given item is allowed in carry-on and/or checked baggage on a flight in the USA.
You must respond ONLY with a valid JSON object. Do not add any introductory text, explanations, or markdown formatting.
The JSON object must have the following structure:
{
  "carry_on": boolean,
  "checked_bag": boolean,
  "description": "A brief explanation of the rules and any quantity limits."
}

For example, if the item is "Laptop", your response should be:
{
  "carry_on": true,
  "checked_bag": true,
  "description": "Laptops are allowed in both carry-on and checked bags. It is strongly recommended to keep them in your carry-on."
}
If the item is "Dynamite", your response should be:
{
  "carry_on": false,
  "checked_bag": false,
  "description": "Explosives like dynamite are strictly forbidden in both carry-on and checked baggage."
}
"""

# 8. Define the API endpoints
@app.post("/check-item", response_model=TSAResponse)
async def check_item(request: ItemRequest):
    """
    Accepts an item name and returns its TSA carry-on and checked bag status.
    Also stores the result in Supabase for history tracking.
    """
    try:
        completion = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct", # A good, fast, and cheap model
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.item_name},
            ],
            temperature=0.1, # Lower temperature for more deterministic results
            max_tokens=150,
        )

        response_content = completion.choices[0].message.content
        
        # Parse the JSON string from the AI's response
        data = json.loads(response_content)

        # Store the result in Supabase
        try:
            supabase.table("tsa_checks").insert({
                "item_name": request.item_name,
                "carry_on": data["carry_on"],
                "checked_bag": data["checked_bag"],
                "description": data["description"]
            }).execute()
        except Exception as db_error:
            # Log the database error but don't fail the request
            print(f"Database error: {str(db_error)}")

        # FastAPI will automatically validate this against the TSAResponse model
        return data

    except json.JSONDecodeError:
        # Handle cases where the AI doesn't return valid JSON
        raise HTTPException(
            status_code=500,
            detail="Error: The AI model returned a malformed response.",
        )
    except Exception as e:
        # Handle other potential errors (e.g., API key issue, network error)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/history", response_model=List[TSACheckHistory])
async def get_check_history(limit: Optional[int] = 10):
    """
    Returns the history of TSA checks from the database.
    """
    try:
        result = supabase.table("tsa_checks").select("*").order("created_at", desc=True).limit(limit).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.get("/history/{item_name}")
async def get_item_history(item_name: str):
    """
    Returns the history for a specific item name.
    """
    try:
        result = supabase.table("tsa_checks").select("*").eq("item_name", item_name).order("created_at", desc=True).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving item history: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the TSA Item Checker API. Go to /docs for more info."}
