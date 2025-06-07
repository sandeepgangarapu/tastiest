import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from openai import OpenAI
from supabase import create_client, Client

# 1. Configuration using Pydantic's BaseSettings
# This automatically reads environment variables from a .env file
class Settings(BaseSettings):
    OPENROUTER_API_KEY: str
    SUPABASE_URL: str = "https://lknezacnsyzjtzusinbl.supabase.co"
    SUPABASE_ANON_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxrbmV6YWNuc3l6anR6dXNpbmJsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkyNTM0NjQsImV4cCI6MjA2NDgyOTQ2NH0.JbrRtMHpNJUuV0Nw-C-U5jlV9VAPhJSPuoLDEEqUUV0"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# 2. Initialize the OpenAI client to point to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
)

# 3. Initialize Supabase client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)

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

# 8. Define the API endpoint
@app.post("/check-item", response_model=TSAResponse)
async def check_item(request: ItemRequest):
    """
    Accepts an item name and returns its TSA carry-on and checked bag status.
    Also stores the result in Supabase database.
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
            result = supabase.table("tsa_checks").insert({
                "item_name": request.item_name,
                "carry_on": data["carry_on"],
                "checked_bag": data["checked_bag"],
                "description": data["description"]
            }).execute()
            
            print(f"Successfully stored TSA check for '{request.item_name}' in Supabase")
            
        except Exception as supabase_error:
            print(f"Failed to store in Supabase: {str(supabase_error)}")
            # Continue execution even if Supabase storage fails
            pass

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

@app.get("/")
def read_root():
    return {"message": "Welcome to the TSA Item Checker API. Go to /docs for more info."}

@app.get("/history")
async def get_history():
    """
    Get the history of all TSA checks from Supabase.
    """
    try:
        result = supabase.table("tsa_checks").select("*").order("created_at", desc=True).execute()
        return {"history": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")
