import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=api_key)

# --- Pydantic Models ---
class PromptRequest(BaseModel):
    prompt: str
    model_name: str = "gemini-2.5-flash-preview-04-17" # Default model

class GeminiResponse(BaseModel):
    response: str

# --- FastAPI App ---
app = FastAPI(
    title="Gemini API Backend",
    description="A simple FastAPI backend to interact with the Google Gemini API.",
    version="0.1.0",
)

# --- CORS Middleware ---
# Define allowed origins (adjust as needed for production)
origins = [
    "http://localhost", # Allow localhost (any port)
    "http://localhost:4200", # Specifically allow Angular default dev server
    "http://127.0.0.1", # Allow loopback (any port)
    "http://127.0.0.1:4200", # Allow loopback with Angular port
    # Add your frontend's production URL here if applicable
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of origins allowed to make requests
    allow_credentials=True, # Allow cookies to be included in requests
    allow_methods=["*"], # Allow all standard methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# --- Helper Function for Gemini Interaction ---
async def get_gemini_model(model_name: str = "gemini-2.5-flash-preview-04-17"):
    """Initializes and returns a Gemini model instance."""
    try:
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        # Handle potential errors during model initialization (e.g., invalid model name)
        raise HTTPException(status_code=500, detail=f"Failed to initialize Gemini model: {e}")

# --- API Endpoint ---
@app.post("/api/generate", response_model=GeminiResponse)
async def generate_text(
    request: PromptRequest,
    model: genai.GenerativeModel = Depends(get_gemini_model) # Inject model via dependency
):
    """
    Receives a prompt and generates text using the specified Gemini model.
    """
    try:
        # Use the injected model instance
        generation_config = genai.types.GenerationConfig(
            # Optional: configure temperature, top_p, etc.
            # temperature=0.7
        )
        safety_settings = [ # Configure safety settings as needed
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
             {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

        response = await model.generate_content_async(
            request.prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Basic error handling for response
        if not response.candidates:
             raise HTTPException(status_code=500, detail="Gemini API did not return any candidates.")
        if response.prompt_feedback.block_reason:
             # If the prompt itself was blocked, return a 400
             raise HTTPException(
                status_code=400,
                detail=f"Prompt blocked due to {response.prompt_feedback.block_reason.name}. Safety Feedback: {response.prompt_feedback.safety_ratings}"
             )

        # Check if generation stopped for reasons other than 'STOP'
        candidate = response.candidates[0]
        # Compare using the .name attribute or directly if it's already a string
        finish_reason_value = getattr(candidate.finish_reason, 'name', candidate.finish_reason) # Get name or value itself

        if finish_reason_value != 'STOP':
            finish_reason_name = finish_reason_value # Already have the name/value
            safety_ratings = candidate.safety_ratings
            prompt_feedback_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback.block_reason else "N/A"

            detail_msg = f"Generation failed or stopped early. Finish Reason: {finish_reason_name}."
            if safety_ratings:
                detail_msg += f" Candidate Safety Ratings: {safety_ratings}."
            # Include prompt feedback details as they might be relevant even if the prompt wasn't fully blocked
            detail_msg += f" Prompt Block Reason: {prompt_feedback_reason}. Prompt Safety Ratings: {response.prompt_feedback.safety_ratings}."

            # Use 400 if stopped due to SAFETY, otherwise 500 for other non-STOP reasons
            status_code = 400 if finish_reason_name == 'SAFETY' else 500

            raise HTTPException(
                status_code=status_code,
                detail=detail_msg
            )

        # Existing check for empty response text (might be redundant now but safe to keep)
        if not response.text:
             # This might happen if the content is blocked by safety settings, but finish_reason was STOP (unlikely but possible)
             # Or if the model genuinely returned nothing despite stopping normally.
             finish_reason = getattr(candidate.finish_reason, 'name', candidate.finish_reason)
             safety_ratings = candidate.safety_ratings
             detail_msg = f"Gemini API returned an empty response despite successful completion. Finish Reason: {finish_reason}."
             if safety_ratings:
                 detail_msg += f" Safety Ratings: {safety_ratings}"
             # Treat empty response as a server-side issue unless explicitly safety-related
             # (Safety blockage resulting in empty text should ideally be caught by finish_reason != STOP check above)
             raise HTTPException(status_code=500, detail=detail_msg)


        return GeminiResponse(response=response.text)

    except genai.types.BlockedPromptException as e:
        # This exception specifically occurs if the prompt is blocked *before* generation attempt
        raise HTTPException(status_code=400, detail=f"Prompt blocked by API before generation: {e}")
    except genai.types.StopCandidateException as e:
         # This might occur for various reasons during generation
         raise HTTPException(status_code=500, detail=f"Generation stopped by API during processing: {e}")
    except Exception as e:
        # Catch-all for other potential errors during generation or setup
        # Log the error server-side for debugging
        print(f"Unhandled exception in /api/generate: {type(e).__name__}: {e}") # Basic server-side logging
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {type(e).__name__}")

# --- Run Server (for local development) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    # Use port 8000 by default, or from environment variable
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 