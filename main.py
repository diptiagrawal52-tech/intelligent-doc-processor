import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai import errors

app = FastAPI(
    title="Intelligent Document Processor",
    description="FastAPI backend to parse unformatted documents into structured JSON using Gemini 2.5 Flash.",
    version="1.0.0"
)

# Pydantic models for structured output schema
class LineItem(BaseModel):
    description: str = Field(description="Description of the item or service.")
    quantity: Optional[int] = Field(None, description="Quantity of items purchased.")
    unit_price: Optional[float] = Field(None, description="Price per unit.")
    amount: Optional[float] = Field(None, description="Total amount for this line item.")

class InvoiceData(BaseModel):
    vendor_name: str = Field(description="The name of the vendor/issuer of the document.")
    invoice_date: Optional[str] = Field(None, description="The date of the invoice/manifest, formatted as YYYY-MM-DD if possible.")
    line_items: List[LineItem] = Field(default=[], description="List of line items found in the document.")
    total_amount: float = Field(description="The total amount of the invoice/manifest.")

# Input model for the API request
class ExtractRequest(BaseModel):
    text: str = Field(..., description="The raw, unformatted text of the document (invoice or shipping manifest).")

def get_genai_client() -> genai.Client:
    """
    Initializes and returns the GenAI client using the GEMINI_API_KEY environment variable.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable is not set. Please configure it in your environment."
        )
    return genai.Client(api_key=api_key)

@app.get("/")
async def root():
    """
    Root endpoint to verify the API status.
    """
    return {"status": "online"}

@app.post("/extract", response_model=InvoiceData)
async def extract_document(request: ExtractRequest):
    """
    POST route to parse unformatted document text into a structured JSON payload using Gemini 2.5 Flash.
    """
    client = get_genai_client()
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.text,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InvoiceData,
                system_instruction=(
                    "You are an expert document parsing assistant. "
                    "Analyze the provided raw text (simulating an unformatted invoice or shipping manifest) "
                    "and extract the details into the requested JSON schema. "
                    "Ensure vendor_name is correctly identified. Parse and convert invoice_date into YYYY-MM-DD format if date details are present. "
                    "Extract all line items with description, quantity, unit_price, and amount. "
                    "Extract the total_amount accurately."
                )
            )
        )
        
        if response.parsed:
            return response.parsed
        
        raise HTTPException(
            status_code=502,
            detail="Gemini did not return structured output matching the schema."
        )
        
    except errors.APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during extraction: {str(e)}"
        )
