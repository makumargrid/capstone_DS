import os
from google import genai
from dotenv import load_dotenv

from src.logger import get_agent_logger

# Ensure environment variables are loaded
load_dotenv()
logger = get_agent_logger()

def generate_cad_code(prompt: str, model_name: str = 'gemini-2.5-pro') -> str:
    """Generates CadQuery Python code from a natural language prompt."""
    logger.info("Initializing LLM client...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your .env file.")
        
    client = genai.Client(api_key=api_key)
    
    system_prompt = """
You are an expert CAD engineer and Python developer. Your task is to generate strict CadQuery Python code based on the user's request.

RULES:
You must output ONLY valid Python code. No markdown formatting, no conversational filler, no explanations.
The code must instantiate a CadQuery solid and assign it to a variable named result_solid.
Use standard CadQuery syntax (e.g., import cadquery as cq).
Ensure dimensions are in millimeters.
CRITICAL: Do NOT hallucinate or invent CadQuery selectors (e.g., do not use `LargestAreaSelector` or custom selectors). Stick strictly to standard string selectors like `>Z`, `<Z`, `>X`, `not >Z`, `%Plane`, or use direct geometric operations.
"""
    full_prompt = f"{system_prompt}\nRequest:{prompt}"
    
    logger.info("Sending prompt to LLM. Thinking...")
    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt
    )
    
    # Strip any potential markdown formatting from the response
    code = response.text.replace('```python', '').replace('```', '').strip()
    return code

def extract_expected_dimensions(prompt: str, model_name: str = 'gemini-2.5-pro') -> dict:
    """Uses the LLM to dynamically predict the expected bounding box of the object based on the prompt."""
    logger.info("Dynamically calculating expected dimensions from prompt...")
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    system_prompt = """
You are a geometry analysis agent. Based on the user's CAD prompt, calculate the expected bounding box dimensions (X, Y, Z in millimeters).
Return ONLY a valid JSON object in this exact format, with no markdown formatting:
{
    "x_mm": float,
    "y_mm": float,
    "z_mm": float,
    "tolerance_mm": float
}
Set the tolerance based on how complex/curved the object is (e.g., 1.0 for simple boxes, 5.0-10.0 for complex radial/curved objects).
"""
    full_prompt = f"{system_prompt}\nRequest:{prompt}"
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt
        )
        import json
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        dims = json.loads(clean_text)
        logger.info(f"Dynamically calculated dimensions: {dims}")
        return dims
    except Exception as e:
        logger.error(f"Failed to extract dimensions dynamically: {e}")
        # Fallback dimensions if extraction fails
        return {"x_mm": 100.0, "y_mm": 100.0, "z_mm": 100.0, "tolerance_mm": 10.0}
