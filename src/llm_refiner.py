# src/llm_refiner.py

import requests
import json
import os

class LLMRefiner:
    """
    A class to refine passages using an LLM.
    
    This version connects to the Gemini API for text refinement.
    """
    def __init__(self, api_key: str):
        print("Initializing LLM Refiner with Gemini API connection.")
        self.api_key = api_key
        # Using a fast, lightweight model for this task.
        self.model_name = "gemini-2.5-flash-preview-05-20"
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

    def refine_passage(self, passage: str) -> str:
        """
        Refines a given passage by fixing minor issues like cut-off sentences
        or repetitions using the Gemini API.
        
        Args:
            passage: The text passage to refine.
            
        Returns:
            The refined passage text or the original passage if refinement fails.
        """
        prompt = (
            "You are a text editor. Correct any cut-off sentences or repeated "
            "phrases in the following text. Do not add any new information. "
            "Your response should be only the corrected text. "
            f"Text to correct: '{passage}'"
        )
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
        }
        
        try:
            response = requests.post(
                self.api_url,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'}
            )
            
            # Check for a successful response
            response.raise_for_status()
            
            # Parse the JSON response
            result = response.json()
            
            if 'candidates' in result and result['candidates']:
                refined_text = result['candidates'][0]['content']['parts'][0]['text']
                return refined_text
            else:
                print("Warning: API response did not contain a valid candidate.")
                return passage
                
        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error: {err}")
            print(f"Response content: {err.response.text}")
            return passage
        except Exception as err:
            print(f"An error occurred during API call: {err}")
            return passage
