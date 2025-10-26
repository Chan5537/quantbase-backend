"""
Configuration for bot personalization
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class PersonalizationConfig:
    
    def __init__(self):
        self.claude_api_key: Optional[str] = os.getenv('CLAUDE_API_KEY')
        self.claude_model: str = os.getenv('CLAUDE_MODEL')
        self.input_dir: str = "personalize_model_parameters/input"
        self.output_dir: str = "personalize_model_parameters/output"
        
    def validate(self) -> bool:
        if not self.claude_api_key:
            raise ValueError("CLAUDE_API_KEY environment variable is required")
        return True
