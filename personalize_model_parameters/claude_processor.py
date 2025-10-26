"""
Claude AI processor for bot personalization
"""
import json
import logging
from typing import Dict, Any
import anthropic
from .config import PersonalizationConfig
from .prompts import BOT_PERSONALIZER_PROMPT, INTERACTIVE_PERSONALIZATION_PROMPT


logger = logging.getLogger(__name__)


class PersonalizationProcessor:
    
    def __init__(self, config: PersonalizationConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.claude_api_key)
        
    def personalize_bot(
        self, 
        default_parameters: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main personalization method that takes default parameters and user preferences 
        and returns personalized parameters.
        """
        try:
            prompt = self._create_personalization_prompt(
                default_parameters, 
                user_preferences
            )
            
            response = self.client.messages.create(
                model=self.config.claude_model,
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            personalized_data = self._parse_claude_response(response.content[0].text)
            
            logger.info("Successfully personalized bot parameters")
            return personalized_data
            
        except Exception as e:
            logger.error(f"Error personalizing bot with Claude: {e}")
            raise
    
    def modify_parameters_interactively(
        self,
        current_parameters: Dict[str, Any],
        user_request: str
    ) -> Dict[str, Any]:
        """
        Interactively modify parameters based on a user's natural language request.
        """
        try:
            prompt = INTERACTIVE_PERSONALIZATION_PROMPT.format(
                current_parameters_json=json.dumps(current_parameters, indent=2),
                user_request=user_request
            )
            
            response = self.client.messages.create(
                model=self.config.claude_model,
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            modifications = self._parse_claude_response(response.content[0].text)
            
            logger.info("Successfully modified parameters based on user request")
            return modifications
            
        except Exception as e:
            logger.error(f"Error modifying parameters interactively: {e}")
            raise
    
    def _create_personalization_prompt(
        self,
        default_parameters: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> str:
        """Create the full personalization prompt with all inputs."""
        return BOT_PERSONALIZER_PROMPT.format(
            default_parameters_json=json.dumps(default_parameters, indent=2),
            user_preferences_json=json.dumps(user_preferences, indent=2)
        )
    
    def _parse_claude_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from Claude's response."""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in Claude response")
            
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"Error parsing Claude response: {e}")
            logger.error(f"Response text: {response_text}")
            raise
