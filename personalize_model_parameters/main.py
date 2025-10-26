"""
Main application for bot personalization
Provides interactive CLI to personalize trading bot parameters
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import sys

from .config import PersonalizationConfig
from .claude_processor import PersonalizationProcessor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BotPersonalizationApp:
    
    def __init__(self):
        self.config = PersonalizationConfig()
        self.config.validate()
        self.processor = PersonalizationProcessor(self.config)
        self.current_parameters: Optional[Dict[str, Any]] = None
        
    def load_default_parameters(self) -> Dict[str, Any]:
        """Load default bot parameters from input directory."""
        input_file = Path(self.config.input_dir) / "default_parameters.json"
        
        if not input_file.exists():
            # Return default parameters if file doesn't exist
            logger.warning(f"Default parameters file not found at {input_file}, using defaults")
            return self._get_default_parameters()
        
        with open(input_file, 'r') as f:
            return json.load(f)
    

    
    def save_personalized_parameters(self, data: Dict[str, Any]):
        """Save personalized parameters to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(self.config.output_dir) / f"personalized_parameters_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved personalized parameters to {output_file}")
        return output_file
    
    def personalize_with_preferences(self, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Personalize bot with user preferences."""
        default_params = self.load_default_parameters()
        
        result = self.processor.personalize_bot(
            default_parameters=default_params,
            user_preferences=user_preferences
        )
        
        self.current_parameters = result.get('modified_parameters', result)
        return result
    
    def modify_interactively(self, user_request: str) -> Dict[str, Any]:
        """Modify current parameters based on user request."""
        if not self.current_parameters:
            # Load or use defaults
            self.current_parameters = self.load_default_parameters()
        
        result = self.processor.modify_parameters_interactively(
            current_parameters=self.current_parameters,
            user_request=user_request
        )
        
        self.current_parameters = result.get('modified_parameters', result)
        return result
    
    def run_interactive_mode(self):
        """Run interactive CLI mode."""
        print("\n" + "="*70)
        print("🤖 ALGORITHMIC CRYPTO TRADING BOT PERSONALIZER")
        print("="*70 + "\n")
        
        # Initial personalization
        print("Please provide your trading preferences:")
        print("(You can describe in natural language or we'll collect structured input)\n")
        
        # Get user preferences
        user_input = input("Enter your trading preferences: ").strip()
        
        if not user_input:
            print("\nUsing default preferences...")
            user_preferences = self._get_default_preferences()
        else:
            # Convert natural language to structured preferences
            user_preferences = {
                "user_request": user_input,
                "preferences_description": user_input
            }
        
        print("\n⚙️  Personalizing your bot...")
        
        try:
            result = self.personalize_with_preferences(user_preferences)
            
            # Display summary
            print("\n" + "="*70)
            print("✅ PERSONALIZATION COMPLETE")
            print("="*70)
            
            # Display changes
            if 'parameter_changes_summary' in result:
                print("\n📊 PARAMETER CHANGES:")
                for change in result['parameter_changes_summary'][:5]:  # Show first 5
                    print(f"  • {change['parameter_name']}: {change['original_value']} → {change['new_value']}")
                    print(f"    Reason: {change['reasoning']}\n")
            
            # Display warnings
            if 'warnings' in result and result['warnings']:
                print("\n⚠️  WARNINGS:")
                for warning in result['warnings']:
                    print(f"  • {warning}")
            
            # Save result
            output_file = self.save_personalized_parameters(result)
            print(f"\n💾 Saved to: {output_file}")
            
            # Interactive modification loop
            print("\n" + "="*70)
            print("🔄 INTERACTIVE MODIFICATION MODE")
            print("Type 'exit' to quit, or describe changes you'd like to make")
            print("="*70 + "\n")
            
            while True:
                user_request = input("💬 Your request: ").strip()
                
                if user_request.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_request:
                    continue
                
                print("\n⚙️  Processing your request...")
                try:
                    modifications = self.modify_interactively(user_request)
                    
                    print("\n📊 CHANGES APPLIED:")
                    if 'changes_applied' in modifications:
                        for change in modifications['changes_applied']:
                            print(f"  • {change['parameter']}: {change['old_value']} → {change['new_value']}")
                            print(f"    Reason: {change['reasoning']}\n")
                    
                    if 'warnings' in modifications and modifications['warnings']:
                        print("\n⚠️  WARNINGS:")
                        for warning in modifications['warnings']:
                            print(f"  • {warning}")
                    
                    if 'questions' in modifications and modifications['questions']:
                        print("\n❓ QUESTIONS:")
                        for question in modifications['questions']:
                            print(f"  • {question}")
                    
                    # Save updated version
                    output_file = self.save_personalized_parameters({
                        'modified_parameters': self.current_parameters,
                        'modifications': modifications
                    })
                    print(f"\n💾 Updated parameters saved!")
                    
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    logger.error(f"Error in interactive modification: {e}")
                
                print()
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Error in personalization: {e}")
            sys.exit(1)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Return default bot parameters."""
        return {
            "window": 15,
            "k_sigma": 1.2,
            "risk_factor": 0.5,
            "base_trade_size": 0.001
        }
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Return default user preferences."""
        return {
            "risk_tolerance": "medium",
            "trading_style": "swing_trading",
            "goals": "steady_growth",
            "description": "Medium risk, balanced approach"
        }


def main():
    """Main entry point."""
    app = BotPersonalizationApp()
    app.run_interactive_mode()


if __name__ == "__main__":
    main()
