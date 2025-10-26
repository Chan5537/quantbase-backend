# -*- coding: utf-8 -*-
"""
Test script for bot personalization system
"""
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from personalize_model_parameters.config import PersonalizationConfig
from personalize_model_parameters.claude_processor import PersonalizationProcessor


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        config = PersonalizationConfig()
        config.validate()
        print("✅ Configuration loaded successfully")
        print(f"   Model: {config.claude_model}")
        print(f"   Input dir: {config.input_dir}")
        print(f"   Output dir: {config.output_dir}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_input_files():
    """Test input files exist and are valid JSON."""
    print("\nTesting input files...")
    config = PersonalizationConfig()
    
    input_dir = Path(config.input_dir)
    
    # Check default parameters
    params_file = input_dir / "default_parameters.json"
    if params_file.exists():
        with open(params_file, 'r') as f:
            data = json.load(f)
        print(f"✅ Default parameters loaded ({len(str(data))} bytes)")
    else:
        print(f"⚠️  Default parameters file not found at {params_file}")


def test_processor_init():
    """Test processor initialization."""
    print("\nTesting processor initialization...")
    try:
        config = PersonalizationConfig()
        config.validate()
        processor = PersonalizationProcessor(config)
        print("✅ Processor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Processor initialization error: {e}")
        print("   Note: This requires CLAUDE_API_KEY in .env file")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("BOT PERSONALIZATION SYSTEM - TEST SUITE")
    print("="*70)
    
    # Run tests
    config_ok = test_config()
    test_input_files()
    
    if config_ok:
        processor_ok = test_processor_init()
        
        if processor_ok:
            print("\n" + "="*70)
            print("✅ ALL TESTS PASSED")
            print("="*70)
            print("\nYou can now run the full personalization system with:")
            print("  python -m personalize_model_parameters.main")
        else:
            print("\n" + "="*70)
            print("⚠️  PARTIAL TESTS PASSED")
            print("="*70)
            print("\nConfiguration is valid but Claude API is not initialized.")
            print("Make sure you have CLAUDE_API_KEY set in your .env file.")
    else:
        print("\n" + "="*70)
        print("❌ TESTS FAILED")
        print("="*70)
        print("\nPlease check your configuration.")


if __name__ == "__main__":
    main()
