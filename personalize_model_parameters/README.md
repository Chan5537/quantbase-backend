# Bot Personalization Module

This module personalizes algorithmic trading bot parameters based on user preferences, while leveraging ML model predictions for 7-day crypto price forecasting.

## Overview

The bot personalization system allows users to customize their trading bot's behavior through:

1. **Initial Configuration**: Set up the bot with personalized parameters based on trading style, risk tolerance, and goals
2. **Interactive Modification**: Continuously refine parameters through natural language requests
3. **AI-Powered Optimization**: Claude AI analyzes user preferences and market conditions to recommend optimal parameter configurations

## Features

- 🎯 **10 Parameter Categories**: Comprehensive control over position sizing, entry/exit strategies, risk management, trading style, and more
- 🤖 **AI-Powered Personalization**: Leverages Claude AI to understand natural language preferences and generate optimized configurations
- 📊 **ML Model Integration**: Works with 7-day ML prediction data to balance user preferences with market intelligence
- 🔄 **Interactive Refinement**: Iteratively modify parameters through conversational interface
- ⚠️ **Risk Management**: Built-in warnings and safety checks to prevent dangerous configurations
- 💾 **Configuration Persistence**: Saves personalized parameters with timestamps for version tracking

## Architecture

```
personalize_model_parameters/
├── __init__.py
├── config.py                    # Configuration management
├── prompts.py                   # Claude AI prompt templates
├── claude_processor.py          # Claude AI integration
├── main.py                      # Interactive CLI application
├── input/
│   ├── default_parameters.json  # Default bot parameters
│   └── model_predictions.json   # ML model predictions (7-day)
└── output/
    └── personalized_parameters_*.json  # Generated configurations
```

## Usage

### Basic Usage

Run the interactive personalization tool:

```bash
cd personalize_model_parameters
python -m personalize_model_parameters.main
```

Or import and use programmatically:

```python
from personalize_model_parameters.config import PersonalizationConfig
from personalize_model_parameters.claude_processor import PersonalizationProcessor

# Initialize
config = PersonalizationConfig()
processor = PersonalizationProcessor(config)

# Personalize bot
result = processor.personalize_bot(
    default_parameters={...},
    model_predictions={...},
    user_preferences={...}
)

# Access modified parameters
personalized_params = result['modified_parameters']
```

### Interactive Flow

1. **Initial Setup**: Enter your trading preferences (e.g., "I want more aggressive trading with higher sensitivity to price movements")
2. **Review Changes**: See which parameters were modified and why
3. **Iterative Refinement**: Make continuous adjustments (e.g., "Make it more conservative", "Increase risk factor to 0.8")
4. **Save Configuration**: Output saved with timestamp

## Parameters

### 1. Window (rolling lookback)

- **Type**: Integer
- **Range**: 5-30 typically (suggested: 10-20)
- **Description**: Number of time periods to look back for calculating statistics
- **Lower values**: More responsive to recent price changes
- **Higher values**: Smoother, less sensitive to recent movements

### 2. K_Sigma (standard deviation multiplier)

- **Type**: Float
- **Range**: 0.5-3.0 typically (suggested: 1.0-2.0)
- **Description**: Multiplier for standard deviation in volatility calculations
- **Lower values**: Tighter bands, more trades, higher sensitivity
- **Higher values**: Wider bands, fewer trades, lower sensitivity

### 3. Risk Factor (risk appetite)

- **Type**: Float
- **Range**: 0.0-1.0 (suggested: 0.3-0.7)
- **Description**: Controls overall risk behavior
- **0.0**: Safe/conservative (smaller positions, more cautious)
- **1.0**: Aggressive (larger positions, more risk-taking)

### 4. Base Trade Size (position sizing)

- **Type**: Float
- **Range**: 0.0001-0.01 typically (suggested: 0.0005-0.002)
- **Description**: Base size of trades in the asset being traded
- **Lower values**: Smaller positions, lower risk
- **Higher values**: Larger positions, higher risk

## Example Interactions

### Initial Personalization

**User Input:**

> "I want more aggressive trading with higher sensitivity. Increase my risk tolerance."

**System Response:**

- Increases `risk_factor` from 0.5 to 0.7
- Increases `base_trade_size` from 0.001 to 0.0015
- Decreases `k_sigma` from 1.2 to 1.0 (more sensitive)
- Decreases `window` from 15 to 12 (more responsive)
- Provides warnings about increased risk

### Interactive Modification

**User Input:**

> "Make it more conservative"

**System Response:**

- Modifies `risk_factor` from 0.7 to 0.4
- Decreases `base_trade_size` from 0.0015 to 0.0008
- Shows reasoning and warnings

**User Input:**

> "Increase my sensitivity to recent price movements"

**System Response:**

- Reduces `window` from 12 to 10 (shorter lookback)
- Reduces `k_sigma` from 1.0 to 0.9 (tighter bands)
- Explains that these changes will generate more trades

## Output Format

The system generates JSON output with:

```json
{
  "modified_parameters": {
    "window": 12,
    "k_sigma": 1.0,
    "risk_factor": 0.7,
    "base_trade_size": 0.0015
  },
  "parameter_changes_summary": [
    {
      "parameter_name": "risk_factor",
      "original_value": 0.5,
      "new_value": 0.7,
      "reasoning": "User requested more aggressive trading"
    }
  ],
  "risk_assessment": {
    "overall_risk_level": "high",
    "risk_factors": [
      "Increased position sizes",
      "More sensitive to price movements"
    ],
    "suitability": "Aggressive traders comfortable with higher risk"
  },
  "expected_performance": {
    "expected_trade_frequency": "many",
    "expected_sensitivity": "high - will react quickly to price movements",
    "position_sizing_characterization": "aggressive"
  },
  "warnings": ["Higher risk of drawdowns with aggressive settings"],
  "recommendations": [
    "Monitor positions closely",
    "Consider reducing risk_factor if experiencing losses"
  ]
}
```

## Environment Variables

Required in `.env`:

```env
CLAUDE_API_KEY=your_claude_api_key_here
CLAUDE_MODEL=claude-sonnet-4-5
```

## Dependencies

- `anthropic`: Claude AI API
- `python-dotenv`: Environment variable management
- Standard library: json, logging, datetime, pathlib

## Future Enhancements

- [ ] Web UI for visual parameter editing
- [ ] Backtesting integration with historical performance
- [ ] A/B testing different parameter configurations
- [ ] Real-time performance monitoring and auto-adjustment
- [ ] Portfolio-level optimization across multiple assets
- [ ] Integration with live trading exchanges

## License

Part of the Algorithmic Crypto Trading Marketplace project.
