"""
Settings loader for experiments.
Loads configuration from YAML.
"""

import yaml
import os
from typing import Any, Dict

class Settings:
    """Settings object with attribute-style access to configuration."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Settings(value))
            else:
                setattr(self, key, value)
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute access with defaults."""
        return None
    
    def __repr__(self) -> str:
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Settings({', '.join(attrs)})"

# Settings singleton cache
_settings_cache = None

def load_settings(config_file: str = None, force_reload: bool = False) -> Settings:
    global _settings_cache
    
    # Clear cache if force reload requested
    if force_reload:
        clear_settings_cache()
    
    # Return cached settings if available
    if _settings_cache is not None:
        return _settings_cache
    
    if config_file is None:
        # Default to settings.yaml in same directory as this file
        config_dir = os.path.dirname(__file__)
        config_file = os.path.join(config_dir, "settings.yaml")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        print(f"[CONFIG] Loaded settings from {config_file}")
        _settings_cache = Settings(config_dict)  # Cache the settings
        return _settings_cache
        
    except FileNotFoundError:
        print(f"[CONFIG WARNING] Settings file not found: {config_file}")
        print("[CONFIG] Using default settings")
        
        # Fallback default configuration
        default_config = {
            "voting": {
                "promotion_threshold": 5,
                "demotion_threshold": -1,
                "demote_multiplier": 2
            },
            "ai": {
                "model": "gpt-4o-mini",
                "max_tokens": 100,
                "temperature": 0.7,
                "timeout": 30,
                "token_budget": 3000
            },
            "civil_gate": {
                "toxicity_threshold": 0.3,
                "use_openai_moderation": True
            },
            "rate_limiting": {
                "enabled": True,
                "window_seconds": 300,
                "max_messages": 3
            },
            "timing": {
                "message_interval": 3.0,
                "jitter_range": 1.0
            }
        }
        
        _settings_cache = Settings(default_config)  # Cache default settings too
        return _settings_cache
    
    except yaml.YAMLError as e:
        print(f"[CONFIG ERROR] Invalid YAML in {config_file}: {e}")
        raise

def clear_settings_cache():
    """Clear the settings cache to force reload."""
    global _settings_cache
    _settings_cache = None