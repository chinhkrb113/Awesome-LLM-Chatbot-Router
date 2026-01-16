import yaml
import os
import logging
import aiofiles
from pathlib import Path
from typing import Dict, List, Any, Union
from app.core.models import ActionConfig, RuleConfig
from app.core.exceptions import ConfigurationError
from app.core.config import settings, Settings

# Use standard logger initially, can switch to app.core.logger later
logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, 
                 action_catalog_path: Union[str, Path] = None, 
                 rule_config_path: Union[str, Path] = None,
                 system_config_path: Union[str, Path] = None):
        
        # Use settings defaults if not provided
        self.action_catalog_path = Path(action_catalog_path or settings.ACTION_CATALOG_PATH)
        self.rule_config_path = Path(rule_config_path or settings.KEYWORD_RULES_PATH)
        self.system_config_path = Path(system_config_path or settings.SYSTEM_CONFIG_PATH)
        
        self.actions: Dict[str, ActionConfig] = {}
        self.rules: Dict[str, RuleConfig] = {}
        
        # Use Pydantic Settings for system config
        self.settings: Settings = settings

    async def load(self):
        """Load all configurations with error handling"""
        try:
            await self._load_actions()
            await self._load_rules()
            await self._load_system_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration load failed: {str(e)}")

    async def _load_actions(self):
        if not self.action_catalog_path.exists():
             logger.warning(f"Action catalog not found at {self.action_catalog_path}")
             return

        async with aiofiles.open(self.action_catalog_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            catalog_data = yaml.safe_load(content)
            if catalog_data and 'action_catalog' in catalog_data:
                self.actions = {} # Reset
                for item in catalog_data['action_catalog']:
                    action = ActionConfig(**item)
                    self.actions[action.action_id] = action

    async def _load_rules(self):
        if not self.rule_config_path.exists():
             logger.warning(f"Rule config not found at {self.rule_config_path}")
             return

        async with aiofiles.open(self.rule_config_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            rule_data = yaml.safe_load(content)
            if rule_data and 'keyword_rules' in rule_data:
                self.rules = {} # Reset
                for action_id, rules in rule_data['keyword_rules'].items():
                    self.rules[action_id] = RuleConfig(**rules)

    async def _load_system_config(self):
        if not self.system_config_path.exists():
            # Already initialized with defaults in settings
            logger.info(f"System config not found at {self.system_config_path}. Using environment/defaults.")
            return

        async with aiofiles.open(self.system_config_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            yaml_config = yaml.safe_load(content) or {}
            
            # Merge YAML config into Pydantic Settings
            # Note: This is a runtime merge. For strict Env precedence, 
            # we might want to load YAML *before* creating Settings, 
            # but Pydantic Settings prefers Env > File.
            # Here we are treating YAML as a file source similar to .env but structured.
            # A simple way is to update the current settings object if possible, 
            # or recreate it. Pydantic settings are immutable by default but we can workaround.
            # However, for this implementation, let's assume YAML overrides defaults but Env overrides YAML.
            # To achieve this cleanly with Pydantic, we would need a custom source.
            
            # For now, let's manually update key sections if they exist in YAML
            # This allows runtime reloading of YAML to take effect
            
            if "system" in yaml_config:
                self.settings.system = self.settings.system.copy(update=yaml_config["system"])
            if "memory" in yaml_config:
                self.settings.memory = self.settings.memory.copy(update=yaml_config["memory"])
            if "weights" in yaml_config:
                self.settings.weights = self.settings.weights.copy(update=yaml_config["weights"])
            if "thresholds" in yaml_config:
                self.settings.thresholds = self.settings.thresholds.copy(update=yaml_config["thresholds"])
            if "logging" in yaml_config:
                self.settings.logging = self.settings.logging.copy(update=yaml_config["logging"])
            if "timeouts" in yaml_config:
                self.settings.timeouts = self.settings.timeouts.copy(update=yaml_config["timeouts"])
            if "redis" in yaml_config:
                self.settings.redis = self.settings.redis.copy(update=yaml_config["redis"])


    # --- Accessors ---

    def get_action(self, action_id: str) -> ActionConfig:
        return self.actions.get(action_id)
    
    def get_rule(self, action_id: str) -> RuleConfig:
        return self.rules.get(action_id, RuleConfig())

    def get_all_actions(self) -> List[ActionConfig]:
        return list(self.actions.values())

    # System Config Accessors with Defaults (Safe Access via Settings)
    def get_memory_config(self) -> Dict[str, int]:
        return self.settings.memory.model_dump()

    def get_weights(self) -> Dict[str, float]:
        return self.settings.weights.model_dump()
    
    def get_thresholds(self) -> Dict[str, float]:
        return self.settings.thresholds.model_dump()
    
    def get_redis_config(self) -> Dict[str, Any]:
        return self.settings.redis.model_dump()

    # Raw content for Admin API
    async def get_action_catalog_raw(self) -> str:
        async with aiofiles.open(self.action_catalog_path, 'r', encoding='utf-8') as f: return await f.read()

    async def get_rule_config_raw(self) -> str:
        async with aiofiles.open(self.rule_config_path, 'r', encoding='utf-8') as f: return await f.read()

    async def update_action_catalog(self, content: str):
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")
        async with aiofiles.open(self.action_catalog_path, 'w', encoding='utf-8') as f: await f.write(content)
        await self.load()

    async def update_rule_config(self, content: str):
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")
        async with aiofiles.open(self.rule_config_path, 'w', encoding='utf-8') as f: await f.write(content)
        await self.load()
