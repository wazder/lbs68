"""
Configuration management for the Luggage Analysis System
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

from utils import setup_logging

logger = setup_logging()


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    sam_model_type: str = "vit_b"
    sam_checkpoint_path: Optional[str] = None
    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    enable_caching: bool = True
    cache_dir: str = "model_cache"


@dataclass
class ProcessingConfig:
    """Configuration for processing settings."""
    similarity_threshold: float = 75.0
    luggage_detection_threshold: float = 0.7
    batch_size: int = 1
    max_image_size: int = 2048
    enable_segmentation: bool = True
    enable_feature_analysis: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_file_logging: bool = False
    max_log_size_mb: int = 10
    backup_count: int = 3


@dataclass
class CacheConfig:
    """Configuration for cache settings."""
    max_age_days: int = 30
    max_size_gb: float = 5.0
    auto_cleanup: bool = True
    cleanup_interval_hours: int = 24


@dataclass
class OutputConfig:
    """Configuration for output settings."""
    default_output_dir: str = "output"
    create_detailed_reports: bool = True
    create_summary_reports: bool = True
    save_similarity_matrix: bool = True
    save_processed_images: bool = False
    archive_old_results: bool = True
    archive_age_days: int = 7


@dataclass
class LuggageAnalysisConfig:
    """Main configuration class."""
    model: ModelConfig
    processing: ProcessingConfig
    logging: LoggingConfig
    cache: CacheConfig
    output: OutputConfig
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Validate similarity threshold
        if not 0 <= self.processing.similarity_threshold <= 100:
            self.processing.similarity_threshold = 75.0
            logger.warning("Invalid similarity threshold, reset to 75.0")
        
        # Validate device
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if self.model.device not in valid_devices and not self.model.device.startswith("cuda:"):
            self.model.device = "auto"
            logger.warning("Invalid device, reset to 'auto'")
        
        # Validate SAM model type
        valid_sam_types = ["vit_b", "vit_l", "vit_h"]
        if self.model.sam_model_type not in valid_sam_types:
            self.model.sam_model_type = "vit_b"
            logger.warning("Invalid SAM model type, reset to 'vit_b'")
        
        # Validate logging level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level.upper() not in valid_levels:
            self.logging.level = "INFO"
            logger.warning("Invalid logging level, reset to 'INFO'")


class ConfigManager:
    """Manages configuration loading, saving, and environment variable overrides."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_config()
    
    def _get_default_config(self) -> LuggageAnalysisConfig:
        """Get default configuration."""
        return LuggageAnalysisConfig(
            model=ModelConfig(),
            processing=ProcessingConfig(),
            logging=LoggingConfig(),
            cache=CacheConfig(),
            output=OutputConfig()
        )
    
    def _load_config_from_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    logger.error(f"Unsupported config file format: {file_path.suffix}")
                    return None
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            return None
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        env_mappings = {
            # Model settings
            'LUGGAGE_SAM_MODEL_TYPE': ['model', 'sam_model_type'],
            'LUGGAGE_SAM_CHECKPOINT': ['model', 'sam_checkpoint_path'],
            'LUGGAGE_CLIP_MODEL': ['model', 'clip_model_name'],
            'LUGGAGE_DEVICE': ['model', 'device'],
            'LUGGAGE_ENABLE_CACHING': ['model', 'enable_caching'],
            'LUGGAGE_CACHE_DIR': ['model', 'cache_dir'],
            
            # Processing settings
            'LUGGAGE_SIMILARITY_THRESHOLD': ['processing', 'similarity_threshold'],
            'LUGGAGE_DETECTION_THRESHOLD': ['processing', 'luggage_detection_threshold'],
            'LUGGAGE_BATCH_SIZE': ['processing', 'batch_size'],
            'LUGGAGE_MAX_IMAGE_SIZE': ['processing', 'max_image_size'],
            'LUGGAGE_ENABLE_SEGMENTATION': ['processing', 'enable_segmentation'],
            'LUGGAGE_ENABLE_FEATURES': ['processing', 'enable_feature_analysis'],
            
            # Logging settings
            'LUGGAGE_LOG_LEVEL': ['logging', 'level'],
            'LUGGAGE_LOG_FILE': ['logging', 'log_file'],
            'LUGGAGE_LOG_FORMAT': ['logging', 'log_format'],
            'LUGGAGE_ENABLE_FILE_LOGGING': ['logging', 'enable_file_logging'],
            
            # Cache settings
            'LUGGAGE_CACHE_MAX_AGE_DAYS': ['cache', 'max_age_days'],
            'LUGGAGE_CACHE_MAX_SIZE_GB': ['cache', 'max_size_gb'],
            'LUGGAGE_CACHE_AUTO_CLEANUP': ['cache', 'auto_cleanup'],
            
            # Output settings
            'LUGGAGE_OUTPUT_DIR': ['output', 'default_output_dir'],
            'LUGGAGE_CREATE_DETAILED': ['output', 'create_detailed_reports'],
            'LUGGAGE_CREATE_SUMMARY': ['output', 'create_summary_reports'],
            'LUGGAGE_SAVE_MATRIX': ['output', 'save_similarity_matrix'],
            'LUGGAGE_SAVE_IMAGES': ['output', 'save_processed_images'],
            'LUGGAGE_ARCHIVE_RESULTS': ['output', 'archive_old_results'],
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to the nested dictionary
                current = config_dict
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Convert value to appropriate type
                final_key = config_path[-1]
                try:
                    # Try to convert to appropriate type based on current value
                    if isinstance(current.get(final_key), bool):
                        current[final_key] = env_value.lower() in ['true', '1', 'yes', 'on']
                    elif isinstance(current.get(final_key), (int, float)):
                        current[final_key] = float(env_value) if '.' in env_value else int(env_value)
                    else:
                        current[final_key] = env_value
                    
                    logger.info(f"Applied environment override: {env_var}={env_value}")
                    
                except ValueError as e:
                    logger.warning(f"Failed to convert environment variable {env_var}={env_value}: {e}")
        
        return config_dict
    
    def _load_config(self) -> LuggageAnalysisConfig:
        """Load configuration from file and environment variables."""
        config_dict = {}
        
        # Load from default locations if no path specified
        if self.config_path is None:
            default_locations = [
                Path("config.yaml"),
                Path("config.yml"),
                Path("config.json"),
                Path.home() / ".luggage_analysis" / "config.yaml",
                Path("/etc/luggage_analysis/config.yaml")
            ]
            
            for location in default_locations:
                if location.exists():
                    self.config_path = location
                    logger.info(f"Found config file: {location}")
                    break
        
        # Load from file if available
        if self.config_path and self.config_path.exists():
            file_config = self._load_config_from_file(self.config_path)
            if file_config:
                config_dict = file_config
                logger.info(f"Loaded configuration from {self.config_path}")
        
        # Apply environment variable overrides\n        config_dict = self._apply_env_overrides(config_dict)\n        \n        # Create configuration object with defaults for missing values\n        try:\n            # Convert nested dict to dataclass\n            model_config = ModelConfig(**config_dict.get('model', {}))\n            processing_config = ProcessingConfig(**config_dict.get('processing', {}))\n            logging_config = LoggingConfig(**config_dict.get('logging', {}))\n            cache_config = CacheConfig(**config_dict.get('cache', {}))\n            output_config = OutputConfig(**config_dict.get('output', {}))\n            \n            config = LuggageAnalysisConfig(\n                model=model_config,\n                processing=processing_config,\n                logging=logging_config,\n                cache=cache_config,\n                output=output_config\n            )\n            \n            logger.info(\"Configuration loaded successfully\")\n            return config\n            \n        except Exception as e:\n            logger.error(f\"Failed to parse configuration: {e}\")\n            logger.info(\"Using default configuration\")\n            return self._get_default_config()\n    \n    def save_config(self, file_path: Optional[str] = None) -> bool:\n        \"\"\"Save current configuration to file.\"\"\"\n        if file_path:\n            save_path = Path(file_path)\n        elif self.config_path:\n            save_path = self.config_path\n        else:\n            save_path = Path(\"config.yaml\")\n        \n        try:\n            # Create directory if it doesn't exist\n            save_path.parent.mkdir(parents=True, exist_ok=True)\n            \n            # Convert config to dict\n            config_dict = {\n                'model': asdict(self.config.model),\n                'processing': asdict(self.config.processing),\n                'logging': asdict(self.config.logging),\n                'cache': asdict(self.config.cache),\n                'output': asdict(self.config.output)\n            }\n            \n            # Save based on file extension\n            with open(save_path, 'w', encoding='utf-8') as f:\n                if save_path.suffix.lower() in ['.yaml', '.yml']:\n                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)\n                else:\n                    json.dump(config_dict, f, indent=2)\n            \n            logger.info(f\"Configuration saved to {save_path}\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Failed to save configuration to {save_path}: {e}\")\n            return False\n    \n    def get_config(self) -> LuggageAnalysisConfig:\n        \"\"\"Get current configuration.\"\"\"\n        return self.config\n    \n    def update_config(self, **kwargs) -> bool:\n        \"\"\"Update configuration values.\"\"\"\n        try:\n            for key, value in kwargs.items():\n                if hasattr(self.config, key):\n                    setattr(self.config, key, value)\n                    logger.info(f\"Updated config: {key} = {value}\")\n                else:\n                    logger.warning(f\"Unknown config key: {key}\")\n            return True\n        except Exception as e:\n            logger.error(f\"Failed to update configuration: {e}\")\n            return False\n    \n    def create_example_config(self, file_path: str = \"config.example.yaml\") -> bool:\n        \"\"\"Create an example configuration file.\"\"\"\n        try:\n            example_config = self._get_default_config()\n            \n            # Add comments to the example\n            config_dict = {\n                'model': {\n                    'sam_model_type': 'vit_b',  # SAM model size: vit_b, vit_l, vit_h\n                    'sam_checkpoint_path': None,  # Path to SAM checkpoint (auto-download if null)\n                    'clip_model_name': 'openai/clip-vit-base-patch32',  # CLIP model from HuggingFace\n                    'device': 'auto',  # Device: auto, cpu, cuda, mps\n                    'enable_caching': True,  # Enable model caching for faster loading\n                    'cache_dir': 'model_cache'  # Directory for cached models\n                },\n                'processing': {\n                    'similarity_threshold': 75.0,  # Similarity threshold (0-100)\n                    'luggage_detection_threshold': 0.7,  # Luggage detection threshold (0-1)\n                    'batch_size': 1,  # Batch size for processing\n                    'max_image_size': 2048,  # Maximum image size in pixels\n                    'enable_segmentation': True,  # Enable SAM segmentation\n                    'enable_feature_analysis': True  # Enable detailed feature analysis\n                },\n                'logging': {\n                    'level': 'INFO',  # Logging level: DEBUG, INFO, WARNING, ERROR\n                    'log_file': None,  # Log file path (null for console only)\n                    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',\n                    'enable_file_logging': False,  # Enable logging to file\n                    'max_log_size_mb': 10,  # Maximum log file size in MB\n                    'backup_count': 3  # Number of backup log files to keep\n                },\n                'cache': {\n                    'max_age_days': 30,  # Maximum age of cached models in days\n                    'max_size_gb': 5.0,  # Maximum cache size in GB\n                    'auto_cleanup': True,  # Automatically cleanup old cache\n                    'cleanup_interval_hours': 24  # Cache cleanup interval in hours\n                },\n                'output': {\n                    'default_output_dir': 'output',  # Default output directory\n                    'create_detailed_reports': True,  # Create detailed JSON reports\n                    'create_summary_reports': True,  # Create human-readable summaries\n                    'save_similarity_matrix': True,  # Save similarity matrix CSV\n                    'save_processed_images': False,  # Save processed images\n                    'archive_old_results': True,  # Automatically archive old results\n                    'archive_age_days': 7  # Age threshold for archiving in days\n                }\n            }\n            \n            with open(file_path, 'w', encoding='utf-8') as f:\n                f.write(\"# Luggage Analysis System Configuration\\n\")\n                f.write(\"# You can copy this to config.yaml and modify as needed\\n\\n\")\n                yaml.dump(config_dict, f, default_flow_style=False, indent=2)\n            \n            logger.info(f\"Example configuration created: {file_path}\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Failed to create example config: {e}\")\n            return False\n\n\n# Global config manager instance\n_global_config_manager = None\n\n\ndef get_config_manager(config_path: Optional[str] = None) -> ConfigManager:\n    \"\"\"Get global configuration manager instance.\"\"\"\n    global _global_config_manager\n    if _global_config_manager is None:\n        _global_config_manager = ConfigManager(config_path)\n    return _global_config_manager\n\n\ndef get_config() -> LuggageAnalysisConfig:\n    \"\"\"Get current configuration.\"\"\"\n    return get_config_manager().get_config()