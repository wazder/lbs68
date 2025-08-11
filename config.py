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
    sam_model_type: str = "vit_h"
    sam_checkpoint_path: Optional[str] = None
    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    enable_caching: bool = True
    cache_dir: str = "model_cache"


@dataclass
class ProcessingConfig:
    """Configuration for advanced precision processing settings."""
    similarity_threshold: float = 75.0  # Balanced threshold for good grouping
    luggage_detection_threshold: float = 0.8  # Higher precision
    batch_size: int = 1
    max_image_size: int = 2048
    enable_segmentation: bool = True
    enable_feature_analysis: bool = False
    enable_geometric_verification: bool = False  # Disabled for looser grouping
    enable_ensemble_voting: bool = False  # Disabled for simpler grouping


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
        
        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)
        
        # Create configuration object with defaults for missing values
        try:
            # Convert nested dict to dataclass
            model_config = ModelConfig(**config_dict.get('model', {}))
            processing_config = ProcessingConfig(**config_dict.get('processing', {}))
            logging_config = LoggingConfig(**config_dict.get('logging', {}))
            cache_config = CacheConfig(**config_dict.get('cache', {}))
            output_config = OutputConfig(**config_dict.get('output', {}))
            
            config = LuggageAnalysisConfig(
                model=model_config,
                processing=processing_config,
                logging=logging_config,
                cache=cache_config,
                output=output_config
            )
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Failed to parse configuration: {e}")
            logger.info("Using default configuration")
            return self._get_default_config()
    
    def save_config(self, file_path: Optional[str] = None) -> bool:
        """Save current configuration to file."""
        if file_path:
            save_path = Path(file_path)
        elif self.config_path:
            save_path = self.config_path
        else:
            save_path = Path("config.yaml")
        
        try:
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dict
            config_dict = {
                'model': asdict(self.config.model),
                'processing': asdict(self.config.processing),
                'logging': asdict(self.config.logging),
                'cache': asdict(self.config.cache),
                'output': asdict(self.config.output)
            }
            
            # Save based on file extension
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {save_path}: {e}")
            return False
    
    def get_config(self) -> LuggageAnalysisConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> bool:
        """Update configuration values."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"Updated config: {key} = {value}")
                else:
                    logger.warning(f"Unknown config key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def create_example_config(self, file_path: str = "config.example.yaml") -> bool:
        """Create an example configuration file."""
        try:
            example_config = self._get_default_config()
            
            # Add comments to the example
            config_dict = {
                'model': {
                    'sam_model_type': 'vit_b',  # SAM model size: vit_b, vit_l, vit_h
                    'sam_checkpoint_path': None,  # Path to SAM checkpoint (auto-download if null)
                    'clip_model_name': 'openai/clip-vit-base-patch32',  # CLIP model from HuggingFace
                    'device': 'auto',  # Device: auto, cpu, cuda, mps
                    'enable_caching': True,  # Enable model caching for faster loading
                    'cache_dir': 'model_cache'  # Directory for cached models
                },
                'processing': {
                    'similarity_threshold': 95.0,  # Similarity threshold (0-100) - Ultra-precise default
                    'luggage_detection_threshold': 0.8,  # Luggage detection threshold (0-1)
                    'batch_size': 1,  # Batch size for processing
                    'max_image_size': 2048,  # Maximum image size in pixels
                    'enable_segmentation': True,  # Enable SAM segmentation
                    'enable_feature_analysis': True,  # Enable detailed feature analysis
                    'enable_geometric_verification': True,  # Enable geometric verification
                    'enable_ensemble_voting': True  # Enable ensemble voting
                },
                'logging': {
                    'level': 'INFO',  # Logging level: DEBUG, INFO, WARNING, ERROR
                    'log_file': None,  # Log file path (null for console only)
                    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'enable_file_logging': False,  # Enable logging to file
                    'max_log_size_mb': 10,  # Maximum log file size in MB
                    'backup_count': 3  # Number of backup log files to keep
                },
                'cache': {
                    'max_age_days': 30,  # Maximum age of cached models in days
                    'max_size_gb': 5.0,  # Maximum cache size in GB
                    'auto_cleanup': True,  # Automatically cleanup old cache
                    'cleanup_interval_hours': 24  # Cache cleanup interval in hours
                },
                'output': {
                    'default_output_dir': 'output',  # Default output directory
                    'create_detailed_reports': True,  # Create detailed JSON reports
                    'create_summary_reports': True,  # Create human-readable summaries
                    'save_similarity_matrix': True,  # Save similarity matrix CSV
                    'save_processed_images': False,  # Save processed images
                    'archive_old_results': True,  # Automatically archive old results
                    'archive_age_days': 7  # Age threshold for archiving in days
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Luggage Analysis System Configuration\n")
                f.write("# You can copy this to config.yaml and modify as needed\n\n")
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Example configuration created: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create example config: {e}")
            return False


# Global config manager instance
_global_config_manager = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_path)
    return _global_config_manager


def get_config() -> LuggageAnalysisConfig:
    """Get current configuration."""
    return get_config_manager().get_config()