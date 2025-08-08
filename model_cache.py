"""
Model caching and persistence utilities for the Luggage Analysis System
"""

import os
import pickle
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import torch

from utils import setup_logging, format_file_size

logger = setup_logging()


class ModelCache:
    """Handles caching and persistence of loaded models."""
    
    def __init__(self, cache_dir: str = "model_cache"):
        """
        Initialize model cache.
        
        Args:
            cache_dir: Directory to store cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"Model cache initialized: {self.cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        return {
            "created": time.time(),
            "models": {},
            "last_cleanup": time.time()
        }
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_model_hash(self, model_type: str, checkpoint_path: str, device: str) -> str:
        """Generate unique hash for model configuration."""
        # Create hash based on model parameters
        config_str = f"{model_type}_{checkpoint_path}_{device}"
        
        # Add checkpoint file hash if it exists
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                # Read first and last 1MB for faster hashing
                f.seek(0)
                start_chunk = f.read(1024 * 1024)
                f.seek(-1024 * 1024, 2)
                end_chunk = f.read(1024 * 1024)
                config_str += hashlib.md5(start_chunk + end_chunk).hexdigest()
        
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def get_cache_path(self, model_hash: str) -> Path:
        """Get cache file path for a model hash."""
        return self.cache_dir / f"model_{model_hash}.cache"
    
    def is_cached(self, model_type: str, checkpoint_path: str, device: str) -> bool:
        """Check if model is cached."""
        model_hash = self._get_model_hash(model_type, checkpoint_path, device)
        cache_path = self.get_cache_path(model_hash)
        
        if not cache_path.exists():
            return False
            
        # Check if cache is valid
        if model_hash not in self.metadata["models"]:
            return False
            
        cache_info = self.metadata["models"][model_hash]
        
        # Check if checkpoint file was modified
        if os.path.exists(checkpoint_path):
            checkpoint_mtime = os.path.getmtime(checkpoint_path)
            if checkpoint_mtime > cache_info.get("cached_at", 0):
                logger.info(f"Cache invalid: checkpoint modified after caching")
                return False
        
        return True
    
    def cache_model(self, model, model_type: str, checkpoint_path: str, device: str):
        """Cache a loaded model."""
        model_hash = self._get_model_hash(model_type, checkpoint_path, device)
        cache_path = self.get_cache_path(model_hash)
        
        try:
            logger.info(f"Caching model: {model_type} -> {cache_path}")
            
            # Save model state dict instead of full model to save space
            model_state = {
                'state_dict': model.state_dict(),
                'model_type': model_type,
                'device': str(device),
                'cached_at': time.time()
            }
            
            # Use torch.save for model state
            torch.save(model_state, cache_path)
            
            # Update metadata
            cache_size = cache_path.stat().st_size
            self.metadata["models"][model_hash] = {
                'model_type': model_type,
                'checkpoint_path': checkpoint_path,
                'device': str(device),
                'cached_at': time.time(),
                'cache_size': cache_size,
                'access_count': 0,
                'last_access': time.time()
            }
            
            self._save_metadata()
            logger.info(f"Model cached successfully ({format_file_size(cache_size)})")
            
        except Exception as e:
            logger.error(f"Failed to cache model: {e}")
            if cache_path.exists():
                cache_path.unlink()  # Clean up partial cache
    
    def load_cached_model(self, model_class, model_type: str, checkpoint_path: str, device: str):
        """Load a cached model."""
        model_hash = self._get_model_hash(model_type, checkpoint_path, device)
        cache_path = self.get_cache_path(model_hash)
        
        if not self.is_cached(model_type, checkpoint_path, device):
            return None
        
        try:
            logger.info(f"Loading cached model: {cache_path}")
            start_time = time.time()
            
            # Load cached model state
            cached_state = torch.load(cache_path, map_location=device)
            
            # Create new model instance
            model = model_class()
            model.load_state_dict(cached_state['state_dict'])
            model.to(device)
            
            # Update access statistics
            if model_hash in self.metadata["models"]:
                self.metadata["models"][model_hash]["access_count"] += 1
                self.metadata["models"][model_hash]["last_access"] = time.time()
                self._save_metadata()
            
            load_time = time.time() - start_time
            logger.info(f"Cached model loaded in {load_time:.2f}s")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load cached model: {e}")
            # Remove corrupted cache
            if cache_path.exists():
                cache_path.unlink()
            if model_hash in self.metadata["models"]:
                del self.metadata["models"][model_hash]
                self._save_metadata()
            return None
    
    def cleanup_cache(self, max_age_days: int = 30, max_size_gb: float = 5.0):
        """Clean up old or excessive cache files."""
        logger.info("Starting cache cleanup...")
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        
        # Get all cached models sorted by last access
        cached_models = []
        total_size = 0
        
        for model_hash, info in self.metadata["models"].items():
            cache_path = self.get_cache_path(model_hash)
            if cache_path.exists():
                cached_models.append((
                    model_hash,
                    info.get("last_access", 0),
                    info.get("cache_size", cache_path.stat().st_size)
                ))
                total_size += info.get("cache_size", 0)
        
        # Sort by last access (oldest first)
        cached_models.sort(key=lambda x: x[1])
        
        removed_count = 0
        freed_size = 0
        
        for model_hash, last_access, size in cached_models:
            cache_path = self.get_cache_path(model_hash)
            should_remove = False
            reason = ""
            
            # Remove if too old
            if current_time - last_access > max_age_seconds:
                should_remove = True
                reason = f"too old ({(current_time - last_access) / 86400:.1f} days)"
            
            # Remove if total size exceeds limit (remove oldest first)
            elif total_size > max_size_bytes:
                should_remove = True
                reason = f"size limit exceeded (total: {format_file_size(total_size)})"
                total_size -= size
            
            if should_remove:
                try:
                    cache_path.unlink()
                    if model_hash in self.metadata["models"]:
                        del self.metadata["models"][model_hash]
                    removed_count += 1
                    freed_size += size
                    logger.info(f"Removed cached model {model_hash}: {reason}")
                except Exception as e:
                    logger.error(f"Failed to remove cache {model_hash}: {e}")
        
        if removed_count > 0:
            self.metadata["last_cleanup"] = current_time
            self._save_metadata()
            logger.info(f"Cache cleanup completed: {removed_count} models removed, {format_file_size(freed_size)} freed")
        else:
            logger.info("Cache cleanup completed: nothing to remove")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "total_models": len(self.metadata["models"]),
            "total_size": 0,
            "models": []
        }
        
        for model_hash, info in self.metadata["models"].items():
            cache_path = self.get_cache_path(model_hash)
            if cache_path.exists():
                stats["total_size"] += info.get("cache_size", 0)
                stats["models"].append({
                    "hash": model_hash,
                    "type": info.get("model_type", "unknown"),
                    "size": format_file_size(info.get("cache_size", 0)),
                    "access_count": info.get("access_count", 0),
                    "last_access": time.ctime(info.get("last_access", 0))
                })
        
        stats["total_size_formatted"] = format_file_size(stats["total_size"])
        return stats


# Global cache instance
_global_cache = None


def get_model_cache() -> ModelCache:
    """Get global model cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ModelCache()
    return _global_cache