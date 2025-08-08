import os
import numpy as np
import torch
import cv2
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import requests
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import logging
from pathlib import Path
import time

from utils import setup_logging, validate_image_file, retry_on_failure, memory_cleanup, safe_file_operation
from model_cache import get_model_cache

# Setup logging
logger = setup_logging()

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
    logger.info("Segment Anything (SAM) is available")
except ImportError as e:
    SAM_AVAILABLE = False
    logger.warning(f"Segment Anything not available: {e}")
    logger.info("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
    logger.info("CLIP (transformers) is available")
except ImportError as e:
    CLIP_AVAILABLE = False
    logger.warning(f"CLIP not available: {e}")
    logger.info("Install with: pip install transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS is available for efficient similarity search")
except ImportError as e:
    FAISS_AVAILABLE = False
    logger.warning(f"FAISS not available: {e}")
    logger.info("Install with: pip install faiss-cpu")


class LuggageComparator:
    """
    A comprehensive system for comparing luggage photos using SAM segmentation
    and CLIP embeddings to achieve high accuracy despite lighting, angle, and
    clothing variations.
    """
    
    def __init__(
        self, 
        sam_model_type: str = "vit_b",
        sam_checkpoint_path: Optional[str] = None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        enable_logging: bool = True
    ):
        """
        Initialize the luggage comparator with SAM and CLIP models.
        
        Args:
            sam_model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            sam_checkpoint_path: Path to SAM checkpoint file
            clip_model_name: CLIP model name from HuggingFace
            device: Device to use ('auto', 'cpu', 'cuda')
            enable_logging: Enable detailed logging
        """
        if enable_logging:
            self.logger = logger
        else:
            self.logger = logging.getLogger('luggage_analysis')
            self.logger.disabled = True
            
        self.logger.info("Initializing LuggageComparator...")
        
        try:
            self.device = self._setup_device(device)
            self.logger.info(f"Using device: {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to setup device: {e}")
            self.device = torch.device("cpu")
            self.logger.info("Falling back to CPU")
        
        # Initialize models
        self.sam_predictor = None
        self.clip_model = None
        self.clip_processor = None
        self.embeddings_db = {}
        self.faiss_index = None
        
        # Model setup with error handling
        self.sam_model_type = sam_model_type
        self.clip_model_name = clip_model_name
        
        # Setup SAM with graceful fallback
        if SAM_AVAILABLE:
            try:
                self._setup_sam(sam_model_type, sam_checkpoint_path)
                self.logger.info("SAM model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load SAM model: {e}")
                self.logger.info("Segmentation features will be disabled")
                self.sam_predictor = None
        else:
            self.logger.info("SAM not available - segmentation features disabled")
            
        # Setup CLIP with graceful fallback
        if CLIP_AVAILABLE:
            try:
                self._setup_clip(clip_model_name)
                self.logger.info("CLIP model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load CLIP model: {e}")
                self.logger.info("Embedding features will be disabled")
                self.clip_model = None
                self.clip_processor = None
        else:
            self.logger.info("CLIP not available - embedding features disabled")
            
        # Check if we have at least one working model
        if self.sam_predictor is None and self.clip_model is None:
            self.logger.warning("No models loaded successfully - limited functionality available")
        
        self.logger.info("LuggageComparator initialization completed")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the computation device with fallback options."""
        if device == "auto":
            if torch.cuda.is_available():
                try:
                    # Test CUDA device
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    self.logger.info("CUDA is available and working")
                    return torch.device("cuda")
                except Exception as e:
                    self.logger.warning(f"CUDA available but not working: {e}")
                    
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    # Test MPS device
                    test_tensor = torch.zeros(1).to('mps')
                    del test_tensor
                    self.logger.info("MPS is available and working")
                    return torch.device("mps")
                except Exception as e:
                    self.logger.warning(f"MPS available but not working: {e}")
                    
            self.logger.info("Using CPU device")
            return torch.device("cpu")
        else:
            try:
                device_obj = torch.device(device)
                # Test the specified device
                test_tensor = torch.zeros(1).to(device_obj)
                del test_tensor
                if device_obj.type == 'cuda':
                    torch.cuda.empty_cache()
                return device_obj
            except Exception as e:
                self.logger.error(f"Specified device '{device}' not working: {e}")
                self.logger.info("Falling back to CPU")
                return torch.device("cpu")
    
    @retry_on_failure(max_retries=2, delay=2.0)
    def _setup_sam(self, model_type: str, checkpoint_path: Optional[str]):
        """Initialize SAM model for segmentation with robust error handling."""
        if model_type not in ['vit_b', 'vit_l', 'vit_h']:
            raise ValueError(f"Invalid SAM model type: {model_type}. Must be 'vit_b', 'vit_l', or 'vit_h'")
        
        self.logger.info(f"Loading SAM model: {model_type}")
        
        # Determine checkpoint path
        if checkpoint_path and Path(checkpoint_path).exists():
            self.logger.info(f"Using provided checkpoint: {checkpoint_path}")
            final_checkpoint_path = checkpoint_path
        else:
            # Use default checkpoint path
            final_checkpoint_path = f"sam_{model_type}.pth"
            
            if not Path(final_checkpoint_path).exists():
                self.logger.info(f"Checkpoint not found, downloading SAM {model_type}...")
                checkpoint_url = self._get_sam_checkpoint_url(model_type)
                
                try:
                    self._download_file(checkpoint_url, final_checkpoint_path)
                    self.logger.info(f"Successfully downloaded SAM checkpoint to {final_checkpoint_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to download SAM checkpoint: {e}")
            else:
                self.logger.info(f"Using existing checkpoint: {final_checkpoint_path}")
        
        # Validate checkpoint file
        checkpoint_size = Path(final_checkpoint_path).stat().st_size / (1024 * 1024)  # MB
        if checkpoint_size < 50:  # SAM models are typically >300MB
            raise RuntimeError(f"Checkpoint file seems corrupted (size: {checkpoint_size:.1f}MB)")
        
        # Try to load from cache first
        cache = get_model_cache()
        
        # Check if we can load from cache
        if cache.is_cached(model_type, final_checkpoint_path, str(self.device)):
            self.logger.info("Loading SAM model from cache...")
            try:
                # Create a dummy model instance to get the class
                temp_sam = sam_model_registry[model_type]()
                sam_class = type(temp_sam.image_encoder)  # Get the encoder class
                del temp_sam
                
                # Try to load full model from cache
                sam = cache.load_cached_model(lambda: sam_model_registry[model_type](checkpoint=final_checkpoint_path),
                                             model_type, final_checkpoint_path, str(self.device))
                
                if sam is not None:
                    sam.to(self.device)
                    self.sam_predictor = SamPredictor(sam)
                    self.logger.info(f"SAM model ({model_type}) loaded from cache successfully")
                    return  # Successfully loaded from cache
                else:
                    self.logger.warning("Failed to load from cache, loading from checkpoint...")
            except Exception as e:
                self.logger.warning(f"Cache loading failed: {e}, falling back to checkpoint loading")
        
        # Load model from checkpoint with memory management
        with memory_cleanup():
            try:
                self.logger.info(f"Loading SAM model from checkpoint ({checkpoint_size:.1f}MB)...")
                start_time = time.time()
                
                sam = sam_model_registry[model_type](checkpoint=final_checkpoint_path)
                
                self.logger.info(f"Moving SAM model to {self.device}...")
                sam.to(self.device)
                
                self.sam_predictor = SamPredictor(sam)
                
                load_time = time.time() - start_time
                self.logger.info(f"SAM model ({model_type}) loaded successfully on {self.device} in {load_time:.2f}s")
                
                # Cache the loaded model for future use
                try:
                    self.logger.info("Caching SAM model for faster future loading...")
                    cache.cache_model(sam, model_type, final_checkpoint_path, str(self.device))
                except Exception as e:
                    self.logger.warning(f"Failed to cache model: {e}")
                
            except torch.cuda.OutOfMemoryError:
                self.logger.error("CUDA out of memory loading SAM model, falling back to CPU")
                if self.device.type != 'cpu':
                    self.device = torch.device('cpu')
                    sam.to(self.device)
                    self.sam_predictor = SamPredictor(sam)
                else:
                    # Still try to cache even on CPU fallback
                    try:
                        cache.cache_model(sam, model_type, final_checkpoint_path, str(self.device))
                    except:
                        pass  # Don't fail if caching fails
                    raise RuntimeError("Insufficient memory to load SAM model even on CPU")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize SAM model: {e}")
    
    @retry_on_failure(max_retries=2, delay=1.0)
    def _setup_clip(self, model_name: str):
        """Initialize CLIP model for embeddings with robust error handling and caching."""
        self.logger.info(f"Loading CLIP model: {model_name}")
        
        # Try to load from cache first
        cache = get_model_cache()
        cache_key = f"clip_{model_name.replace('/', '_')}"
        
        # For CLIP, we use model name as "checkpoint path" since it's from HuggingFace
        if cache.is_cached(cache_key, model_name, str(self.device)):
            self.logger.info("Loading CLIP model from cache...")
            try:
                # Load processor (always load fresh as it's lightweight)
                self.clip_processor = CLIPProcessor.from_pretrained(model_name)
                
                # Try to load model from cache
                cached_model = cache.load_cached_model(
                    lambda: CLIPModel.from_pretrained(model_name),
                    cache_key, model_name, str(self.device)
                )
                
                if cached_model is not None:
                    self.clip_model = cached_model
                    self.clip_model.to(self.device)
                    
                    # Quick test
                    test_image = Image.new('RGB', (224, 224), color='white')
                    inputs = self.clip_processor(images=test_image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        _ = self.clip_model.get_image_features(**inputs)
                    
                    self.logger.info(f"CLIP model ({model_name}) loaded from cache successfully")
                    return
                else:
                    self.logger.warning("Failed to load CLIP from cache, loading from HuggingFace...")
            except Exception as e:
                self.logger.warning(f"Cache loading failed: {e}, falling back to HuggingFace loading")
        
        with memory_cleanup():
            try:
                # Load processor first (lighter)
                self.logger.info("Loading CLIP processor...")
                self.clip_processor = CLIPProcessor.from_pretrained(model_name)
                
                # Load model
                self.logger.info(f"Loading CLIP model and moving to {self.device}...")
                start_time = time.time()
                
                self.clip_model = CLIPModel.from_pretrained(model_name)
                self.clip_model.to(self.device)
                
                load_time = time.time() - start_time
                
                # Test the model with a dummy input
                self.logger.info("Testing CLIP model...")
                test_image = Image.new('RGB', (224, 224), color='white')
                inputs = self.clip_processor(images=test_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    _ = self.clip_model.get_image_features(**inputs)
                
                self.logger.info(f"CLIP model ({model_name}) loaded successfully on {self.device} in {load_time:.2f}s")
                
                # Cache the loaded model for future use
                try:
                    self.logger.info("Caching CLIP model for faster future loading...")
                    cache.cache_model(self.clip_model, cache_key, model_name, str(self.device))
                except Exception as e:
                    self.logger.warning(f"Failed to cache CLIP model: {e}")
                
            except torch.cuda.OutOfMemoryError:
                self.logger.error("CUDA out of memory loading CLIP model, falling back to CPU")
                if self.device.type != 'cpu':
                    self.device = torch.device('cpu')
                    if self.clip_model is not None:
                        self.clip_model.to(self.device)
                        # Try to cache the CPU version too
                        try:
                            cache.cache_model(self.clip_model, cache_key, model_name, str(self.device))
                        except:
                            pass
                else:
                    raise RuntimeError("Insufficient memory to load CLIP model even on CPU")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize CLIP model: {e}")
    
    def _get_sam_checkpoint_url(self, model_type: str) -> str:
        """Get download URL for SAM checkpoint."""
        urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }
        return urls.get(model_type, urls["vit_b"])
    
    @retry_on_failure(max_retries=3, delay=5.0, exponential_backoff=True)
    def _download_file(self, url: str, filepath: str):
        """Download a file from URL with progress tracking and error handling."""
        self.logger.info(f"Downloading {url} to {filepath}...")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress tracking
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024 * 10) == 0:  # Log every 10MB
                                self.logger.info(f"Download progress: {progress:.1f}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)")
            
            self.logger.info(f"Download completed: {filepath} ({downloaded/1024/1024:.1f}MB)")
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout downloading {url}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Connection error downloading {url}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP error downloading {url}: {e}")
        except Exception as e:
            # Clean up partial download
            if Path(filepath).exists():
                Path(filepath).unlink()
            raise RuntimeError(f"Failed to download {url}: {e}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path or URL with robust error handling."""
        self.logger.debug(f"Loading image: {image_path}")
        
        try:
            if image_path.startswith(('http://', 'https://')):
                # Handle URL images
                self.logger.debug("Loading image from URL")
                response = requests.get(image_path, timeout=30)
                response.raise_for_status()
                
                if len(response.content) == 0:
                    raise ValueError("Empty image data received from URL")
                
                image = Image.open(BytesIO(response.content))
                
            else:
                # Handle local file images
                image_path = Path(image_path)
                
                # Validate file before opening
                if not validate_image_file(image_path):
                    raise ValueError(f"Invalid or corrupted image file: {image_path}")
                
                image = Image.open(image_path)
            
            # Validate image properties
            if image.width == 0 or image.height == 0:
                raise ValueError("Image has zero width or height")
                
            if image.width * image.height > 100000000:  # 100MP limit
                self.logger.warning(f"Very large image ({image.width}x{image.height}), this may cause memory issues")
            
            # Convert to RGB if needed
            original_mode = image.mode
            if image.mode != 'RGB':
                self.logger.debug(f"Converting image from {original_mode} to RGB")
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Validate the resulting array
            if image_array.shape[2] != 3:
                raise ValueError(f"Expected RGB image (3 channels), got {image_array.shape[2]} channels")
            
            self.logger.debug(f"Successfully loaded image: {image_array.shape[1]}x{image_array.shape[0]} pixels")
            return image_array
            
        except requests.exceptions.Timeout:
            raise ValueError(f"Timeout loading image from URL: {image_path}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error loading image from URL {image_path}: {e}")
        except Image.UnidentifiedImageError:
            raise ValueError(f"Cannot identify image file (unsupported format): {image_path}")
        except PermissionError:
            raise ValueError(f"Permission denied accessing image file: {image_path}")
        except FileNotFoundError:
            raise ValueError(f"Image file not found: {image_path}")
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {e}")
    
    def segment_luggage(
        self, 
        image: np.ndarray, 
        point_prompts: Optional[List[Tuple[int, int]]] = None,
        box_prompt: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Segment luggage from image using SAM.
        
        Args:
            image: Input image as numpy array
            point_prompts: List of (x, y) points indicating luggage
            box_prompt: Bounding box [x1, y1, x2, y2] around luggage
            
        Returns:
            Binary mask of the luggage
        """
        if self.sam_predictor is None:
            raise RuntimeError("SAM model not available")
        
        self.sam_predictor.set_image(image)
        
        # If no prompts provided, use automatic segmentation
        if point_prompts is None and box_prompt is None:
            # Use center point as default prompt
            h, w = image.shape[:2]
            point_prompts = [(w//2, h//2)]
        
        # Prepare prompts
        input_points = np.array(point_prompts) if point_prompts else None
        input_labels = np.array([1] * len(point_prompts)) if point_prompts else None
        input_boxes = np.array(box_prompt) if box_prompt else None
        
        # Generate mask
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_boxes,
            multimask_output=True
        )
        
        # Return the best mask (highest score)
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx]
    
    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to image, keeping only the masked region."""
        # Create a 3-channel mask
        mask_3d = np.stack([mask] * 3, axis=-1)
        
        # Apply mask (keep masked area, set rest to white background)
        masked_image = image * mask_3d + (1 - mask_3d) * 255
        
        return masked_image.astype(np.uint8)
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract CLIP embedding from image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized embedding vector
        """
        if self.clip_model is None or self.clip_processor is None:
            raise RuntimeError("CLIP model not available")
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Process image
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            # Normalize embedding
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().flatten()
    
    def detect_luggage(self, image: np.ndarray, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Detect if image contains luggage using CLIP text classification.
        
        Args:
            image: Input image as numpy array
            threshold: Classification threshold (0-1)
            
        Returns:
            Dictionary with detection results
        """
        if self.clip_model is None or self.clip_processor is None:
            return {'is_luggage': True, 'confidence': 0.0, 'reason': 'CLIP not available, assuming luggage'}
        
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Define text prompts for classification
            luggage_prompts = [
                "a photo of a suitcase",
                "a photo of a hard shell suitcase", 
                "a photo of a travel suitcase",
                "a photo of a rolling suitcase",
                "a photo of a luggage bag",
                "a photo of a travel bag",
                "a photo of a backpack",
                "a photo of a duffel bag",
                "a photo of a trolley bag",
                "a photo of a carry-on luggage",
                "a photo of a checked luggage",
                "a photo of a wheeled suitcase",
                "a photo of a travel case",
                "a photo of a luggage set"
            ]
            
            non_luggage_prompts = [
                "a photo of a cardboard box",
                "a photo of a box with bags on it",
                "a photo of a storage box",
                "a photo of a package",
                "a photo of a person",
                "a photo of a wall",
                "a photo of a floor", 
                "a photo of furniture",
                "a photo of a room",
                "a photo of random objects",
                "a photo of clothing",
                "a photo of electronics",
                "a photo of household items",
                "a photo of a shopping bag",
                "a photo of a plastic bag",
                "a photo of a paper bag",
                "a photo of a tote bag",
                "a photo of a handbag",
                "a photo of a purse",
                "a photo of a messenger bag",
                "a photo of a gym bag",
                "a photo of a sports bag",
                "a photo of a beach bag",
                "a photo of a cosmetic bag",
                "a photo of a makeup bag"
            ]
            
            all_prompts = luggage_prompts + non_luggage_prompts
            
            # Process image and text
            inputs = self.clip_processor(
                text=all_prompts, 
                images=pil_image, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = torch.softmax(logits_per_image, dim=-1)
            
            # Calculate luggage vs non-luggage scores
            luggage_probs = probs[0][:len(luggage_prompts)]
            non_luggage_probs = probs[0][len(luggage_prompts):]
            
            luggage_score = float(torch.sum(luggage_probs))
            non_luggage_score = float(torch.sum(non_luggage_probs))
            
            # Normalize scores
            total_score = luggage_score + non_luggage_score
            if total_score > 0:
                luggage_confidence = luggage_score / total_score
            else:
                luggage_confidence = 0.5
            
            is_luggage = luggage_confidence >= threshold
            
            # Find best matching prompt for explanation
            best_prompt_idx = torch.argmax(probs[0])
            best_prompt = all_prompts[best_prompt_idx]
            best_confidence = float(probs[0][best_prompt_idx])
            
            return {
                'is_luggage': is_luggage,
                'confidence': round(luggage_confidence, 3),
                'luggage_score': round(luggage_score, 3),
                'non_luggage_score': round(non_luggage_score, 3),
                'best_match': best_prompt,
                'best_match_confidence': round(best_confidence, 3),
                'reason': f"Best match: '{best_prompt}' ({best_confidence:.1%})"
            }
            
        except Exception as e:
            print(f"Warning: Luggage detection failed: {e}")
            return {'is_luggage': True, 'confidence': 0.0, 'reason': f'Detection error: {e}'}
    
    def process_image(
        self, 
        image_path: str,
        point_prompts: Optional[List[Tuple[int, int]]] = None,
        box_prompt: Optional[List[int]] = None,
        save_masked_image: Optional[str] = None
    ) -> np.ndarray:
        """
        Complete pipeline: load image -> segment -> extract embedding.
        
        Args:
            image_path: Path to image file
            point_prompts: Points indicating luggage location
            box_prompt: Bounding box around luggage
            save_masked_image: Path to save the masked image (optional)
            
        Returns:
            CLIP embedding of the masked luggage
        """
        # Load image
        image = self.load_image(image_path)
        
        # Segment luggage
        if self.sam_predictor is not None:
            mask = self.segment_luggage(image, point_prompts, box_prompt)
            masked_image = self.apply_mask(image, mask)
        else:
            # If SAM not available, use original image
            print("Warning: Using original image without segmentation")
            masked_image = image
        
        # Save masked image if requested
        if save_masked_image:
            Image.fromarray(masked_image).save(save_masked_image)
        
        # Extract embedding
        embedding = self.extract_embedding(masked_image)
        
        return embedding
    
    def compare_images(
        self, 
        image1_path: str, 
        image2_path: str,
        **kwargs
    ) -> float:
        """
        Compare two luggage images and return similarity percentage.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            **kwargs: Additional arguments for process_image
            
        Returns:
            Similarity percentage (0-100)
        """
        # Process both images
        embedding1 = self.process_image(image1_path, **kwargs)
        embedding2 = self.process_image(image2_path, **kwargs)
        
        # Compute cosine similarity
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        
        # Convert to percentage
        similarity_percentage = (similarity + 1) / 2 * 100  # Map from [-1,1] to [0,100]
        
        return similarity_percentage
    
    def add_to_database(self, image_id: str, image_path: str, **kwargs) -> np.ndarray:
        """Add image embedding to database."""
        embedding = self.process_image(image_path, **kwargs)
        self.embeddings_db[image_id] = embedding
        return embedding
    
    def find_similar_images(
        self, 
        query_image_path: str, 
        top_k: int = 5,
        threshold: float = 0.0,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Find similar images in the database.
        
        Args:
            query_image_path: Path to query image
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0-100)
            **kwargs: Additional arguments for process_image
            
        Returns:
            List of (image_id, similarity_percentage) tuples
        """
        if not self.embeddings_db:
            return []
        
        # Process query image
        query_embedding = self.process_image(query_image_path, **kwargs)
        
        # Compare with all images in database
        similarities = []
        for image_id, embedding in self.embeddings_db.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarity_percentage = (similarity + 1) / 2 * 100
            
            if similarity_percentage >= threshold:
                similarities.append((image_id, similarity_percentage))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def setup_faiss_index(self, embedding_dim: int = 512):
        """Setup FAISS index for efficient similarity search."""
        if not FAISS_AVAILABLE:
            print("FAISS not available - using linear search")
            return
        
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        print(f"FAISS index created with dimension {embedding_dim}")
    
    def save_database(self, filepath: str):
        """Save embeddings database to file."""
        np.savez(filepath, **self.embeddings_db)
        self.logger.info(f"Database saved to {filepath}")
    
    def load_database(self, filepath: str):
        """Load embeddings database from file."""
        data = np.load(filepath)
        self.embeddings_db = {key: data[key] for key in data.files}
        self.logger.info(f"Database loaded from {filepath} with {len(self.embeddings_db)} images")
    
    def cleanup(self):
        """Clean up resources and free memory."""
        self.logger.info("Cleaning up LuggageComparator resources...")
        
        # Clear embeddings database
        if hasattr(self, 'embeddings_db'):
            self.embeddings_db.clear()
        
        # Clean up models (they will be cached)
        if hasattr(self, 'sam_predictor'):
            self.sam_predictor = None
        
        if hasattr(self, 'clip_model'):
            self.clip_model = None
            
        if hasattr(self, 'clip_processor'):
            self.clip_processor = None
            
        if hasattr(self, 'faiss_index'):
            self.faiss_index = None
        
        # Force cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("LuggageComparator cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'logger'):
                self.cleanup()
        except:
            pass  # Avoid errors during shutdown


def main():
    """Example usage of the LuggageComparator."""
    # Initialize the comparator
    comparator = LuggageComparator()
    
    # Example: Compare two images
    # similarity = comparator.compare_images("luggage1.jpg", "luggage2.jpg")
    # print(f"Similarity: {similarity:.2f}%")
    
    # Example: Build a database and search
    # comparator.add_to_database("luggage_001", "path/to/luggage1.jpg")
    # comparator.add_to_database("luggage_002", "path/to/luggage2.jpg")
    # 
    # similar_images = comparator.find_similar_images("query_luggage.jpg", top_k=3)
    # for image_id, similarity in similar_images:
    #     print(f"{image_id}: {similarity:.2f}% similar")
    
    print("LuggageComparator initialized successfully!")
    print("Use comparator.compare_images(img1, img2) to compare two images")
    print("Use comparator.add_to_database(id, path) to build a searchable database")


if __name__ == "__main__":
    main()