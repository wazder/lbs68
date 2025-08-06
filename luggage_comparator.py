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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    from segment_anything import SamModel, SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment-anything not available. Install with: pip install segment-anything")

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")


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
        device: str = "auto"
    ):
        """
        Initialize the luggage comparator with SAM and CLIP models.
        
        Args:
            sam_model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            sam_checkpoint_path: Path to SAM checkpoint file
            clip_model_name: CLIP model name from HuggingFace
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        
        # Initialize models
        self.sam_predictor = None
        self.clip_model = None
        self.clip_processor = None
        self.embeddings_db = {}
        self.faiss_index = None
        
        # Setup SAM
        if SAM_AVAILABLE:
            self._setup_sam(sam_model_type, sam_checkpoint_path)
        else:
            print("SAM not available - segmentation features disabled")
            
        # Setup CLIP
        if CLIP_AVAILABLE:
            self._setup_clip(clip_model_name)
        else:
            print("CLIP not available - embedding features disabled")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _setup_sam(self, model_type: str, checkpoint_path: Optional[str]):
        """Initialize SAM model for segmentation."""
        try:
            if checkpoint_path and os.path.exists(checkpoint_path):
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            else:
                # Download checkpoint if not provided
                checkpoint_url = self._get_sam_checkpoint_url(model_type)
                checkpoint_path = f"sam_{model_type}.pth"
                
                if not os.path.exists(checkpoint_path):
                    print(f"Downloading SAM checkpoint: {model_type}")
                    self._download_file(checkpoint_url, checkpoint_path)
                
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            
            sam.to(self.device)
            self.sam_predictor = SamPredictor(sam)
            print(f"SAM model ({model_type}) loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Failed to load SAM: {e}")
            self.sam_predictor = None
    
    def _setup_clip(self, model_name: str):
        """Initialize CLIP model for embeddings."""
        try:
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            print(f"CLIP model ({model_name}) loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Failed to load CLIP: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def _get_sam_checkpoint_url(self, model_type: str) -> str:
        """Get download URL for SAM checkpoint."""
        urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }
        return urls.get(model_type, urls["vit_b"])
    
    def _download_file(self, url: str, filepath: str):
        """Download a file from URL."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path or URL."""
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return np.array(image)
            
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
        print(f"Database saved to {filepath}")
    
    def load_database(self, filepath: str):
        """Load embeddings database from file."""
        data = np.load(filepath)
        self.embeddings_db = {key: data[key] for key in data.files}
        print(f"Database loaded from {filepath} with {len(self.embeddings_db)} images")


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