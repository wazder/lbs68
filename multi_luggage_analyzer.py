"""
Multi Luggage Photo Analysis and Grouping System

This system analyzes multiple luggage photos, groups those belonging to the same luggage,
and provides detailed feature analysis and reporting.
"""

import os
import json
import numpy as np
import cv2
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from luggage_comparator import LuggageComparator


class LuggageFeatureAnalyzer:
    """Class for analyzing luggage features."""
    
    def __init__(self):
        self.color_names = {
            'black': ([0, 0, 0], [50, 50, 50]),
            'brown': ([20, 20, 10], [100, 80, 60]),
            'navy': ([20, 20, 50], [60, 60, 120]),
            'gray': ([50, 50, 50], [150, 150, 150]),
            'blue': ([50, 50, 100], [120, 120, 255]),
            'red': ([100, 20, 20], [255, 80, 80]),
            'green': ([20, 80, 20], [80, 200, 80]),
            'yellow': ([200, 200, 50], [255, 255, 150]),
            'white': ([200, 200, 200], [255, 255, 255]),
            'orange': ([200, 100, 20], [255, 180, 80]),
            'purple': ([100, 50, 100], [180, 120, 180]),
            'pink': ([200, 100, 150], [255, 180, 220])
        }
    
    def analyze_color(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze luggage color."""
        if mask is not None:
            masked_image = image * np.stack([mask] * 3, axis=-1)
            pixels = masked_image[mask > 0]
        else:
            pixels = image.reshape(-1, 3)
        
        if len(pixels) == 0:
            return {'dominant_color': 'unknown', 'color_distribution': {}}
        
        # Find dominant color
        mean_color = np.mean(pixels, axis=0)
        
        # Find closest color name
        closest_color = 'other'
        min_distance = float('inf')
        
        for color_name, (low, high) in self.color_names.items():
            low = np.array(low)
            high = np.array(high)
            
            if np.all(mean_color >= low) and np.all(mean_color <= high):
                distance = np.linalg.norm(mean_color - (low + high) / 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_name
        
        # Calculate color distribution
        color_counts = {}
        for color_name, (low, high) in self.color_names.items():
            low = np.array(low)
            high = np.array(high)
            
            in_range = np.all((pixels >= low) & (pixels <= high), axis=1)
            count = np.sum(in_range)
            percentage = (count / len(pixels)) * 100
            
            if percentage > 5:  # Colors above 5%
                color_counts[color_name] = round(percentage, 1)
        
        return {
            'dominant_color': closest_color,
            'mean_rgb': mean_color.tolist(),
            'color_distribution': color_counts
        }
    
    def analyze_size(self, mask: np.ndarray) -> Dict[str, Any]:
        """Analyze luggage size."""
        if mask is None:
            return {'size_category': 'unknown', 'pixel_area': 0}
        
        # Calculate pixel count in mask
        pixel_area = np.sum(mask > 0)
        
        # Determine size category (relative)
        total_pixels = mask.shape[0] * mask.shape[1]
        area_ratio = pixel_area / total_pixels
        
        if area_ratio > 0.6:
            size_category = 'extra_large'
        elif area_ratio > 0.4:
            size_category = 'large'
        elif area_ratio > 0.2:
            size_category = 'medium'
        elif area_ratio > 0.1:
            size_category = 'small'
        else:
            size_category = 'extra_small'
        
        # Bounding box dimensions
        if pixel_area > 0:
            y_coords, x_coords = np.where(mask > 0)
            bbox_height = np.max(y_coords) - np.min(y_coords)
            bbox_width = np.max(x_coords) - np.min(x_coords)
            aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
        else:
            bbox_height = bbox_width = aspect_ratio = 0
        
        return {
            'size_category': size_category,
            'pixel_area': int(pixel_area),
            'area_ratio': round(area_ratio, 3),
            'bbox_height': int(bbox_height),
            'bbox_width': int(bbox_width),
            'aspect_ratio': round(aspect_ratio, 2)
        }
    
    def analyze_texture(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze luggage texture."""
        if mask is not None:
            # Apply mask
            masked_image = image * np.stack([mask] * 3, axis=-1)
            gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
            gray[mask == 0] = 0
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if np.sum(gray > 0) == 0:
            return {'texture_type': 'unknown', 'smoothness': 0}
        
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        if mask is not None:
            edge_values = edge_magnitude[mask > 0]
        else:
            edge_values = edge_magnitude.flatten()
        
        if len(edge_values) == 0:
            edge_mean = 0
        else:
            edge_mean = np.mean(edge_values)
        
        # Determine texture type
        if edge_mean < 20:
            texture_type = 'smooth'
        elif edge_mean < 50:
            texture_type = 'lightly_textured'
        elif edge_mean < 100:
            texture_type = 'moderately_textured'
        else:
            texture_type = 'highly_textured'
        
        # Smoothness measure (lower value = smoother)
        smoothness = round(edge_mean, 2)
        
        return {
            'texture_type': texture_type,
            'smoothness': smoothness,
            'edge_intensity': round(edge_mean, 2)
        }
    
    def detect_brand_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect brand and features (simple approach)."""
        # This function can be extended with advanced OCR and brand recognition algorithms
        
        features = {
            'has_wheels': False,
            'has_handle': False,
            'has_zippers': False,
            'material_type': 'unknown',
            'estimated_brand': 'unknown'
        }
        
        if mask is not None:
            masked_image = image * np.stack([mask] * 3, axis=-1)
            analysis_area = masked_image[mask > 0]
        else:
            analysis_area = image
        
        # Simple feature detection (can be improved)
        # Wheel detection - dark circular shapes
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=50)
        if circles is not None:
            features['has_wheels'] = len(circles[0]) > 0
        
        # Material type estimation (based on color and texture)
        color_analysis = self.analyze_color(image, mask)
        texture_analysis = self.analyze_texture(image, mask)
        
        # Hard case vs soft case estimation
        if texture_analysis['smoothness'] < 30 and color_analysis['dominant_color'] in ['black', 'brown', 'navy']:
            features['material_type'] = 'hard_plastic_ABS'
        elif texture_analysis['smoothness'] > 50:
            features['material_type'] = 'fabric_soft'
        else:
            features['material_type'] = 'mixed_other'
        
        return features


class MultiLuggageAnalyzer:
    """Main class for analyzing multiple luggage photos."""
    
    def __init__(self, similarity_threshold: float = 75.0):
        """
        Args:
            similarity_threshold: Threshold for considering luggage as same (%)
        """
        self.comparator = LuggageComparator()
        self.feature_analyzer = LuggageFeatureAnalyzer()
        self.similarity_threshold = similarity_threshold
        
        self.processed_images = {}
        self.similarity_matrix = None
        self.groups = []
        
    def process_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """Process all images and extract features."""
        print(f"Processing: {len(image_paths)} photos...")
        
        for i, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                print(f"⚠ File not found: {image_path}")
                continue
            
            try:
                # Load image
                image = self.comparator.load_image(image_path)
                
                # SAM segmentation
                mask = None
                if self.comparator.sam_predictor is not None:
                    mask = self.comparator.segment_luggage(image)
                
                # Extract embedding
                embedding = self.comparator.process_image(image_path)
                
                # Analyze features
                color_features = self.feature_analyzer.analyze_color(image, mask)
                size_features = self.feature_analyzer.analyze_size(mask)
                texture_features = self.feature_analyzer.analyze_texture(image, mask)
                brand_features = self.feature_analyzer.detect_brand_features(image, mask)
                
                # Store results
                image_id = f"img_{i:03d}_{os.path.basename(image_path)}"
                self.processed_images[image_id] = {
                    'path': image_path,
                    'embedding': embedding,
                    'features': {
                        'color': color_features,
                        'size': size_features,
                        'texture': texture_features,
                        'brand': brand_features
                    },
                    'mask': mask
                }
                
                print(f"✓ Processed: {image_id}")
                
            except Exception as e:
                print(f"✗ Error ({image_path}): {e}")
        
        print(f"Total processed: {len(self.processed_images)} photos")
        return self.processed_images
    
    def calculate_similarity_matrix(self) -> np.ndarray:
        """Calculate similarity matrix between all photos."""
        image_ids = list(self.processed_images.keys())
        n = len(image_ids)
        
        if n == 0:
            return np.array([])
        
        # Get embeddings
        embeddings = []
        for img_id in image_ids:
            embeddings.append(self.processed_images[img_id]['embedding'])
        
        embeddings = np.array(embeddings)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Convert [-1,1] range to [0,100] percentage
        similarity_percentage = (similarity_matrix + 1) / 2 * 100
        
        self.similarity_matrix = similarity_percentage
        return similarity_percentage
    
    def group_similar_luggage(self) -> List[Dict[str, Any]]:
        """Group similar luggage."""
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()
        
        if len(self.similarity_matrix) == 0:
            return []
        
        image_ids = list(self.processed_images.keys())
        n = len(image_ids)
        
        # Apply threshold for grouping
        similarity_binary = (self.similarity_matrix >= self.similarity_threshold).astype(int)
        
        # Simple grouping algorithm
        visited = set()
        groups = []
        
        for i, img_id in enumerate(image_ids):
            if img_id in visited:
                continue
            
            # Start new group
            group = {
                'group_id': len(groups) + 1,
                'images': [img_id],
                'similarities': {},
                'common_features': {},
                'confidence': 0.0
            }
            
            visited.add(img_id)
            
            # Find other images similar to this one
            for j, other_img_id in enumerate(image_ids):
                if i != j and other_img_id not in visited:
                    similarity = self.similarity_matrix[i][j]
                    if similarity >= self.similarity_threshold:
                        group['images'].append(other_img_id)
                        group['similarities'][other_img_id] = round(similarity, 2)
                        visited.add(other_img_id)
            
            # Calculate group features
            if len(group['images']) > 1:
                group['common_features'] = self._analyze_group_features(group['images'])
                group['confidence'] = self._calculate_group_confidence(group['images'])
                groups.append(group)
        
        self.groups = groups
        return groups
    
    def _analyze_group_features(self, image_ids: List[str]) -> Dict[str, Any]:
        """Analyze common features within the group."""
        if not image_ids:
            return {}
        
        # Collect all features
        colors = []
        sizes = []
        textures = []
        materials = []
        
        for img_id in image_ids:
            features = self.processed_images[img_id]['features']
            colors.append(features['color']['dominant_color'])
            sizes.append(features['size']['size_category'])
            textures.append(features['texture']['texture_type'])
            materials.append(features['brand']['material_type'])
        
        # Find most common features
        common_features = {
            'dominant_color': Counter(colors).most_common(1)[0][0],
            'size_category': Counter(sizes).most_common(1)[0][0],
            'texture_type': Counter(textures).most_common(1)[0][0],
            'material_type': Counter(materials).most_common(1)[0][0],
            'color_consistency': len(set(colors)) == 1,
            'size_consistency': len(set(sizes)) == 1,
            'texture_consistency': len(set(textures)) == 1,
            'material_consistency': len(set(materials)) == 1
        }
        
        # Calculate average values
        avg_area = np.mean([self.processed_images[img_id]['features']['size']['pixel_area'] for img_id in image_ids])
        avg_smoothness = np.mean([self.processed_images[img_id]['features']['texture']['smoothness'] for img_id in image_ids])
        
        common_features.update({
            'average_pixel_area': round(avg_area, 0),
            'average_smoothness': round(avg_smoothness, 2)
        })
        
        return common_features
    
    def _calculate_group_confidence(self, image_ids: List[str]) -> float:
        """Calculate group confidence score."""
        if len(image_ids) < 2:
            return 0.0
        
        # Average intra-group similarity
        similarities = []
        for i in range(len(image_ids)):
            for j in range(i+1, len(image_ids)):
                img1_idx = list(self.processed_images.keys()).index(image_ids[i])
                img2_idx = list(self.processed_images.keys()).index(image_ids[j])
                similarities.append(self.similarity_matrix[img1_idx][img2_idx])
        
        avg_similarity = np.mean(similarities)
        
        # Feature consistency bonus
        common_features = self._analyze_group_features(image_ids)
        consistency_bonus = 0
        for key in ['color_consistency', 'size_consistency', 'texture_consistency', 'material_consistency']:
            if common_features.get(key, False):
                consistency_bonus += 5
        
        confidence = min(100, avg_similarity + consistency_bonus)
        return round(confidence, 2)
    
    def convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types for JSON serialization."""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        return obj
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate detailed analysis report."""
        if not self.groups:
            self.group_similar_luggage()
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_images': len(self.processed_images),
            'similarity_threshold': self.similarity_threshold,
            'total_groups_found': len(self.groups),
            'single_images': len(self.processed_images) - sum(len(group['images']) for group in self.groups),
            'groups': [],
            'individual_images': []
        }
        
        # Add groups to report
        for group in self.groups:
            group_report = {
                'group_id': group['group_id'],
                'image_count': len(group['images']),
                'confidence_score': group['confidence'],
                'images': [],
                'common_features': group['common_features'],
                'similarity_analysis': group['similarities']
            }
            
            # Add details for each image in the group
            for img_id in group['images']:
                img_data = self.processed_images[img_id]
                image_report = {
                    'image_id': img_id,
                    'file_path': img_data['path'],
                    'features': img_data['features'],
                    'similarity_to_others': {}
                }
                
                # Calculate this image's similarity to others in group
                for other_img_id in group['images']:
                    if other_img_id != img_id:
                        img1_idx = list(self.processed_images.keys()).index(img_id)
                        img2_idx = list(self.processed_images.keys()).index(other_img_id)
                        similarity = self.similarity_matrix[img1_idx][img2_idx]
                        image_report['similarity_to_others'][other_img_id] = round(similarity, 2)
                
                group_report['images'].append(image_report)
            
            report['groups'].append(group_report)
        
        # Add single images not in any group
        grouped_images = set()
        for group in self.groups:
            grouped_images.update(group['images'])
        
        for img_id, img_data in self.processed_images.items():
            if img_id not in grouped_images:
                single_image_report = {
                    'image_id': img_id,
                    'file_path': img_data['path'],
                    'features': img_data['features'],
                    'reason_for_isolation': 'No other photos found above similarity threshold'
                }
                report['individual_images'].append(single_image_report)
        
        return report
    
    def save_results(self, output_dir: str = "luggage_analysis_results"):
        """Save results to files."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main report
        report = self.generate_detailed_report()
        # Convert NumPy types to Python native types for JSON serialization
        report = self.convert_numpy_types(report)
        report_path = os.path.join(output_dir, f"luggage_analysis_report_{timestamp}.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Save similarity matrix
        if self.similarity_matrix is not None:
            matrix_path = os.path.join(output_dir, f"similarity_matrix_{timestamp}.csv")
            image_ids = list(self.processed_images.keys())
            
            with open(matrix_path, 'w', encoding='utf-8') as f:
                # Header row
                f.write("Image," + ",".join(image_ids) + "\n")
                
                # Each row
                for i, img_id in enumerate(image_ids):
                    row = [img_id] + [f"{self.similarity_matrix[i][j]:.2f}" for j in range(len(image_ids))]
                    f.write(",".join(row) + "\n")
        
        # Create summary report
        summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
        self._create_summary_report(summary_path, report)
        
        print(f"Results saved:")
        print(f"- Main report: {report_path}")
        print(f"- Similarity matrix: {matrix_path}")
        print(f"- Summary report: {summary_path}")
        
        return {
            'report_path': report_path,
            'matrix_path': matrix_path,
            'summary_path': summary_path
        }
    
    def _create_summary_report(self, filepath: str, report: Dict[str, Any]):
        """Create summary report."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("LUGGAGE PHOTO ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {report['analysis_timestamp']}\n")
            f.write(f"Total Photos: {report['total_images']}\n")
            f.write(f"Similarity Threshold: {report['similarity_threshold']}%\n")
            f.write(f"Groups Found: {report['total_groups_found']}\n")
            f.write(f"Single Photos: {report['single_images']}\n\n")
            
            # Group details
            for group in report['groups']:
                f.write(f"GROUP {group['group_id']} - Confidence: {group['confidence_score']}%\n")
                f.write("-" * 30 + "\n")
                f.write(f"Photo Count: {group['image_count']}\n")
                f.write("Common Features:\n")
                
                cf = group['common_features']
                f.write(f"  - Color: {cf.get('dominant_color', 'Unknown')}\n")
                f.write(f"  - Size: {cf.get('size_category', 'Unknown')}\n")
                f.write(f"  - Texture: {cf.get('texture_type', 'Unknown')}\n")
                f.write(f"  - Material: {cf.get('material_type', 'Unknown')}\n")
                
                f.write("Photos:\n")
                for img in group['images']:
                    img_name = img['image_id']
                    f.write(f"  * {img_name}\n")
                
                f.write("\n")
            
            # Single photos
            if report['individual_images']:
                f.write("INDIVIDUAL PHOTOS\n")
                f.write("-" * 20 + "\n")
                for img in report['individual_images']:
                    f.write(f"- {img['image_id']}\n")


def main():
    """Main usage example."""
    print("Multi Luggage Photo Analysis System")
    print("=" * 40)
    
    # Example usage
    image_paths = [
        # Add paths to photos to analyze here
        # "path/to/luggage1.jpg",
        # "path/to/luggage2.jpg",
        # ...
    ]
    
    if not image_paths:
        print("Usage example:")
        print("analyzer = MultiLuggageAnalyzer(similarity_threshold=75.0)")
        print("analyzer.process_images(['photo1.jpg', 'photo2.jpg', ...])")
        print("analyzer.group_similar_luggage()")
        print("results = analyzer.save_results()")
        return
    
    # Start analysis system
    analyzer = MultiLuggageAnalyzer(similarity_threshold=75.0)
    
    # Process photos
    analyzer.process_images(image_paths)
    
    # Group similar luggage
    groups = analyzer.group_similar_luggage()
    
    # Save results
    results = analyzer.save_results()
    
    print(f"\nAnalysis complete! Found {len(groups)} groups.")


if __name__ == "__main__":
    main()