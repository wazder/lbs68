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
    
    def calculate_color_histogram(self, image: np.ndarray, mask: Optional[np.ndarray] = None, bins: int = 32) -> np.ndarray:
        """Calculate color histogram for similarity comparison."""
        if mask is not None:
            masked_image = image * np.stack([mask] * 3, axis=-1)
            pixels = masked_image[mask > 0]
        else:
            pixels = image.reshape(-1, 3)
        
        if len(pixels) == 0:
            return np.zeros(bins * 3)
        
        # Calculate histogram for each channel
        hist_r = np.histogram(pixels[:, 0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(pixels[:, 1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(pixels[:, 2], bins=bins, range=(0, 256))[0]
        
        # Combine histograms and normalize
        combined_hist = np.concatenate([hist_r, hist_g, hist_b])
        return combined_hist / (np.sum(combined_hist) + 1e-8)
    
    def normalize_lighting(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Normalize lighting to reduce impact of illumination changes."""
        if mask is not None:
            # Convert to LAB color space for better lighting normalization
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # Apply mask to L channel
            masked_l = l_channel * mask
            valid_pixels = masked_l[mask > 0]
            
            if len(valid_pixels) > 0:
                # Normalize L channel
                mean_l = np.mean(valid_pixels)
                std_l = np.std(valid_pixels)
                
                if std_l > 0:
                    # Standardize L channel
                    normalized_l = (l_channel - mean_l) / std_l * 50 + 50
                    normalized_l = np.clip(normalized_l, 0, 100)
                    
                    # Update LAB image
                    lab[:, :, 0] = normalized_l
                    
                    # Convert back to RGB
                    normalized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    return normalized_image
        
        return image
    
    def compare_color_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two color histograms using correlation."""
        return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
    
    def calculate_shape_descriptor(self, mask: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate shape descriptor for similarity comparison."""
        if mask is None or np.sum(mask > 0) == 0:
            return {
                'contour_area': 0,
                'convex_hull_ratio': 0,
                'solidity': 0,
                'extent': 0,
                'roundness': 0,
                'aspect_ratio_category': 'unknown'
            }
        
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'contour_area': 0,
                'convex_hull_ratio': 0,
                'solidity': 0,
                'extent': 0,
                'roundness': 0,
                'aspect_ratio_category': 'unknown'
            }
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape features
        contour_area = cv2.contourArea(largest_contour)
        
        if contour_area > 0:
            # Convex hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            convex_hull_ratio = contour_area / hull_area if hull_area > 0 else 0
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            rect_area = w * h
            extent = contour_area / rect_area if rect_area > 0 else 0
            
            # Solidity
            solidity = contour_area / hull_area if hull_area > 0 else 0
            
            # Roundness (4π*area / perimeter²)
            perimeter = cv2.arcLength(largest_contour, True)
            roundness = (4 * np.pi * contour_area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Aspect ratio category
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 1.8:
                ar_category = 'very_wide'
            elif aspect_ratio > 1.3:
                ar_category = 'wide'
            elif aspect_ratio > 0.7:
                ar_category = 'balanced'
            elif aspect_ratio > 0.5:
                ar_category = 'tall'
            else:
                ar_category = 'very_tall'
        else:
            convex_hull_ratio = extent = solidity = roundness = 0
            ar_category = 'unknown'
        
        return {
            'contour_area': int(contour_area),
            'convex_hull_ratio': round(convex_hull_ratio, 3),
            'solidity': round(solidity, 3),
            'extent': round(extent, 3),
            'roundness': round(roundness, 3),
            'aspect_ratio_category': ar_category
        }
    
    def compare_shape_descriptors(self, desc1: Dict[str, Any], desc2: Dict[str, Any]) -> float:
        """Compare two shape descriptors and return similarity (0-1)."""
        if not desc1 or not desc2:
            return 0.0
        
        # Compare numerical features
        numerical_features = ['convex_hull_ratio', 'solidity', 'extent', 'roundness']
        similarities = []
        
        for feature in numerical_features:
            val1 = desc1.get(feature, 0)
            val2 = desc2.get(feature, 0)
            
            if val1 == 0 and val2 == 0:
                similarity = 1.0
            else:
                max_val = max(val1, val2)
                min_val = min(val1, val2)
                similarity = min_val / max_val if max_val > 0 else 0
            
            similarities.append(similarity)
        
        # Compare aspect ratio category
        ar_similarity = 1.0 if desc1.get('aspect_ratio_category') == desc2.get('aspect_ratio_category') else 0.5
        similarities.append(ar_similarity)
        
        return np.mean(similarities)

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
        
        # Calculate color histogram for similarity comparison
        color_histogram = self.calculate_color_histogram(image, mask)
        
        # Normalize lighting for more robust color analysis
        normalized_image = self.normalize_lighting(image, mask)
        normalized_histogram = self.calculate_color_histogram(normalized_image, mask)
        
        return {
            'dominant_color': closest_color,
            'mean_rgb': mean_color.tolist(),
            'color_distribution': color_counts,
            'color_histogram': color_histogram,
            'normalized_histogram': normalized_histogram
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
        
        # Calculate shape descriptor
        shape_descriptor = self.calculate_shape_descriptor(mask)
        
        return {
            'size_category': size_category,
            'pixel_area': int(pixel_area),
            'area_ratio': round(area_ratio, 3),
            'bbox_height': int(bbox_height),
            'bbox_width': int(bbox_width),
            'aspect_ratio': round(aspect_ratio, 2),
            'shape_descriptor': shape_descriptor
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
    
    def __init__(self, similarity_threshold: float = 85.0):
        """
        Args:
            similarity_threshold: Main threshold for considering luggage as same (%)
        """
        self.comparator = LuggageComparator()
        self.feature_analyzer = LuggageFeatureAnalyzer()
        self.similarity_threshold = similarity_threshold
        
        # Dynamic thresholds for different similarity types
        self.thresholds = {
            'strict': similarity_threshold,              # 85% - strict matching
            'color_robust': similarity_threshold - 10,   # 75% - when colors might vary due to lighting
            'shape_based': similarity_threshold - 15,    # 70% - when shape is primary indicator
            'fallback': similarity_threshold - 20        # 65% - fallback for difficult cases
        }
        
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
                
                # STEP 1: Detect if this is luggage
                luggage_detection = self.comparator.detect_luggage(image, threshold=0.7)
                
                if not luggage_detection['is_luggage']:
                    print(f"⚠ Skipped (not luggage): {os.path.basename(image_path)} - {luggage_detection['reason']}")
                    continue
                
                print(f"✓ Luggage detected: {os.path.basename(image_path)} (confidence: {luggage_detection['confidence']:.1%})")
                
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
                    'luggage_detection': luggage_detection,
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
    
    def calculate_multi_level_similarity(self, img1_id: str, img2_id: str) -> Dict[str, float]:
        """Calculate multi-level similarity between two images."""
        img1_data = self.processed_images[img1_id]
        img2_data = self.processed_images[img2_id]
        
        # 1. CLIP embedding similarity (main similarity)
        embedding1 = img1_data['embedding']
        embedding2 = img2_data['embedding']
        clip_similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        clip_percentage = (clip_similarity + 1) / 2 * 100
        
        # 2. Color histogram similarity
        hist1 = img1_data['features']['color'].get('normalized_histogram', np.array([]))
        hist2 = img2_data['features']['color'].get('normalized_histogram', np.array([]))
        
        if len(hist1) > 0 and len(hist2) > 0:
            color_similarity = self.feature_analyzer.compare_color_histograms(hist1, hist2)
            color_percentage = max(0, color_similarity * 100)  # Convert correlation to percentage
        else:
            color_percentage = 0
        
        # 3. Shape similarity
        shape1 = img1_data['features']['size'].get('shape_descriptor', {})
        shape2 = img2_data['features']['size'].get('shape_descriptor', {})
        shape_similarity = self.feature_analyzer.compare_shape_descriptors(shape1, shape2)
        shape_percentage = shape_similarity * 100
        
        # 4. Texture similarity (based on smoothness)
        smooth1 = img1_data['features']['texture'].get('smoothness', 0)
        smooth2 = img2_data['features']['texture'].get('smoothness', 0)
        
        if smooth1 > 0 and smooth2 > 0:
            max_smooth = max(smooth1, smooth2)
            min_smooth = min(smooth1, smooth2)
            texture_similarity = min_smooth / max_smooth
            texture_percentage = texture_similarity * 100
        else:
            texture_percentage = 0
        
        return {
            'clip_similarity': round(clip_percentage, 2),
            'color_similarity': round(color_percentage, 2),
            'shape_similarity': round(shape_percentage, 2),
            'texture_similarity': round(texture_percentage, 2)
        }
    
    def calculate_combined_similarity(self, similarities: Dict[str, float]) -> float:
        """Calculate weighted combined similarity score."""
        # Weights for different similarity types
        weights = {
            'clip_similarity': 0.5,    # Main visual similarity
            'color_similarity': 0.2,   # Color consistency
            'shape_similarity': 0.2,   # Shape consistency
            'texture_similarity': 0.1  # Texture consistency
        }
        
        combined_score = 0
        total_weight = 0
        
        for sim_type, weight in weights.items():
            if sim_type in similarities and similarities[sim_type] > 0:
                combined_score += similarities[sim_type] * weight
                total_weight += weight
        
        return combined_score / total_weight if total_weight > 0 else 0
    
    def determine_matching_threshold(self, img1_id: str, img2_id: str, similarities: Dict[str, float]) -> Tuple[str, float]:
        """Determine which threshold to use based on similarity characteristics."""
        img1_data = self.processed_images[img1_id]
        img2_data = self.processed_images[img2_id]
        
        # Check if colors are very different (might indicate lighting differences)
        color1 = img1_data['features']['color']['dominant_color']
        color2 = img2_data['features']['color']['dominant_color']
        
        # Check if shapes are very similar even if colors differ
        shape_sim = similarities.get('shape_similarity', 0)
        color_sim = similarities.get('color_similarity', 0)
        clip_sim = similarities.get('clip_similarity', 0)
        
        # Decision logic for threshold selection
        if clip_sim >= self.thresholds['strict']:
            return 'strict', self.thresholds['strict']
        
        elif shape_sim >= 80 and color1 != color2:
            # Same shape, different color -> likely lighting issue
            return 'color_robust', self.thresholds['color_robust']
        
        elif shape_sim >= 85:
            # Very similar shapes
            return 'shape_based', self.thresholds['shape_based']
        
        elif (shape_sim >= 70 and clip_sim >= 65) or (color_sim >= 70 and clip_sim >= 65):
            # Moderate similarity in multiple dimensions
            return 'fallback', self.thresholds['fallback']
        
        else:
            return 'strict', self.thresholds['strict']

    def calculate_similarity_matrix(self) -> np.ndarray:
        """Calculate multi-level similarity matrix between all photos."""
        image_ids = list(self.processed_images.keys())
        n = len(image_ids)
        
        if n == 0:
            return np.array([])
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((n, n))
        
        # Calculate similarities
        print("Calculating multi-level similarities...")
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 100.0  # Same image
                elif i < j:  # Calculate only upper triangle
                    similarities = self.calculate_multi_level_similarity(image_ids[i], image_ids[j])
                    combined_sim = self.calculate_combined_similarity(similarities)
                    similarity_matrix[i][j] = combined_sim
                    similarity_matrix[j][i] = combined_sim  # Symmetric matrix
        
        self.similarity_matrix = similarity_matrix
        return similarity_matrix
    
    def group_similar_luggage(self) -> List[Dict[str, Any]]:
        """Group similar luggage using two-phase matching."""
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()
        
        if len(self.similarity_matrix) == 0:
            return []
        
        image_ids = list(self.processed_images.keys())
        n = len(image_ids)
        
        print("Starting two-phase grouping...")
        
        # PHASE 1: Strict grouping
        print("Phase 1: Strict grouping...")
        groups_phase1 = self._group_with_strict_threshold(image_ids)
        
        # PHASE 2: Relaxed grouping for remaining images
        print("Phase 2: Relaxed grouping for remaining items...")
        grouped_ids_phase1 = set()
        for group in groups_phase1:
            grouped_ids_phase1.update(group['images'])
        
        remaining_ids = [img_id for img_id in image_ids if img_id not in grouped_ids_phase1]
        groups_phase2 = self._group_with_dynamic_thresholds(remaining_ids, grouped_ids_phase1)
        
        # Combine results
        all_groups = groups_phase1 + groups_phase2
        
        # Renumber groups
        for i, group in enumerate(all_groups, 1):
            group['group_id'] = i
        
        self.groups = all_groups
        print(f"Grouping complete: {len(groups_phase1)} strict groups + {len(groups_phase2)} relaxed groups")
        return all_groups
    
    def _group_with_strict_threshold(self, image_ids: List[str]) -> List[Dict[str, Any]]:
        """Phase 1: Group with strict threshold."""
        visited = set()
        groups = []
        
        for i, img_id in enumerate(image_ids):
            if img_id in visited:
                continue
            
            group = {
                'group_id': len(groups) + 1,
                'images': [img_id],
                'similarities': {},
                'common_features': {},
                'confidence': 0.0,
                'matching_type': 'strict'
            }
            
            visited.add(img_id)
            
            # Find other images similar to this one using strict threshold
            for j, other_img_id in enumerate(image_ids):
                if i != j and other_img_id not in visited:
                    similarity = self.similarity_matrix[i][j]
                    if similarity >= self.thresholds['strict']:
                        group['images'].append(other_img_id)
                        group['similarities'][other_img_id] = round(similarity, 2)
                        visited.add(other_img_id)
            
            # Only keep groups with more than 1 image
            if len(group['images']) > 1:
                group['common_features'] = self._analyze_group_features(group['images'])
                group['confidence'] = self._calculate_group_confidence(group['images'])
                group['explanation'] = self._generate_group_explanation(group)
                groups.append(group)
        
        return groups
    
    def _group_with_dynamic_thresholds(self, remaining_ids: List[str], already_grouped: set) -> List[Dict[str, Any]]:
        """Phase 2: Group remaining images with dynamic thresholds."""
        if not remaining_ids:
            return []
        
        visited = set()
        groups = []
        
        for i, img_id in enumerate(remaining_ids):
            if img_id in visited:
                continue
            
            group = {
                'group_id': len(groups) + 1,
                'images': [img_id],
                'similarities': {},
                'detailed_similarities': {},
                'common_features': {},
                'confidence': 0.0,
                'matching_type': 'dynamic'
            }
            
            visited.add(img_id)
            
            # Find other images using dynamic thresholds
            for j, other_img_id in enumerate(remaining_ids):
                if i != j and other_img_id not in visited:
                    # Calculate detailed similarities
                    similarities = self.calculate_multi_level_similarity(img_id, other_img_id)
                    combined_sim = self.calculate_combined_similarity(similarities)
                    
                    # Determine appropriate threshold
                    threshold_type, threshold_value = self.determine_matching_threshold(img_id, other_img_id, similarities)
                    
                    if combined_sim >= threshold_value:
                        group['images'].append(other_img_id)
                        group['similarities'][other_img_id] = round(combined_sim, 2)
                        group['detailed_similarities'][other_img_id] = {
                            'combined': round(combined_sim, 2),
                            'threshold_used': threshold_type,
                            'threshold_value': threshold_value,
                            'breakdown': similarities
                        }
                        visited.add(other_img_id)
            
            # Only keep groups with more than 1 image
            if len(group['images']) > 1:
                group['common_features'] = self._analyze_group_features(group['images'])
                group['confidence'] = self._calculate_group_confidence(group['images'])
                group['explanation'] = self._generate_group_explanation(group)
                groups.append(group)
        
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
    
    def _generate_group_explanation(self, group: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for why items are grouped together."""
        if len(group['images']) < 2:
            return {'reason': 'Single item group', 'details': []}
        
        explanations = []
        common_features = group['common_features']
        matching_type = group.get('matching_type', 'unknown')
        
        # Add matching type explanation
        if matching_type == 'strict':
            explanations.append(f"Matched with strict threshold ({self.thresholds['strict']}%)")
        elif matching_type == 'dynamic':
            explanations.append("Matched with dynamic thresholds (adaptive similarity detection)")
        
        # Color similarity
        if common_features.get('color_consistency', False):
            color = common_features.get('dominant_color', 'unknown')
            explanations.append(f"Same color: {color}")
        elif matching_type == 'dynamic':
            explanations.append("Color differences compensated by lighting normalization")
        
        # Size similarity  
        if common_features.get('size_consistency', False):
            size = common_features.get('size_category', 'unknown')
            explanations.append(f"Same size category: {size}")
        
        # Texture similarity
        if common_features.get('texture_consistency', False):
            texture = common_features.get('texture_type', 'unknown')
            explanations.append(f"Same texture: {texture}")
        
        # Material similarity
        if common_features.get('material_consistency', False):
            material = common_features.get('material_type', 'unknown')
            explanations.append(f"Same material: {material}")
        
        # Visual similarity
        similarities = list(group['similarities'].values())
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            max_sim = max(similarities)
            explanations.append(f"Visual similarity: {avg_sim:.1f}% average, {max_sim:.1f}% maximum")
        
        # Detailed similarity breakdown for dynamic matching
        if matching_type == 'dynamic' and 'detailed_similarities' in group:
            for img_id, details in group['detailed_similarities'].items():
                breakdown = details['breakdown']
                threshold_used = details['threshold_used']
                explanations.append(f"Advanced analysis used {threshold_used} threshold:")
                explanations.append(f"  - CLIP: {breakdown.get('clip_similarity', 0):.1f}%")
                explanations.append(f"  - Color: {breakdown.get('color_similarity', 0):.1f}%") 
                explanations.append(f"  - Shape: {breakdown.get('shape_similarity', 0):.1f}%")
                explanations.append(f"  - Texture: {breakdown.get('texture_similarity', 0):.1f}%")
                break  # Just show one example
        
        # CLIP detection consistency
        detection_types = []
        for img_id in group['images']:
            detection = self.processed_images[img_id].get('luggage_detection', {})
            best_match = detection.get('best_match', '')
            if best_match:
                detection_types.append(best_match)
        
        if detection_types:
            # Find most common detection type
            detection_counter = Counter(detection_types)
            most_common = detection_counter.most_common(1)[0]
            if most_common[1] > 1:  # If more than one image has this detection
                explanations.append(f"Detected as similar object type: {most_common[0]}")
        
        return {
            'reason': f'Items grouped using {matching_type} matching with multiple similarity factors',
            'details': explanations,
            'similarity_scores': group['similarities'],
            'matching_approach': matching_type
        }
    
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
        print("analyzer = MultiLuggageAnalyzer(similarity_threshold=85.0)")
        print("analyzer.process_images(['photo1.jpg', 'photo2.jpg', ...])")
        print("analyzer.group_similar_luggage()")
        print("results = analyzer.save_results()")
        return
    
    # Start analysis system
    analyzer = MultiLuggageAnalyzer(similarity_threshold=85.0)
    
    # Process photos
    analyzer.process_images(image_paths)
    
    # Group similar luggage
    groups = analyzer.group_similar_luggage()
    
    # Save results
    results = analyzer.save_results()
    
    print(f"\nAnalysis complete! Found {len(groups)} groups.")


if __name__ == "__main__":
    main()