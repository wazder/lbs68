#!/usr/bin/env python3
"""
MAIN LUGGAGE ANALYZER
Advanced luggage grouping with maximum precision
"""

import os
import json
import numpy as np
import cv2
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import torch
from PIL import Image
import faiss
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from luggage_comparator import LuggageComparator
from utils import get_image_files, setup_logging

class LuggageAnalyzer:
    """
    Main Luggage Analyzer with Advanced Precision
    """
    
    def __init__(self, similarity_threshold: float = 85.0):
        self.logger = setup_logging()
        self.comparator = LuggageComparator()
        self.processed_images = {}
        self.groups = []
        self.similarity_threshold = similarity_threshold
        
        # Advanced analysis settings
        self.multi_level_similarity = True
        self.geometric_verification = True  
        self.color_analysis = True
        self.shape_analysis = True
        self.texture_analysis = True
        self.keypoint_matching = True
        self.adaptive_threshold = True
    
    def analyze_images(self, image_paths: List[str], threshold: float = None) -> Dict[str, Any]:
        """Main analysis method - process and group images."""
        if threshold is not None:
            self.similarity_threshold = threshold
            
        self.logger.info(f"LUGGAGE ANALYSIS STARTING: {len(image_paths)} images, threshold: {self.similarity_threshold}%")
        
        # Process all images
        self.process_images(image_paths)
        
        # Group similar images
        self.group_with_ultra_precision(self.similarity_threshold, self.adaptive_threshold)
        
        # Return results
        return {
            'groups': self.groups,
            'total_photos': len(self.processed_images),
            'processed_images': {k: {'path': v['path']} for k, v in self.processed_images.items()},
            'analysis_date': datetime.now().isoformat()
        }
        
    def group_similar_luggage(self):
        """Compatibility method for old interface."""
        return self.group_with_ultra_precision(self.similarity_threshold, self.adaptive_threshold)
        
    def process_images(self, image_paths: List[str]):
        """Advanced precision image processing."""
        self.logger.info("ADVANCED PRECISION PROCESSING STARTING")
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Load image
            image = self.comparator.load_image(image_path)
            
            # Multi-level feature extraction
            features = self._extract_ultra_features(image, image_path)
            
            # Store with unique ID
            image_id = f"img_{i:03d}_{os.path.basename(image_path)}"
            self.processed_images[image_id] = {
                'path': image_path,
                'features': features,
                'embedding': features['clip_embedding']
            }
    
    def _extract_ultra_features(self, image: np.ndarray, image_path: str) -> Dict[str, Any]:
        """Advanced precision feature extraction."""
        features = {}
        
        # 1. CLIP Embedding
        features['clip_embedding'] = self.comparator.process_image(image_path)
        
        # 2. SAM Segmentation
        if self.comparator.sam_predictor is not None:
            mask = self.comparator.segment_luggage(image)
            features['mask'] = mask
            features['mask_area'] = np.sum(mask > 0)
        else:
            features['mask'] = None
            features['mask_area'] = 0
        
        # 3. Color Analysis (Multi-space)
        features['color'] = self._analyze_color_ultra(image, features.get('mask'))
        
        # 4. Shape Analysis
        features['shape'] = self._analyze_shape_ultra(image, features.get('mask'))
        
        # 5. Texture Analysis
        features['texture'] = self._analyze_texture_ultra(image, features.get('mask'))
        
        # 6. Edge Analysis
        features['edges'] = self._analyze_edges_ultra(image, features.get('mask'))
        
        # 7. Histogram Analysis
        features['histogram'] = self._analyze_histogram_ultra(image, features.get('mask'))
        
        return features
    
    def _analyze_color_ultra(self, image: np.ndarray, mask: np.ndarray = None) -> Dict[str, Any]:
        """Ultra-precision color analysis."""
        if mask is not None:
            masked_image = image * np.stack([mask] * 3, axis=-1)
            analysis_image = masked_image
        else:
            analysis_image = image
        
        # RGB Analysis
        rgb_hist = cv2.calcHist([analysis_image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        rgb_hist = cv2.normalize(rgb_hist, rgb_hist).flatten()
        
        # HSV Analysis
        hsv_image = cv2.cvtColor(analysis_image, cv2.COLOR_RGB2HSV)
        hsv_hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [32, 32, 32], [0, 180, 0, 256, 0, 256])
        hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()
        
        # LAB Analysis
        lab_image = cv2.cvtColor(analysis_image, cv2.COLOR_RGB2LAB)
        lab_hist = cv2.calcHist([lab_image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        lab_hist = cv2.normalize(lab_hist, lab_hist).flatten()
        
        # Dominant color detection
        pixels = analysis_image.reshape(-1, 3)
        if mask is not None:
            pixels = pixels[mask.flatten() > 0]
        
        if len(pixels) > 0:
            # K-means clustering for dominant colors
            pixels_float = pixels.astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels_float, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Get dominant color
            unique, counts = np.unique(labels, return_counts=True)
            dominant_idx = unique[np.argmax(counts)]
            dominant_color = centers[dominant_idx].astype(int)
        else:
            dominant_color = [0, 0, 0]
        
        return {
            'rgb_histogram': rgb_hist,
            'hsv_histogram': hsv_hist,
            'lab_histogram': lab_hist,
            'dominant_color': dominant_color.tolist(),
            'color_variance': np.var(pixels) if len(pixels) > 0 else 0
        }
    
    def _analyze_shape_ultra(self, image: np.ndarray, mask: np.ndarray = None) -> Dict[str, Any]:
        """Ultra-precision shape analysis."""
        if mask is None or np.sum(mask > 0) == 0:
            return {'contours': [], 'area': 0, 'perimeter': 0, 'circularity': 0}
        
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'contours': [], 'area': 0, 'perimeter': 0, 'circularity': 0}
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'bounding_box': [x, y, w, h],
            'contour_points': largest_contour.tolist()
        }
    
    def _analyze_texture_ultra(self, image: np.ndarray, mask: np.ndarray = None) -> Dict[str, Any]:
        """Ultra-precision texture analysis."""
        if mask is not None:
            gray = cv2.cvtColor(image * np.stack([mask] * 3, axis=-1), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # GLCM-like features
        # Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Texture features
        edge_mean = np.mean(edge_magnitude)
        edge_std = np.std(edge_magnitude)
        laplacian_mean = np.mean(laplacian)
        laplacian_std = np.std(laplacian)
        
        return {
            'edge_mean': edge_mean,
            'edge_std': edge_std,
            'laplacian_mean': laplacian_mean,
            'laplacian_std': laplacian_std,
            'texture_energy': np.sum(edge_magnitude**2),
            'texture_entropy': -np.sum(edge_magnitude * np.log(edge_magnitude + 1e-8))
        }
    
    def _analyze_edges_ultra(self, image: np.ndarray, mask: np.ndarray = None) -> Dict[str, Any]:
        """Ultra-precision edge analysis."""
        if mask is not None:
            analysis_image = image * np.stack([mask] * 3, axis=-1)
        else:
            analysis_image = image
        
        gray = cv2.cvtColor(analysis_image, cv2.COLOR_RGB2GRAY)
        
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge features
        edge_density = np.sum(edges > 0) / edges.size
        edge_orientation = self._analyze_edge_orientation(edges)
        
        return {
            'edge_density': edge_density,
            'edge_orientation': edge_orientation,
            'edge_image': edges.tolist()
        }
    
    def _analyze_edge_orientation(self, edges: np.ndarray) -> Dict[str, float]:
        """Analyze edge orientation distribution."""
        # Simple orientation analysis
        h, w = edges.shape
        horizontal_edges = np.sum(edges[:, :w//2]) / np.sum(edges)
        vertical_edges = np.sum(edges[:h//2, :]) / np.sum(edges)
        
        return {
            'horizontal_ratio': horizontal_edges,
            'vertical_ratio': vertical_edges,
            'diagonal_ratio': 1 - horizontal_edges - vertical_edges
        }
    
    def _analyze_histogram_ultra(self, image: np.ndarray, mask: np.ndarray = None) -> Dict[str, Any]:
        """Ultra-precision histogram analysis."""
        if mask is not None:
            analysis_image = image * np.stack([mask] * 3, axis=-1)
        else:
            analysis_image = image
        
        # Multi-channel histograms
        histograms = {}
        for i, channel in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([analysis_image], [i], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms[channel] = hist.tolist()
        
        # Statistical features
        mean_color = np.mean(analysis_image, axis=(0, 1))
        std_color = np.std(analysis_image, axis=(0, 1))
        
        return {
            'histograms': histograms,
            'mean_color': mean_color.tolist(),
            'std_color': std_color.tolist(),
            'color_skewness': self._calculate_skewness(analysis_image)
        }
    
    def _calculate_skewness(self, image: np.ndarray) -> List[float]:
        """Calculate color channel skewness."""
        skewness = []
        for i in range(3):
            channel = image[:, :, i].flatten()
            mean = np.mean(channel)
            std = np.std(channel)
            if std > 0:
                skew = np.mean(((channel - mean) / std) ** 3)
            else:
                skew = 0
            skewness.append(skew)
        return skewness
    
    def calculate_ultra_similarity(self, img1_id: str, img2_id: str) -> float:
        """Ultra-precision similarity calculation."""
        img1_data = self.processed_images[img1_id]
        img2_data = self.processed_images[img2_id]
        
        similarities = {}
        
        # 1. CLIP Embedding Similarity (40% weight)
        clip_sim = cosine_similarity([img1_data['embedding']], [img2_data['embedding']])[0][0]
        similarities['clip'] = (clip_sim + 1) / 2 * 100
        
        # 2. Color Similarity (25% weight)
        color_sim = self._calculate_color_similarity(img1_data['features']['color'], img2_data['features']['color'])
        similarities['color'] = color_sim
        
        # 3. Shape Similarity (20% weight)
        shape_sim = self._calculate_shape_similarity(img1_data['features']['shape'], img2_data['features']['shape'])
        similarities['shape'] = shape_sim
        
        # 4. Texture Similarity (10% weight)
        texture_sim = self._calculate_texture_similarity(img1_data['features']['texture'], img2_data['features']['texture'])
        similarities['texture'] = texture_sim
        
        # 5. Edge Similarity (5% weight)
        edge_sim = self._calculate_edge_similarity(img1_data['features']['edges'], img2_data['features']['edges'])
        similarities['edges'] = edge_sim
        
        # SIFT keypoints disabled for speed - causing hang
        # sift_sim = self._calculate_keypoint_similarity(img1_data, img2_data) 
        # similarities['keypoints'] = sift_sim
        
        # Pure visual similarity weighting - heavy emphasis on CLIP
        # CLIP embeddings are most reliable for identical luggage detection
        weights = {'clip': 0.70, 'color': 0.20, 'shape': 0.08, 'texture': 0.015, 'edges': 0.005}
        final_similarity = sum(similarities[key] * weights[key] for key in weights.keys())
        
        return final_similarity
    
    def _calculate_color_similarity(self, color1: Dict, color2: Dict) -> float:
        """Calculate color similarity."""
        # RGB histogram similarity
        rgb_sim = cv2.compareHist(
            np.array(color1['rgb_histogram'], dtype=np.float32),
            np.array(color2['rgb_histogram'], dtype=np.float32),
            cv2.HISTCMP_CORREL
        )
        
        # HSV histogram similarity
        hsv_sim = cv2.compareHist(
            np.array(color1['hsv_histogram'], dtype=np.float32),
            np.array(color2['hsv_histogram'], dtype=np.float32),
            cv2.HISTCMP_CORREL
        )
        
        # Dominant color similarity (more strict)
        color_diff = np.linalg.norm(
            np.array(color1['dominant_color']) - np.array(color2['dominant_color'])
        )
        dom_color_sim = 1 - (color_diff / (255 * np.sqrt(3)))
        
        # More weight to dominant color for luggage matching
        return (rgb_sim * 0.2 + hsv_sim * 0.2 + dom_color_sim * 0.6) * 100
    
    def _calculate_shape_similarity(self, shape1: Dict, shape2: Dict) -> float:
        """Calculate shape similarity."""
        if shape1['area'] == 0 or shape2['area'] == 0:
            return 0
        
        # Area similarity
        area_sim = min(shape1['area'], shape2['area']) / max(shape1['area'], shape2['area'])
        
        # Aspect ratio similarity
        ar_sim = min(shape1['aspect_ratio'], shape2['aspect_ratio']) / max(shape1['aspect_ratio'], shape2['aspect_ratio'])
        
        # Circularity similarity
        circ_sim = min(shape1['circularity'], shape2['circularity']) / max(shape1['circularity'], shape2['circularity'])
        
        return (area_sim + ar_sim + circ_sim) / 3 * 100
    
    def _calculate_texture_similarity(self, texture1: Dict, texture2: Dict) -> float:
        """Calculate texture similarity."""
        # Edge mean similarity
        edge_mean_sim = 1 - abs(texture1['edge_mean'] - texture2['edge_mean']) / max(texture1['edge_mean'], texture2['edge_mean'])
        
        # Texture energy similarity
        energy_sim = min(texture1['texture_energy'], texture2['texture_energy']) / max(texture1['texture_energy'], texture2['texture_energy'])
        
        return (edge_mean_sim + energy_sim) / 2 * 100
    
    def _calculate_edge_similarity(self, edges1: Dict, edges2: Dict) -> float:
        """Calculate edge similarity."""
        # Edge density similarity
        density_sim = 1 - abs(edges1['edge_density'] - edges2['edge_density'])
        
        return density_sim * 100
    
    def _calculate_keypoint_similarity(self, img1_data: Dict, img2_data: Dict) -> float:
        """SIFT keypoint similarity - DISABLED for performance."""
        # DISABLED: Too slow, causing system hang
        return 0
    
    def group_with_ultra_precision(self, threshold: float = 85.0, adaptive=True):
        """Ultra-precision grouping."""
        self.logger.info(f"ULTRA-PRECISION GROUPING STARTING with PURE VISUAL SIMILARITY v4.0")
        self.logger.info(f"Using pure visual similarity threshold: {threshold:.1f}%")
        
        image_ids = list(self.processed_images.keys())
        n_images = len(image_ids)
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((n_images, n_images))
        
        for i in range(n_images):
            for j in range(i+1, n_images):
                similarity = self.calculate_ultra_similarity(image_ids[i], image_ids[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Adaptive threshold based on similarity distribution
        if adaptive:
            similarities_flat = similarity_matrix[similarity_matrix > 0].flatten()
            mean_sim = np.mean(similarities_flat)
            std_sim = np.std(similarities_flat)
            
            # Adjust threshold based on data distribution
            adaptive_threshold = min(threshold, mean_sim + 1.5 * std_sim)
            # self.logger.info(f"Adaptive threshold: {adaptive_threshold:.1f}% (original: {threshold:.1f}%)")
            threshold = adaptive_threshold
        
        # Create groups using strict threshold-based approach 
        self.groups = []
        used_images = set()
        
        for i in range(n_images):
            if image_ids[i] in used_images:
                continue
            
            # Start new group with this image
            current_group = [image_ids[i]]
            used_images.add(image_ids[i])
            
            # Find all images similar enough to this one using STRICTER criteria
            for j in range(i+1, n_images):
                if image_ids[j] in used_images:
                    continue
                
                # STRICTER APPROACH: Must be similar to ALL images in current group
                # This prevents cross-contamination between different luggage types
                similarities_to_group = []
                for group_img_id in current_group:
                    group_idx = image_ids.index(group_img_id)
                    similarity = similarity_matrix[group_idx, j]
                    similarities_to_group.append(similarity)
                
                # Must meet minimum similarity to ALL group members
                min_similarity = min(similarities_to_group)
                avg_similarity = np.mean(similarities_to_group)
                
                # PURE VISUAL MODE: More flexible for same luggage from different angles
                min_threshold = threshold * 0.80  # 80% of threshold for weakest link
                avg_threshold = threshold * 0.90  # 90% of threshold for average
                
                if min_similarity >= min_threshold and avg_similarity >= avg_threshold:
                    current_group.append(image_ids[j])
                    used_images.add(image_ids[j])
            
            # Create group even with single images (they might be unique luggage)
            group_similarities = []
            if len(current_group) > 1:
                for g_i, img1 in enumerate(current_group):
                    for g_j, img2 in enumerate(current_group):
                        if g_i < g_j:
                            idx1 = image_ids.index(img1)
                            idx2 = image_ids.index(img2)
                            group_similarities.append(similarity_matrix[idx1, idx2])
            
            group = {
                'images': current_group,
                'confidence': np.mean(group_similarities) if group_similarities else 100.0,
                'similarities': {},
                'common_features': self._analyze_group_features(current_group)
            }
            self.groups.append(group)
        
        # Ensure ALL images are included - create single-image groups for remaining images
        all_grouped_images = set()
        for group in self.groups:
            all_grouped_images.update(group['images'])
        
        # Add ungrouped images as individual groups
        for image_id in image_ids:
            if image_id not in all_grouped_images:
                single_group = {
                    'images': [image_id],
                    'confidence': 100.0,  # Perfect confidence for unique items
                    'similarities': {},
                    'common_features': self._analyze_group_features([image_id])
                }
                self.groups.append(single_group)
                self.logger.info(f"Added ungrouped image as individual group: {image_id}")
        
        # Post-processing: Conservative merging to maintain group separation
        self._merge_similar_groups(threshold * 1.05)
        
        # Final verification - ensure we have all images
        final_grouped_images = set()
        for group in self.groups:
            final_grouped_images.update(group['images'])
        
        if len(final_grouped_images) != len(image_ids):
            missing = set(image_ids) - final_grouped_images
            self.logger.warning(f"Missing images detected: {missing}")
        
        self.logger.info(f"ULTRA-PRECISION GROUPING COMPLETED: {len(self.groups)} groups found, {len(final_grouped_images)} images processed")
    
    def _merge_similar_groups(self, merge_threshold: float):
        """Merge groups that are very similar to each other."""
        if len(self.groups) < 2:
            return
        
        merged = True
        while merged:
            merged = False
            for i in range(len(self.groups)):
                for j in range(i + 1, len(self.groups)):
                    if i >= len(self.groups) or j >= len(self.groups):
                        continue
                    
                    # Calculate inter-group similarity
                    group1_images = self.groups[i]['images']
                    group2_images = self.groups[j]['images']
                    
                    inter_similarities = []
                    for img1 in group1_images:
                        for img2 in group2_images:
                            sim = self.calculate_ultra_similarity(img1, img2)
                            inter_similarities.append(sim)
                    
                    avg_inter_sim = np.mean(inter_similarities)
                    
                    # If groups are very similar, merge them
                    if avg_inter_sim >= merge_threshold:
                        # Merge group j into group i
                        self.groups[i]['images'].extend(group2_images)
                        
                        # Recalculate confidence
                        all_images = self.groups[i]['images']
                        all_sims = []
                        for gi in range(len(all_images)):
                            for gj in range(gi + 1, len(all_images)):
                                sim = self.calculate_ultra_similarity(all_images[gi], all_images[gj])
                                all_sims.append(sim)
                        
                        self.groups[i]['confidence'] = np.mean(all_sims) if all_sims else 0
                        self.groups[i]['common_features'] = self._analyze_group_features(all_images)
                        
                        # Remove merged group
                        del self.groups[j]
                        merged = True
                        # self.logger.info(f"Merged similar groups (similarity: {avg_inter_sim:.1f}%)")
                        break
                
                if merged:
                    break
    
    def _analyze_group_features(self, image_ids: List[str]) -> Dict[str, Any]:
        """Analyze common features of a group."""
        if not image_ids:
            return {}
        
        # Get all features
        all_features = [self.processed_images[img_id]['features'] for img_id in image_ids]
        
        # Analyze common features
        common_features = {
            'dominant_color': self._get_common_color([f['color']['dominant_color'] for f in all_features]),
            'size_category': self._get_common_size([f['shape']['area'] for f in all_features]),
            'texture_type': self._get_common_texture([f['texture']['edge_mean'] for f in all_features]),
            'material_type': 'mixed_other'  # Default
        }
        
        return common_features
    
    def _get_common_color(self, colors: List[List[int]]) -> str:
        """Get common color category with more precision."""
        # More precise color categorization
        avg_color = np.mean(colors, axis=0)
        r, g, b = avg_color
        
        # Convert to HSV for better color classification
        rgb_normalized = avg_color / 255.0
        hsv = cv2.cvtColor(np.uint8([[rgb_normalized * 255]]), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv
        
        # More detailed color classification
        if v < 30:  # Very dark
            return 'black'
        elif v > 220 and s < 30:  # Very light
            return 'white'
        elif s < 30:  # Low saturation - grays
            return 'gray'
        elif h < 15 or h > 165:  # Red range
            return 'red'
        elif 15 <= h < 45:  # Orange/Yellow range
            return 'orange_yellow'
        elif 45 <= h < 75:  # Green range
            return 'green'
        elif 75 <= h < 105:  # Cyan range
            return 'cyan'
        elif 105 <= h < 135:  # Blue range
            return 'blue'
        elif 135 <= h <= 165:  # Purple/Magenta range
            return 'purple'
        else:
            return 'mixed'
    
    def _get_common_size(self, areas: List[float]) -> str:
        """Get common size category."""
        avg_area = np.mean(areas)
        
        if avg_area < 10000:
            return 'extra_small'
        elif avg_area < 50000:
            return 'small'
        elif avg_area < 100000:
            return 'medium'
        elif avg_area < 200000:
            return 'large'
        else:
            return 'extra_large'
    
    def _get_common_texture(self, edge_means: List[float]) -> str:
        """Get common texture category."""
        avg_edge = np.mean(edge_means)
        
        if avg_edge < 20:
            return 'smooth'
        elif avg_edge < 50:
            return 'lightly_textured'
        elif avg_edge < 100:
            return 'moderately_textured'
        else:
            return 'highly_textured'
    
    def save_ultra_results(self, output_dir: str = "output"):
        """Save ultra-precision results."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed report
        results = {
            'analysis_date': datetime.now().isoformat(),
            'total_photos': len(self.processed_images),
            'groups': self.groups,
            'processed_images': {k: {'path': v['path']} for k, v in self.processed_images.items()}
        }
        
        json_file = f"{output_dir}/ultra_precision_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = f"{output_dir}/ultra_precision_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("ULTRA-PRECISION LUGGAGE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().isoformat()}\n")
            f.write(f"Total Photos: {len(self.processed_images)}\n")
            f.write(f"Groups Found: {len(self.groups)}\n\n")
            
            for i, group in enumerate(self.groups, 1):
                f.write(f"GROUP {i} - Confidence: {group['confidence']:.1f}%\n")
                f.write("-" * 40 + "\n")
                f.write(f"Photo Count: {len(group['images'])}\n")
                features = group['common_features']
                f.write("Common Features:\n")
                f.write(f"  - Color: {features.get('dominant_color', 'Unknown')}\n")
                f.write(f"  - Size: {features.get('size_category', 'Unknown')}\n")
                f.write(f"  - Texture: {features.get('texture_type', 'Unknown')}\n")
                f.write("Photos:\n")
                for img_id in group['images']:
                    img_path = self.processed_images[img_id]['path']
                    img_name = os.path.basename(img_path)
                    f.write(f"  * {img_name}\n")
                f.write("\n")
        
        return json_file, summary_file

def main():
    """Main analysis function - DEPRECATED: Use run_analysis.py instead."""
    print("WARNING: Direct execution of luggage_analyzer.py is deprecated.")
    print("   Please use the main entry point instead:")
    print("   python run_analysis.py")
    print()
    print("   For help and options:")
    print("   python run_analysis.py --help")
    print()
    
    import sys
    try:
        # Try to import and run the main system
        from run_analysis import main as run_main
        print("Redirecting to run_analysis.py...")
        run_main()
    except ImportError:
        print("ERROR: Could not import run_analysis.py")
        print("   Please run: python run_analysis.py")
        sys.exit(1)

if __name__ == "__main__":
    main() 