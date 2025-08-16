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
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from luggage_comparator import LuggageComparator
from utils import get_image_files, setup_logging

class LuggageAnalyzer:
    """
    Main Luggage Analyzer with Advanced Precision
    """
    
    def __init__(self, similarity_threshold: float = 90.0):
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
        
        # Use visual clustering
        self.group_by_visual_clustering()
        
        # Return results
        return {
            'groups': self.groups,
            'total_photos': len(self.processed_images),
            'processed_images': {k: {'path': v['path']} for k, v in self.processed_images.items()},
            'analysis_date': datetime.now().isoformat()
        }
        
        
    def process_images(self, image_paths: List[str]):
        """Advanced precision image processing."""
# Remove verbose logging
        
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
        
        # 2. Perceptual Hash (for duplicate detection)
        features['perceptual_hash'] = self._calculate_perceptual_hash(image)
        
        # 3. SAM Segmentation
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
    
    def _calculate_perceptual_hash(self, image: np.ndarray) -> int:
        """Calculate perceptual hash for image similarity."""
        # Resize to 8x8 for simplicity
        resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Calculate average pixel value
        avg = np.mean(gray)
        
        # Create binary hash based on whether each pixel is above average
        hash_bits = []
        for row in gray:
            for pixel in row:
                hash_bits.append(1 if pixel > avg else 0)
        
        # Convert binary list to integer
        hash_value = 0
        for bit in hash_bits:
            hash_value = (hash_value << 1) | bit
            
        return hash_value
    
    def calculate_ultra_similarity(self, img1_id: str, img2_id: str) -> float:
        """Ultra-precision similarity calculation with weighted features."""
        img1_data = self.processed_images[img1_id]
        img2_data = self.processed_images[img2_id]
        
        similarities = {}
        
        # 1. CLIP Embedding Similarity (primary - 85% weight)
        clip_sim = cosine_similarity([img1_data['embedding']], [img2_data['embedding']])[0][0]
        similarities['clip'] = (clip_sim + 1) / 2 * 100
        
        # 2. Color Histogram Similarity (secondary - 10% weight)
        if 'color' in img1_data['features'] and 'color' in img2_data['features']:
            color_sim = self._calculate_color_similarity(img1_data['features']['color'], img2_data['features']['color'])
            similarities['color'] = color_sim
        else:
            similarities['color'] = similarities['clip']  # Fallback to CLIP
        
        # 3. Perceptual Hash Similarity (tertiary - 5% weight)
        if 'perceptual_hash' in img1_data['features'] and 'perceptual_hash' in img2_data['features']:
            hash1 = img1_data['features']['perceptual_hash']
            hash2 = img2_data['features']['perceptual_hash']
            # Calculate Hamming distance and convert to similarity
            hamming_dist = bin(hash1 ^ hash2).count('1')
            hash_similarity = max(0, (64 - hamming_dist) / 64 * 100)  # 64-bit hash
            similarities['hash'] = hash_similarity
        else:
            similarities['hash'] = similarities['clip']  # Fallback to CLIP
        
        # ENHANCED weighted combination with material/shape emphasis
        # Higher weight on color and hash for better same-luggage detection
        weights = {'clip': 0.70, 'color': 0.20, 'hash': 0.10}
        final_similarity = sum(similarities[key] * weights[key] for key in weights.keys())
        
        # Bonus for very high individual similarities (same material/shape)
        if similarities['color'] > 95.0:  # Very similar color
            final_similarity = min(100.0, final_similarity * 1.05)
        if similarities['hash'] > 90.0:  # Very similar structure
            final_similarity = min(100.0, final_similarity * 1.03)
        
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
    
    def group_by_visual_clustering(self):
        """Pure visual clustering using CLIP embeddings and K-means/DBSCAN."""
        self.logger.info("PURE VISUAL CLUSTERING STARTING")
        
        image_ids = list(self.processed_images.keys())
        n_images = len(image_ids)
        
        if n_images < 2:
            self.logger.warning("Not enough images for clustering")
            return
        
        # Extract all CLIP embeddings
        embeddings = []
        for img_id in image_ids:
            embedding = self.processed_images[img_id]['embedding']
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings)
        self.logger.info(f"Extracted {len(embeddings)} CLIP embeddings, shape: {embeddings_array.shape}")
        
        # Determine optimal number of clusters
        optimal_k = self._find_optimal_clusters(embeddings_array, max_k=min(20, n_images//2))
        self.logger.info(f"Optimal number of clusters: {optimal_k}")
        
        # Perform K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # Create groups from clusters
        self.groups = []
        for cluster_id in range(optimal_k):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_img_ids = [image_ids[i] for i in cluster_indices]
            
            if len(cluster_img_ids) == 0:
                continue
                
            # Calculate intra-cluster similarities
            group_similarities = []
            if len(cluster_img_ids) > 1:
                for i in range(len(cluster_img_ids)):
                    for j in range(i+1, len(cluster_img_ids)):
                        sim = self.calculate_ultra_similarity(cluster_img_ids[i], cluster_img_ids[j])
                        group_similarities.append(sim)
            
            # Create detailed group profile
            group_profile = self._create_detailed_group_profile(cluster_img_ids)
            
            # Calculate cluster centroid distance (compactness)
            cluster_embeddings = embeddings_array[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
            compactness = 100 - (np.mean(distances) * 100)  # Higher = more compact
            
            group = {
                'images': cluster_img_ids,
                'cluster_id': cluster_id,
                'confidence': np.mean(group_similarities) if group_similarities else compactness,
                'compactness': compactness,
                'similarities': {},
                'common_features': self._analyze_group_features(cluster_img_ids),
                'detailed_profile': group_profile,
                'cluster_center': centroid.tolist()
            }
            
            self.groups.append(group)
            self.logger.info(f"Cluster {cluster_id}: {len(cluster_img_ids)} images, compactness: {compactness:.1f}%")
            self.logger.info(f"  Profile: {group_profile.get('summary', 'N/A')}")
        
        self.logger.info(f"PURE VISUAL CLUSTERING COMPLETED: {optimal_k} clusters found")
    
    def _find_optimal_clusters(self, embeddings: np.ndarray, max_k: int = 20) -> int:
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        n_samples = len(embeddings)
        if n_samples < 4:
            return min(2, n_samples)
        
        # Test different k values
        k_range = range(2, min(max_k + 1, n_samples))
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            if k > n_samples:
                break
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score (skip if too few samples)
            if len(set(cluster_labels)) > 1:
                sil_score = silhouette_score(embeddings, cluster_labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # Find elbow point (biggest drop in inertia)
        if len(inertias) >= 3:
            # Calculate second derivative to find elbow
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            elbow_idx = np.argmax(second_diffs) + 2  # +2 because of double diff
            elbow_k = list(k_range)[elbow_idx] if elbow_idx < len(k_range) else list(k_range)[-1]
        else:
            elbow_k = list(k_range)[0]
        
        # Find best silhouette score
        if silhouette_scores:
            best_sil_idx = np.argmax(silhouette_scores)
            best_sil_k = list(k_range)[best_sil_idx]
        else:
            best_sil_k = elbow_k
        
        # Choose between elbow and silhouette (prefer silhouette if score > 0.3)
        if silhouette_scores and max(silhouette_scores) > 0.3:
            optimal_k = best_sil_k
            self.logger.info(f"Using silhouette-based k={optimal_k} (score: {max(silhouette_scores):.3f})")
        else:
            optimal_k = elbow_k
            self.logger.info(f"Using elbow-based k={optimal_k}")
        
        # Safety bounds: reasonable number of clusters for luggage
        optimal_k = max(2, min(optimal_k, max(10, n_samples // 4)))
        
        return optimal_k
    
    def _create_detailed_group_profile(self, image_ids: List[str]) -> Dict[str, Any]:
        """Create detailed min/max profile for a group."""
        if not image_ids:
            return {}
        
        # Collect all features from group images
        all_features = [self.processed_images[img_id]['features'] for img_id in image_ids]
        
        # Color analysis (RGB values)
        colors = [f['color']['dominant_color'] for f in all_features]
        color_array = np.array(colors)
        
        # Texture analysis (edge statistics) 
        edge_means = [f['texture']['edge_mean'] for f in all_features]
        edge_stds = [f['texture']['edge_std'] for f in all_features]
        texture_energies = [f['texture']['texture_energy'] for f in all_features]
        
        # Shape analysis
        areas = [f['shape']['area'] for f in all_features]
        aspect_ratios = [f['shape']['aspect_ratio'] for f in all_features if f['shape']['aspect_ratio'] > 0]
        circularities = [f['shape']['circularity'] for f in all_features]
        
        # Edge analysis
        edge_densities = [f['edges']['edge_density'] for f in all_features]
        
        # Create detailed profile with min/max ranges
        profile = {
            'color_profile': {
                'red_range': [float(np.min(color_array[:, 0])), float(np.max(color_array[:, 0]))],
                'green_range': [float(np.min(color_array[:, 1])), float(np.max(color_array[:, 1]))],
                'blue_range': [float(np.min(color_array[:, 2])), float(np.max(color_array[:, 2]))],
                'dominant_color_avg': [float(np.mean(color_array[:, 0])), 
                                     float(np.mean(color_array[:, 1])), 
                                     float(np.mean(color_array[:, 2]))],
                'color_variance': float(np.var(color_array))
            },
            'texture_profile': {
                'edge_mean_range': [float(np.min(edge_means)), float(np.max(edge_means))],
                'edge_std_range': [float(np.min(edge_stds)), float(np.max(edge_stds))],
                'texture_energy_range': [float(np.min(texture_energies)), float(np.max(texture_energies))],
                'edge_density_range': [float(np.min(edge_densities)), float(np.max(edge_densities))],
                'texture_consistency': float(1.0 - np.std(edge_means) / (np.mean(edge_means) + 1e-8))
            },
            'shape_profile': {
                'area_range': [float(np.min(areas)), float(np.max(areas))],
                'aspect_ratio_range': [float(np.min(aspect_ratios)) if aspect_ratios else 0.0, 
                                     float(np.max(aspect_ratios)) if aspect_ratios else 0.0],
                'circularity_range': [float(np.min(circularities)), float(np.max(circularities))],
                'avg_area': float(np.mean(areas)),
                'shape_consistency': float(1.0 - np.std(areas) / (np.mean(areas) + 1e-8))
            },
            'size_profile': {
                'size_category': self._categorize_size(np.mean(areas)),
                'size_variance': float(np.var(areas)),
                'size_range': [float(np.min(areas)), float(np.max(areas))]
            }
        }
        
        # Create human-readable summary
        avg_color = profile['color_profile']['dominant_color_avg']
        color_name = self._get_color_name(avg_color)
        avg_area = profile['shape_profile']['avg_area']
        size_cat = profile['size_profile']['size_category']
        texture_consistency = profile['texture_profile']['texture_consistency']
        
        profile['summary'] = f"{color_name} color, {size_cat} size, {texture_consistency:.0%} texture consistency"
        
        return profile
    
    def _categorize_size(self, area: float) -> str:
        """Categorize luggage size based on area."""
        if area < 5000:
            return "extra_small"
        elif area < 20000:
            return "small" 
        elif area < 50000:
            return "medium"
        elif area < 100000:
            return "large"
        else:
            return "extra_large"
    
    def _get_color_name(self, rgb_values: List[float]) -> str:
        """Get color name from RGB values."""
        r, g, b = rgb_values
        
        # Convert to HSV for better color classification
        rgb_normalized = np.array([r, g, b]) / 255.0
        rgb_uint8 = np.uint8([[rgb_normalized * 255]])
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv
        
        if v < 50:  # Very dark
            return "black"
        elif v > 200 and s < 50:  # Very light
            return "white"
        elif s < 50:  # Low saturation
            return "gray"
        elif h < 15 or h > 165:  # Red range
            return "red"
        elif 15 <= h < 45:  # Orange/Yellow
            return "orange/yellow"
        elif 45 <= h < 75:  # Green
            return "green"
        elif 75 <= h < 105:  # Cyan
            return "cyan"
        elif 105 <= h < 135:  # Blue
            return "blue"
        elif 135 <= h <= 165:  # Purple/Magenta
            return "purple"
        else:
            return "mixed"
    
    def _calculate_profile_match(self, search_features: Dict[str, Any], group_profile: Dict[str, Any]) -> float:
        """Calculate how well search image matches group profile (0-100%)."""
        if not group_profile:
            return 50.0  # Neutral score if no profile
        
        match_scores = []
        
        # Color matching
        if 'color_profile' in group_profile and 'color' in search_features:
            search_color = search_features['color']['dominant_color']
            color_profile = group_profile['color_profile']
            
            # Check if search color falls within group's color ranges
            r_match = color_profile['red_range'][0] <= search_color[0] <= color_profile['red_range'][1]
            g_match = color_profile['green_range'][0] <= search_color[1] <= color_profile['green_range'][1]
            b_match = color_profile['blue_range'][0] <= search_color[2] <= color_profile['blue_range'][1]
            
            # Calculate color distance to group average
            group_avg_color = color_profile['dominant_color_avg']
            color_distance = np.linalg.norm(np.array(search_color) - np.array(group_avg_color))
            color_similarity = max(0, 100 - (color_distance / 255 * 100))
            
            # Combine range matching and distance
            range_bonus = 20 if (r_match and g_match and b_match) else 0
            color_score = min(100, color_similarity + range_bonus)
            match_scores.append(('color', color_score, 0.4))  # 40% weight
        
        # Texture matching
        if 'texture_profile' in group_profile and 'texture' in search_features:
            search_texture = search_features['texture']
            texture_profile = group_profile['texture_profile']
            
            # Check texture ranges
            edge_mean_match = texture_profile['edge_mean_range'][0] <= search_texture['edge_mean'] <= texture_profile['edge_mean_range'][1]
            edge_std_match = texture_profile['edge_std_range'][0] <= search_texture['edge_std'] <= texture_profile['edge_std_range'][1]
            energy_match = texture_profile['texture_energy_range'][0] <= search_texture['texture_energy'] <= texture_profile['texture_energy_range'][1]
            
            # Calculate texture score
            range_matches = sum([edge_mean_match, edge_std_match, energy_match])
            texture_score = (range_matches / 3) * 100
            match_scores.append(('texture', texture_score, 0.25))  # 25% weight
        
        # Shape matching
        if 'shape_profile' in group_profile and 'shape' in search_features:
            search_shape = search_features['shape']
            shape_profile = group_profile['shape_profile']
            
            # Check shape ranges
            area_match = shape_profile['area_range'][0] <= search_shape['area'] <= shape_profile['area_range'][1]
            aspect_match = shape_profile['aspect_ratio_range'][0] <= search_shape['aspect_ratio'] <= shape_profile['aspect_ratio_range'][1]
            circ_match = shape_profile['circularity_range'][0] <= search_shape['circularity'] <= shape_profile['circularity_range'][1]
            
            # Calculate shape score
            range_matches = sum([area_match, aspect_match, circ_match])
            shape_score = (range_matches / 3) * 100
            match_scores.append(('shape', shape_score, 0.25))  # 25% weight
        
        # Size matching  
        if 'size_profile' in group_profile and 'shape' in search_features:
            search_area = search_features['shape']['area']
            size_profile = group_profile['size_profile']
            
            # Check size range
            size_match = size_profile['size_range'][0] <= search_area <= size_profile['size_range'][1]
            
            # Size category consistency
            search_size_cat = self._categorize_size(search_area)
            group_size_cat = size_profile['size_category']
            category_match = search_size_cat == group_size_cat
            
            size_score = 100 if (size_match and category_match) else (50 if size_match or category_match else 0)
            match_scores.append(('size', size_score, 0.1))  # 10% weight
        
        # Calculate weighted average
        if match_scores:
            total_weighted_score = sum(score * weight for _, score, weight in match_scores)
            total_weight = sum(weight for _, _, weight in match_scores)
            final_score = total_weighted_score / total_weight if total_weight > 0 else 50.0
        else:
            final_score = 50.0  # Neutral if no matching possible
        
        return final_score
    
    def _detailed_cluster_matching(self, search_embedding: np.ndarray, search_features: Dict[str, Any], cluster: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detailed matching within the best cluster - individual image comparisons."""
        detailed_matches = []
        
        for img_id in cluster['images']:
            # Get image data
            img_data = self.processed_images[img_id]
            img_embedding = img_data['embedding']
            img_features = img_data['features']
            img_path = img_data['path']
            
            # Calculate detailed similarity scores
            visual_sim = cosine_similarity([search_embedding], [img_embedding])[0][0]
            visual_similarity = (visual_sim + 1) / 2 * 100
            
            # Profile matching for this specific image
            profile_match = self._calculate_individual_profile_match(search_features, img_features)
            
            # Embedding distance
            embedding_distance = np.linalg.norm(search_embedding - img_embedding)
            embedding_similarity = max(0, 100 - (embedding_distance * 100))
            
            # Combined individual score
            individual_score = (visual_similarity * 0.6) + (profile_match * 0.3) + (embedding_similarity * 0.1)
            
            match_info = {
                'image_id': img_id,
                'image_path': img_path,
                'image_name': os.path.basename(img_path),
                'visual_similarity': visual_similarity,
                'profile_match': profile_match,
                'embedding_similarity': embedding_similarity,
                'individual_score': individual_score,
                'features_summary': self._get_image_summary(img_features)
            }
            
            detailed_matches.append(match_info)
        
        # Sort by individual score (highest first)
        detailed_matches.sort(key=lambda x: x['individual_score'], reverse=True)
        
        return detailed_matches
    
    def _calculate_individual_profile_match(self, search_features: Dict[str, Any], img_features: Dict[str, Any]) -> float:
        """Calculate profile match between search image and individual image."""
        match_scores = []
        
        # Color matching
        if 'color' in search_features and 'color' in img_features:
            search_color = search_features['color']['dominant_color']
            img_color = img_features['color']['dominant_color']
            color_distance = np.linalg.norm(np.array(search_color) - np.array(img_color))
            color_similarity = max(0, 100 - (color_distance / 255 * 100))
            match_scores.append(('color', color_similarity, 0.4))
        
        # Texture matching
        if 'texture' in search_features and 'texture' in img_features:
            search_edge = search_features['texture']['edge_mean']
            img_edge = img_features['texture']['edge_mean']
            edge_similarity = max(0, 100 - abs(search_edge - img_edge))
            match_scores.append(('texture', edge_similarity, 0.3))
        
        # Shape matching
        if 'shape' in search_features and 'shape' in img_features:
            search_area = search_features['shape']['area']
            img_area = img_features['shape']['area']
            area_ratio = min(search_area, img_area) / max(search_area, img_area)
            area_similarity = area_ratio * 100
            match_scores.append(('shape', area_similarity, 0.3))
        
        # Calculate weighted average
        if match_scores:
            total_weighted_score = sum(score * weight for _, score, weight in match_scores)
            total_weight = sum(weight for _, _, weight in match_scores)
            final_score = total_weighted_score / total_weight if total_weight > 0 else 50.0
        else:
            final_score = 50.0
        
        return final_score
    
    def _get_image_summary(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of image features for display."""
        summary = {}
        
        if 'color' in features:
            color = features['color']['dominant_color']
            color_name = self._get_color_name(color)
            summary['color'] = f"{color_name} {color}"
        
        if 'shape' in features:
            area = features['shape']['area']
            size_cat = self._categorize_size(area)
            summary['size'] = f"{size_cat} ({area:.0f}px²)"
        
        if 'texture' in features:
            edge_mean = features['texture']['edge_mean']
            summary['texture'] = f"Edge: {edge_mean:.1f}"
        
        return summary
    
    def search_and_match(self, search_image_paths: List[str], threshold: float = 85.0) -> Dict[str, Any]:
        """Search and match new images to existing groups using visual similarity."""
        self.logger.info(f"VISUAL SEARCH STARTING: {len(search_image_paths)} search images")
        
        if not self.groups:
            self.logger.error("No existing groups found! Run grouping first.")
            return {}
        
        search_results = []
        
        for search_path in search_image_paths:
            self.logger.info(f"Searching for matches: {os.path.basename(search_path)}")
            
            # Process search image
            search_image = self.comparator.load_image(search_path)
            search_features = self._extract_ultra_features(search_image, search_path)
            search_embedding = search_features['clip_embedding']
            
            # Calculate similarity to each group using detailed profiles
            group_similarities = []
            
            for i, group in enumerate(self.groups):
                # Visual similarity (CLIP embeddings)
                group_sims = []
                for img_id in group['images']:
                    existing_embedding = self.processed_images[img_id]['embedding']
                    similarity = cosine_similarity([search_embedding], [existing_embedding])[0][0]
                    similarity_percent = (similarity + 1) / 2 * 100
                    group_sims.append(similarity_percent)
                
                avg_visual_similarity = np.mean(group_sims)
                max_visual_similarity = np.max(group_sims)
                
                # Profile-based matching
                profile_match_score = self._calculate_profile_match(search_features, group.get('detailed_profile', {}))
                
                # For visual clustering: also use cluster center distance
                cluster_center_similarity = 0
                if 'cluster_center' in group:
                    center_distance = np.linalg.norm(search_embedding - np.array(group['cluster_center']))
                    cluster_center_similarity = max(0, 100 - (center_distance * 100))
                
                # Combined score: 50% visual + 30% profile + 20% cluster center
                combined_score = (avg_visual_similarity * 0.5) + (profile_match_score * 0.3) + (cluster_center_similarity * 0.2)
                
                group_similarities.append({
                    'group_index': i,
                    'cluster_id': group.get('cluster_id', i),
                    'group_name': f"Cluster_{group.get('cluster_id', i)}",
                    'avg_similarity': avg_visual_similarity,
                    'max_similarity': max_visual_similarity,
                    'profile_match': profile_match_score,
                    'cluster_center_similarity': cluster_center_similarity,
                    'combined_score': combined_score,
                    'image_count': len(group['images'])
                })
            
            # Sort by combined score (visual + profile)
            group_similarities.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Find best match
            best_match = group_similarities[0]
            
            # Stage 2: Detailed matching within best cluster
            best_cluster = self.groups[best_match['group_index']]
            detailed_matches = self._detailed_cluster_matching(search_embedding, search_features, best_cluster)
            
            search_result = {
                'search_image': os.path.basename(search_path),
                'search_path': search_path,
                'best_match': best_match,
                'all_similarities': group_similarities,
                'detailed_matches': detailed_matches,
                'potential_match': detailed_matches[0] if detailed_matches else None,
                'is_match': best_match['combined_score'] >= threshold,
                'confidence': best_match['combined_score'],
                'visual_confidence': best_match['avg_similarity'],
                'profile_confidence': best_match['profile_match']
            }
            
            # Log result with detailed matches
            if search_result['is_match']:
                self.logger.info(f"MATCH FOUND: {search_result['search_image']} → {best_match['group_name']} (Combined: {best_match['combined_score']:.1f}%, Visual: {best_match['avg_similarity']:.1f}%, Profile: {best_match['profile_match']:.1f}%)")
            else:
                self.logger.info(f"NO MATCH: {search_result['search_image']} → Best: {best_match['group_name']} (Combined: {best_match['combined_score']:.1f}% < {threshold}%)")
            
            # Log potential match (top individual match)
            if search_result['potential_match']:
                potential = search_result['potential_match']
                self.logger.info(f"POTENTIAL MATCH: {potential['image_name']} (Individual Score: {potential['individual_score']:.1f}%, Visual: {potential['visual_similarity']:.1f}%, Profile: {potential['profile_match']:.1f}%)")
                
                # Log top 3 detailed matches
                self.logger.info(f"TOP MATCHES in {best_match['group_name']}:")
                for i, match in enumerate(detailed_matches[:3], 1):
                    self.logger.info(f"  {i}. {match['image_name']}: {match['individual_score']:.1f}% (Visual: {match['visual_similarity']:.1f}%, Profile: {match['profile_match']:.1f}%)")
            else:
                self.logger.info("No potential matches found in cluster")
            
            search_results.append(search_result)
        
        self.logger.info(f"VISUAL SEARCH COMPLETED: {len(search_results)} images processed")
        
        return {
            'search_results': search_results,
            'existing_groups': self.groups,
            'search_threshold': threshold
        }
    
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