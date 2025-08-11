#!/usr/bin/env python3
"""
ULTRA-PRECISION LUGGAGE ANALYZER
%100 Doğruluk için gelişmiş algoritma
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

class UltraPrecisionAnalyzer:
    """
    %100 Doğruluk için Ultra-Precision Luggage Analyzer
    """
    
    def __init__(self):
        self.logger = setup_logging()
        self.comparator = LuggageComparator()
        self.processed_images = {}
        self.groups = []
        
        # Ultra-precision settings
        self.multi_level_similarity = True
        self.geometric_verification = True
        self.color_analysis = True
        self.shape_analysis = True
        self.texture_analysis = True
        self.ensemble_voting = True
        
    def process_images(self, image_paths: List[str]):
        """Ultra-precision image processing."""
        self.logger.info("🚀 ULTRA-PRECISION PROCESSING BAŞLIYOR!")
        
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
        """Ultra-precision feature extraction."""
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
        
        # Balanced weighting for better luggage separation
        weights = {'clip': 0.35, 'color': 0.30, 'shape': 0.25, 'texture': 0.08, 'edges': 0.02}
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
    
    def group_with_ultra_precision(self, threshold: float = 75.0):
        """Ultra-precision grouping."""
        self.logger.info("🎯 ULTRA-PRECISION GROUPING BAŞLIYOR!")
        
        image_ids = list(self.processed_images.keys())
        n_images = len(image_ids)
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((n_images, n_images))
        
        for i in range(n_images):
            for j in range(i+1, n_images):
                similarity = self.calculate_ultra_similarity(image_ids[i], image_ids[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Create distance matrix (1 - similarity/100)
        distance_matrix = 1 - similarity_matrix / 100
        
        # DBSCAN with strict parameters for high precision
        clustering = DBSCAN(
            eps=1-threshold/100,  # Much stricter: only group if similarity > threshold
            min_samples=2,  # Allow smaller groups to form more precise clusters
            metric='precomputed'
        )
        
        # Perform clustering
        labels = clustering.fit_predict(distance_matrix)
        
        # Create groups
        self.groups = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points (individual photos)
                continue
            
            group_indices = np.where(labels == label)[0]
            group_images = [image_ids[i] for i in group_indices]
            
            if len(group_images) >= 2:
                group = {
                    'images': group_images,
                    'confidence': np.mean([similarity_matrix[i, j] for i in group_indices for j in group_indices if i != j]),
                    'similarities': {},
                    'common_features': self._analyze_group_features(group_images)
                }
                self.groups.append(group)
        
        self.logger.info(f"✅ ULTRA-PRECISION GROUPING TAMAMLANDI: {len(self.groups)} grup bulundu!")
    
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
    """Ultra-precision analysis main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Precision Luggage Analysis")
    parser.add_argument("--folder", default="input", help="Input folder path")
    parser.add_argument("--threshold", type=float, default=75.0, help="Similarity threshold (60-95)")
    parser.add_argument("--output", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    print("🔥 ULTRA-PRECISION LUGGAGE ANALYSIS 🔥")
    print("🎯 HEDEF: %100 DOĞRULUK!")
    print(f"📁 Input: {args.folder}")
    print(f"🎯 Threshold: {args.threshold}%")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = UltraPrecisionAnalyzer()
    
    # Get images
    image_files = [str(f) for f in get_image_files(args.folder)]
    print(f"📁 {len(image_files)} resim işleniyor...")
    
    # Process images
    analyzer.process_images(image_files)
    print("✅ Resimler işlendi!")
    
    # Ultra-precision grouping
    analyzer.group_with_ultra_precision(threshold=args.threshold)
    
    # Save results
    json_file, summary_file = analyzer.save_ultra_results(args.output)
    
    print(f"\n📊 SONUÇLAR:")
    print(f"   Toplam resim: {len(analyzer.processed_images)}")
    print(f"   Grup sayısı: {len(analyzer.groups)}")
    
    for i, group in enumerate(analyzer.groups, 1):
        print(f"   Grup {i}: {len(group['images'])} resim (Confidence: {group['confidence']:.1f}%)")
    
    print(f"\n📁 Kaydedilen dosyalar:")
    print(f"   - {json_file}")
    print(f"   - {summary_file}")
    
    print("\n🎉 ULTRA-PRECISION ANALİZ TAMAMLANDI!")

if __name__ == "__main__":
    main() 