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
    
    def __init__(self, similarity_threshold: float = 90.0, use_sam: bool = True):
        self.logger = setup_logging()
        self.comparator = LuggageComparator()
        self.processed_images = {}
        self.groups = []
        self.similarity_threshold = similarity_threshold
        self.use_sam = use_sam
        
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
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # SAM segmentation for luggage isolation (if enabled)
            if self.use_sam:
                try:
                    segmented_image = self.comparator.segment_luggage(image_rgb)
                    if segmented_image is not None:
                        # Use segmented image for feature extraction
                        features = self.comparator.extract_features(segmented_image)
                        embedding = self.comparator.extract_embedding(segmented_image)
                    else:
                        # Fallback to original image if segmentation fails
                        self.logger.warning(f"SAM segmentation failed for {os.path.basename(image_path)}, using original image")
                        features = self.comparator.extract_features(image_rgb)
                        embedding = self.comparator.extract_embedding(image_rgb)
                except Exception as e:
                    # Fallback to original image if SAM unavailable
                    self.logger.warning(f"SAM error for {os.path.basename(image_path)}: {e}, using original image")
                    features = self.comparator.extract_features(image_rgb)
                    embedding = self.comparator.extract_embedding(image_rgb)
            else:
                # Use original image directly
                features = self.comparator.extract_features(image_rgb)
                embedding = self.comparator.extract_embedding(image_rgb)
            
            # Store with unique ID
            image_id = f"img_{i:03d}_{os.path.basename(image_path)}"
            self.processed_images[image_id] = {
                'path': image_path,
                'features': features,
                'embedding': embedding
            }
    
    def group_by_visual_clustering(self):
        """Pure visual clustering using CLIP embeddings and K-means."""
        self.logger.info("PURE VISUAL CLUSTERING STARTING")
        
        image_ids = list(self.processed_images.keys())
        n_images = len(image_ids)
        
        if n_images < 2:
            self.logger.warning("Not enough images for clustering")
            return
        
        # Extract embeddings matrix
        embeddings = []
        for img_id in image_ids:
            embeddings.append(self.processed_images[img_id]['embedding'])
        
        embeddings_matrix = np.array(embeddings)
        
        # Determine optimal number of clusters using silhouette analysis
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        best_k = 2
        best_score = -1
        
        for k in range(2, min(n_images, 8)):  # Test 2-7 clusters
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_matrix)
            score = silhouette_score(embeddings_matrix, cluster_labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Final clustering with optimal k
        self.logger.info(f"Using {best_k} clusters (silhouette score: {best_score:.3f})")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_matrix)
        
        # Create groups
        self.groups = []
        for cluster_id in range(best_k):
            cluster_images = [image_ids[i] for i in range(n_images) if cluster_labels[i] == cluster_id]
            
            if cluster_images:
                # Calculate cluster embedding (centroid)
                cluster_embeddings = [self.processed_images[img_id]['embedding'] for img_id in cluster_images]
                cluster_embedding = np.mean(cluster_embeddings, axis=0)
                
                group = {
                    'cluster_id': cluster_id,
                    'images': cluster_images,
                    'size': len(cluster_images),
                    'method': 'visual_clustering',
                    'cluster_embedding': cluster_embedding
                }
                
                self.groups.append(group)
        
        self.logger.info(f"VISUAL CLUSTERING COMPLETED: {len(self.groups)} clusters created")
        for i, group in enumerate(self.groups):
            img_names = [os.path.basename(self.processed_images[img_id]['path']) for img_id in group['images']]
            self.logger.info(f"  Cluster {group['cluster_id']}: {group['size']} images - {img_names}")
    
    def search_and_match(self, search_files: List[str], threshold: float = 85.0) -> Dict[str, Any]:
        """Two-stage search: clustering + individual matching."""
        self.logger.info(f"TWO-STAGE SEARCH STARTING: {len(search_files)} search images")
        
        if not self.groups:
            self.logger.error("No existing groups found! Run grouping first.")
            return {}
        
        search_results = []
        
        for search_file in search_files:
            self.logger.info(f"Processing search image: {os.path.basename(search_file)}")
            
            # Process search image
            search_image = cv2.imread(search_file)
            if search_image is None:
                self.logger.error(f"Could not load search image: {search_file}")
                continue
                
            search_image_rgb = cv2.cvtColor(search_image, cv2.COLOR_BGR2RGB)
            
            # SAM segmentation for search image (if enabled)
            if self.use_sam:
                try:
                    segmented_search = self.comparator.segment_luggage(search_image_rgb)
                    if segmented_search is not None:
                        # Use segmented image for feature extraction
                        search_embedding = self.comparator.extract_embedding(segmented_search)
                        search_features = self.comparator.extract_features(segmented_search)
                    else:
                        # Fallback to original image
                        search_embedding = self.comparator.extract_embedding(search_image_rgb)
                        search_features = self.comparator.extract_features(search_image_rgb)
                except Exception as e:
                    # Fallback to original image if SAM fails
                    self.logger.warning(f"SAM error for search image: {e}, using original image")
                    search_embedding = self.comparator.extract_embedding(search_image_rgb)
                    search_features = self.comparator.extract_features(search_image_rgb)
            else:
                # Use original image directly
                search_embedding = self.comparator.extract_embedding(search_image_rgb)
                search_features = self.comparator.extract_features(search_image_rgb)
            
            # Stage 1: Find best cluster
            best_cluster = None
            best_cluster_similarity = 0
            
            for group in self.groups:
                cluster_embedding = group['cluster_embedding']
                cluster_sim = cosine_similarity([search_embedding], [cluster_embedding])[0][0]
                cluster_similarity = (cluster_sim + 1) / 2 * 100
                
                if cluster_similarity > best_cluster_similarity:
                    best_cluster_similarity = cluster_similarity
                    best_cluster = group
            
            # Stage 2: Detailed matching within best cluster
            if best_cluster:
                detailed_matches = self._detailed_cluster_matching(search_embedding, search_features, best_cluster)
                
                # Get best individual match
                best_match = detailed_matches[0] if detailed_matches else None
                
                # Determine if it's a match
                is_match = best_match and best_match['individual_score'] >= threshold
                
                result = {
                    'search_image': os.path.basename(search_file),
                    'search_path': search_file,
                    'is_match': is_match,
                    'confidence': best_match['individual_score'] if best_match else 0.0,
                    'best_cluster': {
                        'cluster_id': best_cluster['cluster_id'],
                        'cluster_similarity': best_cluster_similarity,
                        'cluster_size': best_cluster['size']
                    },
                    'best_match': best_match,
                    'detailed_matches': detailed_matches[:5],  # Top 5
                    'threshold_used': threshold
                }
                
                # Add potential match if below threshold
                if not is_match and best_match and best_match['individual_score'] >= (threshold * 0.7):
                    result['potential_match'] = {
                        'image_name': best_match['image_name'],
                        'individual_score': best_match['individual_score'],
                        'reason': f'Below threshold ({threshold}%) but significant match'
                    }
                
                search_results.append(result)
        
        return {
            'search_results': search_results,
            'total_searches': len(search_files),
            'matches_found': sum(1 for r in search_results if r['is_match']),
            'potential_matches': sum(1 for r in search_results if r.get('potential_match')),
            'threshold_used': threshold,
            'method': 'two_stage_clustering',
            'analysis_date': datetime.now().isoformat()
        }
    
    def _detailed_cluster_matching(self, search_embedding: np.ndarray, search_features: Dict[str, Any], cluster: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detailed matching within the best cluster - individual image comparisons."""
        detailed_matches = []
        
        for img_id in cluster['images']:
            img_data = self.processed_images[img_id]
            img_embedding = img_data['embedding']
            img_features = img_data['features']
            
            # Calculate similarities
            visual_sim = cosine_similarity([search_embedding], [img_embedding])[0][0]
            visual_similarity = (visual_sim + 1) / 2 * 100
            
            profile_match = self._calculate_individual_profile_match(search_features, img_features)
            embedding_similarity = visual_similarity  # Same as visual for now
            
            # Combined score (weighted)
            individual_score = (visual_similarity * 0.6) + (profile_match * 0.3) + (embedding_similarity * 0.1)
            
            detailed_matches.append({
                'image_id': img_id,
                'image_name': os.path.basename(img_data['path']),
                'image_path': img_data['path'],
                'visual_similarity': visual_similarity,
                'profile_match': profile_match,
                'embedding_similarity': embedding_similarity,
                'individual_score': individual_score
            })
        
        # Sort by individual score (highest first)
        detailed_matches.sort(key=lambda x: x['individual_score'], reverse=True)
        
        return detailed_matches
    
    def _calculate_individual_profile_match(self, search_features: Dict[str, Any], img_features: Dict[str, Any]) -> float:
        """Calculate profile match between search image and individual image."""
        match_scores = []
        
        # Color matching (using new simple format)
        if 'color' in search_features and 'color' in img_features:
            search_color = search_features['color']
            img_color = img_features['color']
            # Simple color name matching
            if search_color == img_color:
                color_similarity = 90.0
            elif search_color in ['black', 'gray', 'white'] and img_color in ['black', 'gray', 'white']:
                color_similarity = 70.0  # Neutral colors similar
            else:
                color_similarity = 40.0  # Different colors
            match_scores.append(('color', color_similarity, 0.4))
        
        # Brightness matching
        if 'brightness' in search_features and 'brightness' in img_features:
            search_brightness = search_features['brightness']
            img_brightness = img_features['brightness']
            brightness_diff = abs(search_brightness - img_brightness)
            brightness_similarity = max(0, 100 - (brightness_diff / 255 * 100))
            match_scores.append(('brightness', brightness_similarity, 0.3))
        
        # Texture matching (using texture complexity)
        if 'texture_complexity' in search_features and 'texture_complexity' in img_features:
            search_texture = search_features['texture_complexity']
            img_texture = img_features['texture_complexity']
            texture_diff = abs(search_texture - img_texture)
            # Normalize by typical texture range (0-100)
            texture_similarity = max(0, 100 - (texture_diff / 50 * 100))
            match_scores.append(('texture', texture_similarity, 0.3))
        
        # Calculate weighted average
        if match_scores:
            total_weighted_score = sum(score * weight for _, score, weight in match_scores)
            total_weight = sum(weight for _, _, weight in match_scores)
            final_score = total_weighted_score / total_weight if total_weight > 0 else 50.0
        else:
            final_score = 50.0
        
        return final_score
    
    def direct_search_and_match(self, search_files: List[str], threshold: float = 85.0) -> Dict[str, Any]:
        """Direct search without clustering - compare search image against all input images."""
        self.logger.info(f"DIRECT SEARCH STARTING: {len(search_files)} search images vs {len(self.processed_images)} input images")
        
        search_results = []
        
        for search_file in search_files:
            self.logger.info(f"Processing search image: {os.path.basename(search_file)}")
            
            # Process search image
            search_image = cv2.imread(search_file)
            if search_image is None:
                self.logger.error(f"Could not load search image: {search_file}")
                continue
                
            search_image_rgb = cv2.cvtColor(search_image, cv2.COLOR_BGR2RGB)
            
            # SAM segmentation for search image (if enabled)
            if self.use_sam:
                try:
                    segmented_search = self.comparator.segment_luggage(search_image_rgb)
                    if segmented_search is not None:
                        # Use segmented image for feature extraction
                        search_embedding = self.comparator.extract_embedding(segmented_search)
                        search_features = self.comparator.extract_features(segmented_search)
                    else:
                        # Fallback to original image
                        search_embedding = self.comparator.extract_embedding(search_image_rgb)
                        search_features = self.comparator.extract_features(search_image_rgb)
                except Exception as e:
                    # Fallback to original image if SAM fails
                    self.logger.warning(f"SAM error for search image: {e}, using original image")
                    search_embedding = self.comparator.extract_embedding(search_image_rgb)
                    search_features = self.comparator.extract_features(search_image_rgb)
            else:
                # Use original image directly
                search_embedding = self.comparator.extract_embedding(search_image_rgb)
                search_features = self.comparator.extract_features(search_image_rgb)
            
            # Compare against all input images
            matches = []
            
            for img_id, img_data in self.processed_images.items():
                # Calculate visual similarity
                visual_sim = cosine_similarity([search_embedding], [img_data['embedding']])[0][0]
                visual_similarity = (visual_sim + 1) / 2 * 100
                
                # Calculate profile match
                profile_match = self._calculate_individual_profile_match(search_features, img_data['features'])
                
                # Calculate embedding similarity (different from visual similarity)
                embedding_similarity = visual_similarity  # For now, same as visual
                
                # Combined score
                individual_score = (visual_similarity * 0.6) + (profile_match * 0.3) + (embedding_similarity * 0.1)
                
                matches.append({
                    'image_id': img_id,
                    'image_name': os.path.basename(img_data['path']),
                    'image_path': img_data['path'],
                    'visual_similarity': visual_similarity,
                    'profile_match': profile_match,
                    'embedding_similarity': embedding_similarity,
                    'individual_score': individual_score
                })
            
            # Sort by individual score (highest first)
            matches.sort(key=lambda x: x['individual_score'], reverse=True)
            
            # Get best match
            best_match = matches[0] if matches else None
            
            # Determine if it's a match based on threshold
            is_match = best_match and best_match['individual_score'] >= threshold
            
            result = {
                'search_image': os.path.basename(search_file),
                'search_path': search_file,
                'is_match': is_match,
                'confidence': best_match['individual_score'] if best_match else 0.0,
                'best_match': {
                    'image_name': best_match['image_name'] if best_match else None,
                    'image_path': best_match['image_path'] if best_match else None,
                    'individual_score': best_match['individual_score'] if best_match else 0.0,
                    'visual_similarity': best_match['visual_similarity'] if best_match else 0.0,
                    'profile_match': best_match['profile_match'] if best_match else 0.0,
                    'embedding_similarity': best_match['embedding_similarity'] if best_match else 0.0
                },
                'all_matches': matches[:10],  # Top 10 matches
                'threshold_used': threshold
            }
            
            # Add potential match if below threshold but still significant
            if not is_match and best_match and best_match['individual_score'] >= (threshold * 0.7):
                result['potential_match'] = {
                    'image_name': best_match['image_name'],
                    'individual_score': best_match['individual_score'],
                    'reason': f'Below threshold ({threshold}%) but significant match'
                }
            
            search_results.append(result)
            
            # Log result
            if is_match:
                self.logger.info(f"✅ MATCH: {os.path.basename(search_file)} → {best_match['image_name']} ({best_match['individual_score']:.1f}%)")
            else:
                self.logger.info(f"❌ NO MATCH: {os.path.basename(search_file)} → Best: {best_match['image_name'] if best_match else 'None'} ({best_match['individual_score']:.1f}% < {threshold}%)" if best_match else "❌ NO MATCH: No candidates found")
        
        return {
            'search_results': search_results,
            'total_searches': len(search_files),
            'matches_found': sum(1 for r in search_results if r['is_match']),
            'potential_matches': sum(1 for r in search_results if r.get('potential_match')),
            'threshold_used': threshold,
            'method': 'direct_comparison',
            'analysis_date': datetime.now().isoformat()
        }

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