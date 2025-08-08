#!/usr/bin/env python3
"""
Ensemble Analyzer
Çoklu model ve algoritma kullanarak doğruluğu artırır
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from multi_luggage_analyzer import MultiLuggageAnalyzer
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

class EnsembleAnalyzer:
    """Çoklu model ensemble ile doğruluğu artıran analyzer."""
    
    def __init__(self):
        self.models = {
            'dbscan': DBSCAN,
            'kmeans': KMeans,
            'hierarchical': AgglomerativeClustering
        }
        
    def analyze_with_ensemble(self, image_paths: List[str], threshold: float = 65.0) -> Dict[str, Any]:
        """Ensemble analiz yap."""
        print("🎯 ENSEMBLE ANALİZ BAŞLIYOR...")
        
        # Ana analyzer'ı çalıştır
        main_analyzer = MultiLuggageAnalyzer(similarity_threshold=threshold)
        main_analyzer.process_images(image_paths)
        main_analyzer.calculate_similarity_matrix()
        
        # Farklı clustering algoritmaları dene
        ensemble_results = {}
        
        # 1. DBSCAN ile farklı eps değerleri
        for eps in [0.2, 0.3, 0.4, 0.5]:
            groups = self._cluster_with_dbscan(main_analyzer, eps)
            ensemble_results[f'dbscan_eps_{eps}'] = groups
        
        # 2. K-Means ile farklı k değerleri
        for k in [3, 4, 5, 6]:
            groups = self._cluster_with_kmeans(main_analyzer, k)
            ensemble_results[f'kmeans_k_{k}'] = groups
        
        # 3. Hierarchical clustering
        groups = self._cluster_with_hierarchical(main_analyzer)
        ensemble_results['hierarchical'] = groups
        
        # 4. Consensus voting
        consensus_groups = self._consensus_voting(ensemble_results, main_analyzer)
        
        return {
            'ensemble_results': ensemble_results,
            'consensus_groups': consensus_groups,
            'main_analyzer': main_analyzer
        }
    
    def _cluster_with_dbscan(self, analyzer: MultiLuggageAnalyzer, eps: float) -> List[Dict[str, Any]]:
        """DBSCAN ile clustering."""
        if analyzer.similarity_matrix is None:
            return []
        
        # Similarity matrix'i distance matrix'e çevir
        distance_matrix = 1 - analyzer.similarity_matrix / 100
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)
        
        return self._create_groups_from_labels(analyzer, labels)
    
    def _cluster_with_kmeans(self, analyzer: MultiLuggageAnalyzer, k: int) -> List[Dict[str, Any]]:
        """K-Means ile clustering."""
        if analyzer.similarity_matrix is None:
            return []
        
        # Similarity matrix'i feature matrix'e çevir
        feature_matrix = analyzer.similarity_matrix
        
        # K-Means clustering
        clustering = KMeans(n_clusters=k, random_state=42)
        labels = clustering.fit_predict(feature_matrix)
        
        return self._create_groups_from_labels(analyzer, labels)
    
    def _cluster_with_hierarchical(self, analyzer: MultiLuggageAnalyzer) -> List[Dict[str, Any]]:
        """Hierarchical clustering."""
        if analyzer.similarity_matrix is None:
            return []
        
        # Similarity matrix'i distance matrix'e çevir
        distance_matrix = 1 - analyzer.similarity_matrix / 100
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=0.5,
            linkage='ward',
            metric='precomputed'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        return self._create_groups_from_labels(analyzer, labels)
    
    def _create_groups_from_labels(self, analyzer: MultiLuggageAnalyzer, labels: np.ndarray) -> List[Dict[str, Any]]:
        """Labels'dan gruplar oluştur."""
        groups = []
        image_ids = list(analyzer.processed_images.keys())
        
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            group_indices = np.where(labels == label)[0]
            group_images = [image_ids[i] for i in group_indices]
            
            if len(group_images) >= 2:
                # Grup özelliklerini analiz et
                common_features = analyzer._analyze_group_features(group_images)
                confidence = analyzer._calculate_group_confidence(group_images)
                
                group = {
                    'images': group_images,
                    'confidence': confidence,
                    'common_features': common_features,
                    'matching_type': 'ensemble'
                }
                
                groups.append(group)
        
        return groups
    
    def _consensus_voting(self, ensemble_results: Dict[str, List[Dict[str, Any]]], analyzer: MultiLuggageAnalyzer) -> List[Dict[str, Any]]:
        """Ensemble sonuçlarından consensus oluştur."""
        print("🗳️ CONSENSUS VOTING BAŞLIYOR...")
        
        # Her fotoğraf için hangi gruplarda olduğunu say
        image_votes = {}
        image_ids = list(analyzer.processed_images.keys())
        
        for img_id in image_ids:
            image_votes[img_id] = {}
        
        # Her model sonucunu say
        for model_name, groups in ensemble_results.items():
            for group in groups:
                for img_id in group['images']:
                    if img_id not in image_votes:
                        image_votes[img_id] = {}
                    
                    # Bu fotoğrafın hangi gruplarda olduğunu kaydet
                    group_key = f"{model_name}_{len(group['images'])}"
                    if group_key not in image_votes[img_id]:
                        image_votes[img_id][group_key] = 0
                    image_votes[img_id][group_key] += 1
        
        # Consensus grupları oluştur
        consensus_groups = []
        used_images = set()
        
        # En yüksek oy alan grupları seç
        for img_id in image_ids:
            if img_id in used_images:
                continue
            
            # Bu fotoğrafın en çok hangi gruplarda olduğunu bul
            if img_id in image_votes:
                votes = image_votes[img_id]
                if votes:
                    # En yüksek oy alan grup tipini bul
                    best_group_type = max(votes.items(), key=lambda x: x[1])[0]
                    
                    # Aynı grup tipindeki diğer fotoğrafları bul
                    consensus_group = [img_id]
                    used_images.add(img_id)
                    
                    for other_img_id in image_ids:
                        if other_img_id not in used_images and other_img_id in image_votes:
                            other_votes = image_votes[other_img_id]
                            if best_group_type in other_votes and other_votes[best_group_type] > 0:
                                consensus_group.append(other_img_id)
                                used_images.add(other_img_id)
                    
                    if len(consensus_group) >= 2:
                        # Grup özelliklerini analiz et
                        common_features = analyzer._analyze_group_features(consensus_group)
                        confidence = analyzer._calculate_group_confidence(consensus_group)
                        
                        consensus_groups.append({
                            'images': consensus_group,
                            'confidence': confidence,
                            'common_features': common_features,
                            'matching_type': 'consensus',
                            'vote_count': len([g for g in ensemble_results.values() if any(img_id in group['images'] for group in g)])
                        })
        
        return consensus_groups
    
    def save_ensemble_results(self, results: Dict[str, Any], output_dir: str = "ensemble_results"):
        """Ensemble sonuçlarını kaydet."""
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON raporu
        json_file = os.path.join(output_dir, "ensemble_analysis_report.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Özet raporu
        summary_file = os.path.join(output_dir, "ensemble_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("ENSEMBLE ANALİZ RAPORU\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("MODEL SONUÇLARI:\n")
            for model_name, groups in results['ensemble_results'].items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Grup sayısı: {len(groups)}\n")
                for i, group in enumerate(groups, 1):
                    f.write(f"  Grup {i}: {len(group['images'])} fotoğraf (Confidence: {group['confidence']:.1f}%)\n")
            
            f.write(f"\nCONSENSUS GRUPLARI:\n")
            f.write(f"  Toplam grup: {len(results['consensus_groups'])}\n")
            for i, group in enumerate(results['consensus_groups'], 1):
                f.write(f"  Grup {i}: {len(group['images'])} fotoğraf (Confidence: {group['confidence']:.1f}%, Votes: {group['vote_count']})\n")
        
        return json_file, summary_file


def main():
    """Ensemble analiz testi."""
    print("🎯 ENSEMBLE ANALİZ TESTİ")
    print("=" * 40)
    
    # Input klasöründeki fotoğrafları al
    input_dir = "input"
    image_paths = []
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(input_dir, filename))
    
    # Ensemble analyzer'ı çalıştır
    ensemble_analyzer = EnsembleAnalyzer()
    results = ensemble_analyzer.analyze_with_ensemble(image_paths)
    
    # Sonuçları kaydet
    json_file, summary_file = ensemble_analyzer.save_ensemble_results(results)
    
    print(f"✅ Ensemble analiz tamamlandı!")
    print(f"📊 Consensus grup sayısı: {len(results['consensus_groups'])}")
    print(f"📁 Sonuçlar kaydedildi:")
    print(f"   - {json_file}")
    print(f"   - {summary_file}")


if __name__ == "__main__":
    main() 