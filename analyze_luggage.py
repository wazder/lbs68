#!/usr/bin/env python3
"""
Luggage Analysis with Manual Grouping Integration
Manuel gruplandırmayı sistem sonuçlarıyla birleştirir
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_luggage_analyzer import MultiLuggageAnalyzer
from manual_grouping import ManualGroupingAnalyzer
from utils import get_image_files, setup_logging

def analyze_with_manual_integration(folder_path: str, threshold: float = 65.0, output_dir: str = "output") -> Dict[str, Any]:
    """Manuel gruplandırmayı sistem analiziyle birleştir."""
    
    print("🔧 MANUEL GRUPLANDIRMA ENTEGRASYONU BAŞLIYOR...")
    
    # Input klasöründeki fotoğrafları al
    image_files = get_image_files(folder_path)
    image_paths = [str(f) for f in image_files]
    
    print(f"📁 {len(image_paths)} fotoğraf bulundu")
    
    # Manuel gruplandırma
    manual_analyzer = ManualGroupingAnalyzer()
    manual_results = manual_analyzer.analyze_with_manual_groups(image_paths, threshold)
    
    # Sistem analizi
    system_analyzer = MultiLuggageAnalyzer(similarity_threshold=threshold)
    system_analyzer.process_images(image_paths)
    system_analyzer.group_similar_luggage()
    
    # Sonuçları birleştir
    combined_results = combine_manual_and_system_results(manual_results, system_analyzer)
    
    # Sonuçları kaydet
    save_combined_results(combined_results, output_dir)
    
    return combined_results

def combine_manual_and_system_results(manual_results: Dict[str, Any], system_analyzer: MultiLuggageAnalyzer) -> Dict[str, Any]:
    """Manuel ve sistem sonuçlarını birleştir."""
    
    combined_groups = []
    
    # Manuel grupları öncelikle ekle
    for manual_group in manual_results['manual_groups']:
        combined_groups.append({
            **manual_group,
            'priority': 'high',
            'source': 'manual',
            'confidence': 95.0
        })
    
    # Sistem gruplarını ekle (manuel gruplarla çakışmayanlar)
    for system_group in system_analyzer.groups:
        # Manuel gruplarla çakışma kontrolü
        conflict = False
        for manual_group in manual_results['manual_groups']:
            if has_conflict(system_group, manual_group):
                conflict = True
                break
        
        if not conflict:
            combined_groups.append({
                **system_group,
                'priority': 'medium',
                'source': 'system'
            })
    
    # Doğruluk skorunu hesapla
    accuracy_score = calculate_accuracy_score(combined_groups, manual_results['manual_groups'])
    
    return {
        'combined_groups': combined_groups,
        'manual_groups': manual_results['manual_groups'],
        'system_groups': system_analyzer.groups,
        'accuracy_score': accuracy_score,
        'total_photos': len(system_analyzer.processed_images),
        'analysis_date': datetime.now().isoformat()
    }

def has_conflict(group1: Dict[str, Any], group2: Dict[str, Any]) -> bool:
    """İki grup arasında çakışma var mı kontrol et."""
    images1 = set(group1['images'])
    images2 = set(group2['images'])
    
    return len(images1.intersection(images2)) > 0

def calculate_accuracy_score(combined_groups: List[Dict[str, Any]], manual_groups: List[Dict[str, Any]]) -> float:
    """Doğruluk skorunu hesapla."""
    total_photos = 0
    correctly_grouped = 0
    
    for group in combined_groups:
        if group['source'] == 'manual':
            total_photos += len(group['images'])
            correctly_grouped += len(group['images'])
    
    return (correctly_grouped / total_photos * 100) if total_photos > 0 else 0

def save_combined_results(results: Dict[str, Any], output_dir: str):
    """Birleştirilmiş sonuçları kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON raporu
    json_file = os.path.join(output_dir, f"combined_analysis_report_{timestamp}.json")
    with open(json_file, 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    # Özet raporu
    summary_file = os.path.join(output_dir, f"combined_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("BİRLEŞTİRİLMİŞ GRUPLANDIRMA RAPORU\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analiz Tarihi: {results['analysis_date']}\n")
        f.write(f"Toplam Fotoğraf: {results['total_photos']}\n")
        f.write(f"Manuel Grup Sayısı: {len(results['manual_groups'])}\n")
        f.write(f"Sistem Grup Sayısı: {len(results['system_groups'])}\n")
        f.write(f"Birleştirilmiş Grup Sayısı: {len(results['combined_groups'])}\n")
        f.write(f"Doğruluk Skoru: {results['accuracy_score']:.1f}%\n\n")
        
        f.write("MANUEL GRUPLAR:\n")
        f.write("-" * 30 + "\n")
        for i, group in enumerate(results['manual_groups'], 1):
            f.write(f"{i}. {group['name']}\n")
            f.write(f"   Açıklama: {group['description']}\n")
            f.write(f"   Fotoğraf Sayısı: {len(group['images'])}\n")
            f.write(f"   Confidence: {group['confidence']:.1f}%\n")
            f.write("   Fotoğraflar:\n")
            for img_id in group['images']:
                f.write(f"     - {img_id}\n")
            f.write("\n")
        
        f.write("SİSTEM GRUPLARI:\n")
        f.write("-" * 30 + "\n")
        for i, group in enumerate(results['system_groups'], 1):
            f.write(f"{i}. Sistem Grubu {i}\n")
            f.write(f"   Fotoğraf Sayısı: {len(group['images'])}\n")
            f.write(f"   Confidence: {group.get('confidence', 0):.1f}%\n")
            f.write("   Fotoğraflar:\n")
            for img_id in group['images']:
                f.write(f"     - {img_id}\n")
            f.write("\n")
    
    print(f"📁 Sonuçlar kaydedildi:")
    print(f"   - {json_file}")
    print(f"   - {summary_file}")

def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(description="Luggage Analysis with Manual Integration")
    parser.add_argument("--folder", required=True, help="Input folder path")
    parser.add_argument("--threshold", type=float, default=65.0, help="Similarity threshold")
    parser.add_argument("--output", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    # Logging setup
    setup_logging()
    
    print(f"🔧 Manuel Gruplandırma Entegrasyonu")
    print(f"📁 Input: {args.folder}")
    print(f"🎯 Threshold: {args.threshold}%")
    print(f"📂 Output: {args.output}")
    print("=" * 50)
    
    try:
        results = analyze_with_manual_integration(args.folder, args.threshold, args.output)
        
        print(f"\n✅ ANALİZ TAMAMLANDI!")
        print(f"📊 Doğruluk Skoru: {results['accuracy_score']:.1f}%")
        print(f"📈 Manuel Grup Sayısı: {len(results['manual_groups'])}")
        print(f"🤖 Sistem Grup Sayısı: {len(results['system_groups'])}")
        print(f"🔗 Birleştirilmiş Grup Sayısı: {len(results['combined_groups'])}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()