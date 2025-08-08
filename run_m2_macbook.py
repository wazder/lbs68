#!/usr/bin/env python3
"""
M2 MacBook için optimize edilmiş luggage analysis scripti
8GB RAM için memory-efficient ayarlar
"""

import os
import sys
import gc
import psutil
from pathlib import Path
from multi_luggage_analyzer import MultiLuggageAnalyzer
from utils import get_image_files, setup_logging
from config import get_config_manager

def check_system_resources():
    """Sistem kaynaklarını kontrol et."""
    memory = psutil.virtual_memory()
    print(f"📊 Sistem Durumu:")
    print(f"   RAM: {memory.total / (1024**3):.1f}GB toplam")
    print(f"   Kullanım: {memory.percent}%")
    print(f"   Boş: {memory.available / (1024**3):.1f}GB")
    
    if memory.percent > 80:
        print("⚠️  RAM kullanımı yüksek! Dikkatli olun.")
        return False
    return True

def optimize_for_m2():
    """M2 MacBook için optimizasyonlar."""
    print("🔧 M2 MacBook optimizasyonları uygulanıyor...")
    
    # Memory cleanup
    gc.collect()
    
    # Config'i yükle
    config_manager = get_config_manager("config_m2_macbook.yaml")
    config = config_manager.get_config()
    
    print(f"   Device: {config.model.device}")
    print(f"   Batch size: {config.processing.batch_size}")
    print(f"   Max image size: {config.processing.max_image_size}")
    print(f"   Luggage threshold: {config.processing.luggage_detection_threshold}")
    
    return config

def process_images_safely(image_files, config):
    """Güvenli resim işleme."""
    print(f"\n🖼️  {len(image_files)} resim işleniyor...")
    
    # Memory-efficient analyzer
    analyzer = MultiLuggageAnalyzer(
        similarity_threshold=config.processing.similarity_threshold
    )
    
    try:
        # Resimleri string'e çevir
        image_paths = [str(f) for f in image_files]
        
        # Process images
        print(f"🔍 {len(image_paths)} resim işleniyor...")
        for i, img_path in enumerate(image_paths):
            print(f"   {i+1}. {os.path.basename(img_path)}")
        
        analyzer.process_images(image_paths)
        print("✅ Resimler işlendi")
        
        # Group luggage
        analyzer.group_similar_luggage()
        print("✅ Gruplandırma tamamlandı")
        
        # Show results
        print(f"\n📊 SONUÇLAR:")
        print(f"   Toplam resim: {len(analyzer.processed_images)}")
        print(f"   Grup sayısı: {len(analyzer.groups)}")
        
        for i, group in enumerate(analyzer.groups, 1):
            print(f"   Grup {i}: {len(group['images'])} resim")
        
        # Save results
        results = analyzer.save_results("output")
        print(f"✅ Sonuçlar 'output' klasörüne kaydedildi")
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False
    
    finally:
        # Cleanup
        analyzer.cleanup()
        gc.collect()

def main():
    """Ana fonksiyon."""
    print("🚀 M2 MacBook Luggage Analysis")
    print("=" * 40)
    
    # Sistem kontrolü
    if not check_system_resources():
        print("❌ Yetersiz sistem kaynağı!")
        return
    
    # Optimizasyonlar
    config = optimize_for_m2()
    
    # Input klasörünü kontrol et
    input_folder = "input"
    if not os.path.exists(input_folder):
        print(f"❌ '{input_folder}' klasörü bulunamadı!")
        return
    
    # Resim dosyalarını al
    image_files = get_image_files(input_folder)
    if not image_files:
        print(f"❌ '{input_folder}' klasöründe resim bulunamadı!")
        return
    
    print(f"📁 {len(image_files)} resim bulundu")
    
    # Güvenli işleme
    success = process_images_safely(image_files, config)
    
    if success:
        print("\n🎉 Analiz başarıyla tamamlandı!")
    else:
        print("\n❌ Analiz başarısız!")

if __name__ == "__main__":
    main() 