#!/usr/bin/env python3
"""
Basit debug scripti - adım adım test
"""

import os
import sys
from pathlib import Path
from luggage_comparator import LuggageComparator
from utils import get_image_files

def debug_step_by_step():
    """Adım adım debug."""
    print("🔍 Debug başlıyor...")
    
    # 1. Resim dosyalarını kontrol et
    print("\n1. Resim dosyalarını kontrol et:")
    image_files = get_image_files('input')
    print(f"   Bulunan resimler: {len(image_files)}")
    for i, img in enumerate(image_files):
        print(f"   {i+1}. {img}")
    
    # 2. LuggageComparator'ı başlat
    print("\n2. LuggageComparator başlatılıyor...")
    try:
        comparator = LuggageComparator()
        print("   ✅ LuggageComparator başarıyla başlatıldı")
    except Exception as e:
        print(f"   ❌ LuggageComparator hatası: {e}")
        return
    
    # 3. İlk resmi test et
    if image_files:
        first_image = str(image_files[0])
        print(f"\n3. İlk resim test ediliyor: {first_image}")
        
        try:
            # Resmi yükle
            print("   - Resim yükleniyor...")
            image = comparator.load_image(first_image)
            print(f"   ✅ Resim yüklendi: {image.shape}")
            
            # Embedding çıkar
            print("   - Embedding çıkarılıyor...")
            embedding = comparator.process_image(first_image)
            print(f"   ✅ Embedding çıkarıldı: {embedding.shape}")
            
            print("   ✅ İlk resim başarıyla işlendi!")
            
        except Exception as e:
            print(f"   ❌ İlk resim hatası: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
    
    print("\n🔍 Debug tamamlandı!")

if __name__ == "__main__":
    debug_step_by_step() 