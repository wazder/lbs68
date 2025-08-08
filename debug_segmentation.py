#!/usr/bin/env python3
"""
SAM segmentation debug scripti
"""

import os
import sys
import numpy as np
from pathlib import Path
from luggage_comparator import LuggageComparator
from utils import get_image_files

def debug_segmentation():
    """SAM segmentation'ı test et."""
    print("🔍 SAM Segmentation Debug")
    
    # LuggageComparator'ı başlat
    print("1. LuggageComparator başlatılıyor...")
    comparator = LuggageComparator()
    
    # İlk resmi al
    image_files = get_image_files('input')
    if not image_files:
        print("❌ Resim bulunamadı!")
        return
    
    first_image = str(image_files[0])
    print(f"2. Test resmi: {first_image}")
    
    try:
        # Resmi yükle
        print("3. Resim yükleniyor...")
        image = comparator.load_image(first_image)
        print(f"   ✅ Resim yüklendi: {image.shape}")
        
        # SAM segmentation test et
        print("4. SAM segmentation test ediliyor...")
        if comparator.sam_predictor is not None:
            print("   ✅ SAM predictor mevcut")
            
            # Segmentation yap
            mask = comparator.segment_luggage(image)
            print(f"   ✅ Mask oluşturuldu: {mask.shape}")
            print(f"   ✅ Mask değerleri: min={mask.min()}, max={mask.max()}")
            print(f"   ✅ Mask türü: {mask.dtype}")
            
            # Mask'ın boş olup olmadığını kontrol et
            if np.sum(mask > 0) == 0:
                print("   ⚠️  Mask boş (hiç pixel yok)")
            else:
                print(f"   ✅ Mask'ta {np.sum(mask > 0)} pixel var")
                
        else:
            print("   ❌ SAM predictor yok!")
            
    except Exception as e:
        print(f"   ❌ Hata: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
    
    print("\n🔍 SAM Debug tamamlandı!")

if __name__ == "__main__":
    debug_segmentation() 