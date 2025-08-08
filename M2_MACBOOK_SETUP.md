# M2 MacBook Kurulum ve Çalıştırma Talimatları

## 🚀 Hızlı Başlangıç

### 1. Gerekli Paketleri Yükleyin
```bash
pip install -r requirements.txt
```

### 2. M2 MacBook Optimizasyonları
Proje otomatik olarak M2 MacBook'unuzu algılayacak ve optimize edecek.

### 3. Çalıştırma

**Basit çalıştırma:**
```bash
python run_m2_macbook.py
```

**Normal çalıştırma:**
```bash
python analyze_luggage.py --folder input
```

## 📊 Sistem Gereksinimleri

- **RAM:** 8GB (minimum 6GB boş)
- **Storage:** 5GB boş alan
- **macOS:** 12.0 veya üzeri
- **Python:** 3.8+

## 🔧 Optimizasyonlar

### Memory Optimizasyonları:
- ✅ Batch size: 1 (tek resim işleme)
- ✅ Max image size: 1024px
- ✅ Luggage threshold: 0.3 (düşük)
- ✅ Cache size: 2GB
- ✅ MPS device kullanımı

### Performance Optimizasyonları:
- ✅ Model caching
- ✅ Memory cleanup
- ✅ Garbage collection
- ✅ Resource monitoring

## 📁 Dosya Yapısı

```
lbs68/
├── input/                 # Resimlerinizi buraya koyun
├── output/               # Sonuçlar buraya kaydedilir
├── model_cache/          # Model cache'leri
├── config_m2_macbook.yaml # M2 optimizasyonları
├── run_m2_macbook.py     # Optimize edilmiş script
└── analyze_luggage.py    # Normal script
```

## ⚠️ Önemli Notlar

1. **İlk çalıştırma:** Model indirme 5-10 dakika sürebilir
2. **Memory kullanımı:** ~4-6GB RAM kullanır
3. **Batch işleme:** Aynı anda çok resim işlemeyin
4. **Sıcaklık:** M2 chip ısınabilir, fan hızı artabilir

## 🛠️ Sorun Giderme

### Memory Hatası:
```bash
# Memory cleanup
python -c "import gc; gc.collect()"
```

### Model Yükleme Hatası:
```bash
# Cache'i temizle
rm -rf model_cache/*
```

### MPS Hatası:
```bash
# CPU'ya geç
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 📈 Performans Beklentileri

- **10 resim:** ~2-3 dakika
- **20 resim:** ~4-6 dakika  
- **50 resim:** ~10-15 dakika

## 🎯 Öneriler

1. **Küçük batch'lerle başlayın** (5-10 resim)
2. **Diğer uygulamaları kapatın** (Chrome, Xcode vb.)
3. **Fan sesini normal karşılayın** (M2 ısınır)
4. **İlk çalıştırmada sabırlı olun** (model indirme)

## 📞 Destek

Sorun yaşarsanız:
1. `run_m2_macbook.py` scriptini kullanın
2. Memory kullanımını kontrol edin
3. Config dosyasını kontrol edin 