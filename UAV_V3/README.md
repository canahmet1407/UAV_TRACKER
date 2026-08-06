# ✈️ UAV_V3 - ByteTrack & Kalman Takip Sistemi

## 📌 Projenin Amacı
Bu sürümde CSRT tracker tamamen devreden çıkarılarak endüstri standardı olan **ByteTrack** çoklu nesne takip motoruna geçilmiştir. İnce profilli İHA'ları bozmadan mikro piksel gürültülerini eleyen bir yapı kurulmuştur.

## 🚀 Öne Çıkan Özellikler
* **ByteTrack Algoritması:** Düşük ve yüksek güven skoruna sahip kutuları akıllıca eşleştirerek ID değişimlerini önler.
* **Mikro Gürültü Filtresi:** Kuş veya leke gibi aşırı küçük nesnelerin (15x10 piksel altı) işlenmesini engelleyerek CPU/GPU yükünü azaltır.
* **LK Yedekleme:** ByteTrack'in kısa süreli aksamalarında optik akış devreye girerek hedefi korur.

## 🚀 Çalıştırma ve Kurulum
1. Klasör içerisindeki kod dosyasını çalıştırın:
   ```bash
   python3 UAV_TRACKER_V3.py
