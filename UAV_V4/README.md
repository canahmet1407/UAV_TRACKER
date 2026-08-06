✈️ UAV_V4 - ByteTrack & Kalman Takip Sistemi

## 📌 Projenin Amacı
Bu sürümde klasik CSRT tracker tamamen devreden çıkarılarak, modern nesne takibi sağlayan **ByteTrack** motoruna geçilmiştir. İnce profilli İHA'ları bozmadan mikro piksel gürültülerini eleyen ve uzaktaki hedefleri yüksek başarıyla kilitleyen bir yapıdır.

## 🚀 Öne Çıkan Özellikler
* **ByteTrack Algoritması:** Yüksek ve düşük güven skoruna sahip kutuları akıllıca eşleştirerek hedefin ara sıra kaybolmalarında (occlusion) kimlik (ID) kopmalarını önler.
* **Mikro Gürültü Filtresi:** Kuş veya nokta leke gibi aşırı küçük nesnelerin (15x10 piksel altı) işlenmesini engelleyerek sistemi gereksiz yükten kurtarır.
* **Lucas-Kanade (LK) Yedekleme:** ByteTrack'in geçici olarak ID kaybettiği durumlarda optik akış devreye girerek hedefi takip etmeye devam eder.
* **2D Kalman Filtresi:** Hedefin yörüngesini tahmin ederek kutuyu yumuşatır ve sarsıntıları tolerans eder.

## 🚀 Çalıştırma ve Kurulum
1. Model ağırlık dosyanızı (`KENDİ_MODELİN.pt`) ve test videolarınızı (`1.mp4` - `6.mp4` arası) uygun dizine ekleyin.
2. Kod içerisindeki `MODEL_PATH` ve video yollarını kendi bilgisayarınıza göre kontrol edin.
3. Çalıştırmak için terminalden şu komutu girin:
   ```bash
   python3 UAV_TRACKER_V4.py
