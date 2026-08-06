# ✈️ UAV_V2 - CLAHE Işık Korumalı Hibrit Takip

## 📌 Projenin Amacı
Gökyüzündeki zorlu ışık koşulları (ters ışık, güneş patlamaları, bulut gölgeleri) nedeniyle kararan veya parlayan nesnelerin kaçırılmasını engellemek için geliştirilmiş versiyondur.

## 🚀 Öne Çıkan Özellikler
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Görüntü LAB renk uzayına çevrilir ve sadece parlaklık (`L`) kanalına adaptif histogram eşitleme uygulanır. Renkler bozulmadan karanlık İHA'lar aydınlatılır ve güneş patlamaları dengelenir.
* **Hibrit Takip & Kalman:** İyileştirilmiş görüntü üzerinde çalışan LK ve Kalman kombinasyonu sayesinde takip kararlılığı zirveye çıkarılır.

## 🚀 Çalıştırma ve Kurulum
1. Gerekli kütüphanelerin (`opencv-contrib-python`, `ultralytics`, `torch`) kurulu olduğundan emin olun.
2. Çalıştırmak için:
   ```bash
   python UAV_TRACKER_V2.py  
