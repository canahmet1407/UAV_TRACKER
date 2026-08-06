# ✈️ UAV_V5 - Ultimate Takip (DeepSORT/BoT-SORT + ReID + CLAHE + Kalman)

## 📌 Projenin Amacı
Projenin en gelişmiş, jüriye sunulabilecek en kararlı ve profesyonel versiyonudur. Sistem sadece konumu değil, İHA'nın görsel özelliklerini (dokusunu ve rengini) de ezberleyerek kalabalık senaryolarda doğru hedefi takip eder.

## 🚀 Öne Çıkan Özellikler
* **ReID (Re-Identification) Desteği:** `with_reid: True` ve sınıflandırma modeli entegrasyonu ile yanından başka bir İHA geçse bile sistem kendi hedefinin ID'sini kaybetmez.
* **Otomatik ReID Model İndirme:** Kodun başında tanımlanan BoT-SORT / DeepSORT ayarları (`osnet_x0_25_msmt17.pt` veya `yolov8n-cls.pt`), İHA'nın görsel özelliklerini (görünüm matrisini) çıkarabilmek için Ultralytics kütüphanesi tarafından ilk çalıştırılmada **otomatik olarak indirilir**. Bu, sistemin çalışması için zorunludur ve normal bir durumdur.
* **Sıkılaştırılmış Boşluk Koruması:** Boşluğa veya buluta kilitlenmeyi engellemek için ilk kilitlenme anında minimum **%65 Güven Skoru** şartı aranır.
* **CLAHE & GMC:** Işık optimizasyonu ve küresel hareket telafisi (Global Motion Compensation) bir arada çalışır.

## 🚀 Çalıştırma ve Kurulum
1. Gerekli konfigürasyon dosyalarının (`.yaml`) kod tarafından otomatik oluşturulmasına izin verin.
2. Çalıştırmak için:
   ```bash
   python UAV_TRACKER_V5.py  
