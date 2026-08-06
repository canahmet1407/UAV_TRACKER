# ✈️ UAV_V1 - Kalman Filtreli ve Teknofest Kurallı Hibrit Takip

## 📌 Projenin Amacı
Teknofest Savaşan İHA yarışma kurallarına (hedefin merkez vuruş alanında tutulması, minimum %6 boyut şartı ve 4 saniye kilitlenme süresi) tam uyumlu hale getirilmiş versiyondur. Temel sisteme **Kalman Filtresi** eklenerek yörünge tahmini yapılması sağlanmıştır.

## 🚀 Öne Çıkan Özellikler
* **2D Kalman Filtresi:** İHA'nın hız ve ivmesini hesaplayarak kamera sarsıntılarında veya ani manevralarda konum tahmini yapar.
* **Kilitlenme Timer'ı:** Hedef sarı vuruş alanı içerisine girdiğinde 4 saniyelik geri sayım başlar ve başarılı kilitlenme raporlanır.
* **Gelişmiş Skorlama:** Merkez uzaklığı, güven skoru (confidence) ve alan büyüklüğü ağırlıklandırılarak en doğru aday seçilir.

## 🚀 Çalıştırma ve Kurulum
1. Model ve video dosyalarınızı ilgili dizine yerleştirin.
2. Kod içindeki yol değişkenlerini kontrol edin.
3. Çalıştırmak için:
   ```bash
   python3 UAV_TRACKER_V1.py  
