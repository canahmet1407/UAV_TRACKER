# ✈️ UAV_TRACKER - Temel Hibrit Takip Sistemi (YOLOv8 + LK + CSRT)

## 📌 Projenin Amacı
Bu modül, insansız hava araçlarının gerçek zamanlı tespiti ve takibi için geliştirilmiş **ilk ve temel hibrit mimaridir**. Yalnızca YOLO algoritmalarının yetersiz kaldığı anlık hedef kayıplarını önlemek amacıyla **Lucas-Kanade (LK) Optik Akış** ve **CSRT** algoritmaları entegre edilmiştir.

## 🚀 Öne Çıkan Özellikler
* **YOLOv8:** Belirlenen aralıklarla (frame interval) ana nesne taramasını yapar.
* **Lucas-Kanade Optik Akış:** YOLO'nun tarama yapmadığı ara karelerde pikselleri takip eder.
* **CSRT Yedek Takipçi:** LK'nın zayıfladığı anlarda devreye girer.
* **Hayalet Takip Önleyici (Sabır Sınırı):** Hedef 10 kare boyunca YOLO tarafından doğrulanamazsa kilit otomatik olarak düşürülür ve yeni hedef aranır.

## 🚀 Çalıştırma ve Kurulum
1. Model ağırlık dosyanızı (`.pt`) ve test videonuzu (`1.mp4` - `6.mp4` arası)  VIDEO_VE_MODEL_KLASORU klasörünüze ekleyin.
2. Kod içerisindeki `VIDEO_VE_MODEL_KLASORU` yolunu kendi bilgisayarınıza göre güncelleyin.
3. Çalıştırmak için:
   ```bash
   python3 UAV_TRACKER.py
