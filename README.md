# ✈️ Teknofest 2026 - Savaşan İHA Görüntü İşleme & Hedef Takip Sistemi

Bu depo, **Teknofest 2026 Savaşan İHA** yarışması kapsamında geliştirilen gerçek zamanlı nesne tespiti, hibrit takip (Object Tracking), yörünge tahmini (Kalman) ve görsel tanıma (ReID) algoritmalarını barındırır.

---

## 📂 Kod Versiyonları ve Gelişim Evrimi

| Dosya Adı                       | Kullanılan Teknolojiler                            | Öne Çıkan Özellik                                                     | 
| :---                            | :---                                               | :---                                                                  |
| **`ucak_base.py`**              | YOLOv8 + Lucas-Kanade + CSRT                       | Temel hibrit yapı ve "Hayalet Takip" koruması.                        | 
| **`ucak_v1.py`**                | YOLOv8 + LK + Kalman Filtresi                      | 2D fizik tabanlı yörünge tahmini ve Teknofest kilitlenme timer'ı.     |
| **`ucak_v2_clahe.py`**          | YOLOv8 + LK + Kalman + CLAHE                       | Ters ışık ve güneş patlamalarına karşı dinamik aydınlatma filtresi.   |
| **`ucak_v3_bytetrack.py`**      | YOLOv8 (ByteTrack) + LK + Kalman                   | CSRT yerine modern ByteTrack entegrasyonu, mikro gürültü filtresi.    |
| **`ucak_v5_ultimate.py`**       | YOLOv8 (DeepSORT/BoT-SORT) + ReID + Kalman + CLAHE | Görsel doku tanıma (ReID) ile ID koruması ve en kararlı hedef kilidi. |

---

## ⚠️ Model ve Video Dosyaları Hakkında

Büyük boyutlu model ağırlıkları (`.pt`) ve test videoları (`.mp4`) GitHub depolama sınırları ve optimizasyon nedeniyle bu repoda **bulunmamaktadır**. 

### 📌 Dosya Konumlandırma ve İsimlendirme Kuralları
Sistemi kendi bilgisayarınızda çalıştırmak için dosyaları dilediğiniz bir klasöre koyabilirsiniz, ancak kodların düzgün çalışabilmesi için şu kurallara dikkat etmeniz gerekir:

1. **Video İsimlendirmesi:** Kodlar çalıştırıldığında sizden `1` ile `6` arasında bir kaynak seçimi isteyecektir. Bu yüzden test videolarınızın adı kesinlikle **`1.mp4`, `2.mp4`, `3.mp4`, `4.mp4`, `5.mp4` ve `6.mp4`** olmalıdır.
2. **Yol (Path) Ayarı:** Kodu kendi bilgisayarınıza indikten sonra, scriptlerin en üst kısmındaki `MODEL_PATH` ve video dosya yollarını **kendi bilgisayarınızdaki klasör konumuna göre** güncellemeyi unutmayın.

---

## ⚙️ Gereksinimler

```bash
pip install ultralytics opencv-contrib-python torch numpy
