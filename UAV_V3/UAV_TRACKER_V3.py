import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time

print("GPU (Ekran Karti) Aktif mi?:", torch.cuda.is_available())

print("\n" + "═"*50)
print(" ✈️  TEKNOFEST SAVAŞAN İHA - BYTE+LK+KALMAN SİSTEMİ ")
print("═"*50)
video_secim = input(" Lutfen islenecek kaynagi secin (Kamera icin 0, Video icin 1-6): ")

if not video_secim.isdigit() or not (0 <= int(video_secim) <= 6):
    print(" [!] Gecersiz secim! Varsayilan video (5.mp4) yukleniyor...\n")
    SECILEN_KAYNAK = '/home/ahmet/Desktop/UAV_TRACKER/MODELS_and_VİDEO/5.mp4'
elif video_secim == "0":
    print(" [+] Canli Kamera secildi. Sistem baslatiliyor...\n")
    SECILEN_KAYNAK = 0  
else:
    print(f" [+] {video_secim}.mp4 secildi. Sistem baslatiliyor...\n")
    SECILEN_KAYNAK = f'/home/ahmet/Desktop/UAV_TRACKER/MODELS_and_VİDEO/{video_secim}.mp4'

# ══════════════════════════════════════════════════════════════════════════════
#  AYARLAR
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH         = '/home/ahmet/Desktop/UAV_TRACKER/MODELS_and_VİDEO/KENDİ_MODELİN.pt'
VIDEO_PATH         = SECILEN_KAYNAK
FRAME_W, FRAME_H   = 640, 480    

# ÇÖZÜM 1: Eşiği tekrar %45'e çektik ki uzaktaki uçağı görebilsin
CONF_THRESH        = 0.45    
HEDEF_MIN_YUZDE    = 0.06    
LK_MAX_KAYIP       = 15      

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
FEATURE_PARAMS = dict(maxCorners=300, qualityLevel=0.25, minDistance=5, blockSize=7)
BBOX_MIN_BOYUT = 10    

# ══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════
def bbox_sinirla(bbox, w=FRAME_W, h=FRAME_H):
    x, y, bw, bh = bbox
    x  = max(0, min(x, w - 1))
    y  = max(0, min(y, h - 1))
    bw = max(BBOX_MIN_BOYUT, min(bw, w - x))
    bh = max(BBOX_MIN_BOYUT, min(bh, h - y))
    return (int(x), int(y), int(bw), int(bh))

def lk_noktalari_al(gray, bbox):
    x, y, w, h = bbox
    margin_x = int(w * 0.10)
    margin_y = int(h * 0.10)
    
    x1 = max(0, x + margin_x)
    y1 = max(0, y + margin_y)
    x2 = min(FRAME_W, x + w - margin_x)
    y2 = min(FRAME_H, y + h - margin_y)
    
    if x2 - x1 < BBOX_MIN_BOYUT or y2 - y1 < BBOX_MIN_BOYUT:
        return None
        
    roi = gray[y1:y2, x1:x2]
    pts = cv2.goodFeaturesToTrack(roi, **FEATURE_PARAMS)
    if pts is None: return None
    pts[:, 0, 0] += x1
    pts[:, 0, 1] += y1
    return pts

def lk_bbox_guncelle(pts, eski_bbox):
    cx = np.median(pts[:, 0, 0])
    cy = np.median(pts[:, 0, 1])
    w, h = eski_bbox[2], eski_bbox[3]
    return bbox_sinirla((int(cx - w / 2), int(cy - h / 2), w, h))

# ══════════════════════════════════════════════════════════════════════════════
#  HİBRİT TAKİP SINIFI (BYTETRACK + LK + KALMAN)
# ══════════════════════════════════════════════════════════════════════════════
class ByteTrackLKHibrid:
    MOD_YOK   = "HEDEF ARANIYOR"
    MOD_BYTE  = "BYTETRACK"
    MOD_LK    = "LK_YEDEK"

    def __init__(self, model):
        self.model = model
        self.hedef_id = None
        self.ham_bbox = None
        self.mod = self.MOD_YOK
        
        self.kalman_bbox = None
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  
        self.kf_basladi = False

        self.prev_gray = None
        self.prev_pts = None
        self.kayip_frame_sayaci = 0
        self.son_conf = 0.0

    def guncelle(self, frame, gray):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESH, verbose=False)
        yolo_kutu_buldu = False
        en_iyi_kutu = None
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            max_alan = 0
            for box, obj_id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                
                # ÇÖZÜM 2: Uçağın ince profilini bozmadan, sadece minicik noktaları eledik!
                if w < 15 or h < 10:
                    continue
                
                alan = w * h
                
                if self.hedef_id == obj_id or self.hedef_id is None:
                    if alan > max_alan:
                        max_alan = alan
                        en_iyi_kutu = (x1, y1, w, h)
                        hedef_aday_id = obj_id
                        hedef_aday_conf = float(conf)

            if en_iyi_kutu is not None:
                self.ham_bbox = bbox_sinirla(en_iyi_kutu)
                self.hedef_id = hedef_aday_id
                self.son_conf = hedef_aday_conf
                self.mod = self.MOD_BYTE
                self.kayip_frame_sayaci = 0
                yolo_kutu_buldu = True
                
                yeni_noktalar = lk_noktalari_al(gray, self.ham_bbox)
                if yeni_noktalar is not None:
                    self.prev_pts = yeni_noktalar

        if not yolo_kutu_buldu and self.ham_bbox is not None and self.prev_pts is not None:
            if self.kayip_frame_sayaci < LK_MAX_KAYIP:
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **LK_PARAMS)
                iyi_pts = next_pts[status == 1]
                
                if len(iyi_pts) > 5:
                    self.ham_bbox = lk_bbox_guncelle(iyi_pts.reshape(-1, 1, 2), self.ham_bbox)
                    self.prev_pts = iyi_pts.reshape(-1, 1, 2)
                    self.mod = self.MOD_LK
                    self.kayip_frame_sayaci += 1
                else:
                    self._sifirla()
            else:
                self._sifirla()

        if self.ham_bbox is not None:
            x, y, w, h = self.ham_bbox
            cx, cy = x + w / 2, y + h / 2

            if not self.kf_basladi:
                self.kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                self.kf_basladi = True

            olcum = np.array([[cx], [cy]], dtype=np.float32)
            self.kf.correct(olcum)

            tahmin = self.kf.predict()
            k_cx, k_cy = tahmin[0][0], tahmin[1][0]

            k_x = int(k_cx - w / 2)
            k_y = int(k_cy - h / 2)
            self.kalman_bbox = bbox_sinirla((k_x, k_y, w, h))
        else:
            self.kalman_bbox = None
            self.kf_basladi = False

        self.prev_gray = gray.copy()
        return self.mod != self.MOD_YOK, yolo_kutu_buldu

    def _sifirla(self):
        self.hedef_id = None
        self.ham_bbox = None
        self.kalman_bbox = None
        self.prev_pts = None
        self.mod = self.MOD_YOK
        self.kayip_frame_sayaci = 0
        self.kf_basladi = False

# ══════════════════════════════════════════════════════════════════════════════
#  ANA DÖNGÜ 
# ══════════════════════════════════════════════════════════════════════════════
model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Hata: Video/Kamera açılamadı.")
    exit()

takipci = ByteTrackLKHibrid(model)
kilit_baslangic_zamani = None

print(f"TEKNOFEST KURALLARINA UYGUN HİBRİT TAKİP BAŞLADI! Çıkış için 'q'.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    basarili, yolo_yapildi = takipci.guncelle(frame, gray)

    vurus_margin_x = int(FRAME_W * 0.25)
    vurus_margin_y = int(FRAME_H * 0.10)

    av_x1, av_y1 = vurus_margin_x, vurus_margin_y
    av_x2, av_y2 = FRAME_W - vurus_margin_x, FRAME_H - vurus_margin_y
    av_merkez_x = (av_x1 + av_x2) // 2
    av_merkez_y = (av_y1 + av_y2) // 2

    cv2.rectangle(frame, (av_x1, av_y1), (av_x2, av_y2), (0, 255, 255), 2)
    cv2.circle(frame, (av_merkez_x, av_merkez_y), 4, (0, 255, 255), -1)

    zaman_ms = time.time()
    zaman_metni = time.strftime('%H:%M:%S', time.localtime(zaman_ms)) + f".{int((zaman_ms % 1) * 1000):03d}"
    
    text_size, _ = cv2.getTextSize(zaman_metni, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, zaman_metni, (FRAME_W - text_size[0] - 10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if basarili and takipci.kalman_bbox is not None:
        x, y, w, h = takipci.kalman_bbox
        cx, cy = x + w // 2, y + h // 2
        
        merkez_icerde_mi = (av_x1 <= cx <= av_x2) and (av_y1 <= cy <= av_y2)
        yuzde6_sart_saglandi = (w >= FRAME_W * HEDEF_MIN_YUZDE) or (h >= FRAME_H * HEDEF_MIN_YUZDE)

        if merkez_icerde_mi and yuzde6_sart_saglandi:
            if kilit_baslangic_zamani is None:
                kilit_baslangic_zamani = time.time()
            
            gecen_sure = time.time() - kilit_baslangic_zamani
            
            if gecen_sure >= 4.0:
                gorsel_renk = (0, 0, 255)  
                hedef_durumu = f"!!! BASARILI KILIT !!! ({gecen_sure:.1f}s)"
            else:
                gorsel_renk = (0, 0, 255)  
                hedef_durumu = f"KILITLENILIYOR... ({gecen_sure:.1f}s)"
                
        else:
            kilit_baslangic_zamani = None
            gorsel_renk = (255, 0, 0)  
            hedef_durumu = "TAKIP EDILIYOR"

        cv2.line(frame, (av_merkez_x, av_merkez_y), (cx, cy), gorsel_renk, 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), gorsel_renk, 2)
        cv2.circle(frame, (cx, cy), 3, gorsel_renk, -1) 

        cv2.putText(frame, f"{hedef_durumu}", (x, max(y - 10, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gorsel_renk, 2)

        conf_prefix = "~" if takipci.mod == takipci.MOD_LK else ""
        etiket = f"IHA ID:{takipci.hedef_id} {conf_prefix}%{takipci.son_conf * 100:.1f}"
        (_, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(frame, etiket, (x + 4, min(y + h + th + 6, FRAME_H - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, gorsel_renk, 2)
        
    else:
        kilit_baslangic_zamani = None
        cv2.putText(frame, "HEDEF ARANIYOR...", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    mod_renk = (0, 255, 0) if takipci.mod == takipci.MOD_BYTE else (0, 165, 255)
    if takipci.mod != takipci.MOD_YOK:
        cv2.putText(frame, f"MOD: {takipci.mod}", (FRAME_W - 160, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mod_renk, 2)

    cv2.imshow('Teknofest 2026 Hibrit Takip', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Sistem başarıyla sonlandırıldı.")
