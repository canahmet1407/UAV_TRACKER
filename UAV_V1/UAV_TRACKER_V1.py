import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time

print("GPU (Ekran Karti) Aktif mi?:", torch.cuda.is_available())

print("\n" + "═"*50)
print(" ✈️  TEKNOFEST SAVAŞAN İHA - HİBRİT TAKİP SİSTEMİ ")
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

RE_DETECT_INTERVAL = 4        
CONF_THRESH        = 0.50     
IOU_RESET_ESIGI    = 0.65     
MAX_KAYIP_FRAME    = 10       
HEDEF_MIN_YUZDE    = 0.06     

LK_PARAMS = dict(
    winSize  = (21, 21),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

FEATURE_PARAMS = dict(
    maxCorners   = 300,
    qualityLevel = 0.25,
    minDistance  = 5,
    blockSize    = 7
)

LK_MIN_POINTS  = 5     
LK_REINIT_PTS  = 30    
LK_MAX_HATA    = 1.3   
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

def bbox_gecerli_mi(bbox):
    x, y, w, h = bbox
    return (w >= BBOX_MIN_BOYUT and h >= BBOX_MIN_BOYUT and
            x >= 0 and y >= 0 and
            x + w <= FRAME_W and y + h <= FRAME_H)

def iou(b1, b2):
    ax1, ay1 = b1[0], b1[1]
    ax2, ay2 = ax1 + b1[2], ay1 + b1[3]
    bx1, by1 = b2[0], b2[1]
    bx2, by2 = bx1 + b2[2], by1 + b2[3]
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0

def yolo_tara(model, frame, mevcut_bbox=None):
    results = model(frame, conf=CONF_THRESH, verbose=False)
    en_iyi = None
    en_iyi_conf = 0.0

    if mevcut_bbox is not None:
        en_iyi_iou = 0.0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                aday_bbox = bbox_sinirla((x1, y1, x2 - x1, y2 - y1))
                aday_conf = float(box.conf[0])
                
                skor_iou = iou(aday_bbox, mevcut_bbox)
                if skor_iou > en_iyi_iou and skor_iou > 0.05: 
                    en_iyi_iou = skor_iou
                    en_iyi = aday_bbox
                    en_iyi_conf = aday_conf
                    
        if en_iyi is not None:
            return en_iyi, en_iyi_conf

    en_iyi_skor = -1.0
    
    W_ALAN   = 0.30  
    W_MERKEZ = 0.25  
    W_CONF   = 0.20  
    W_KURAL  = 0.15  
    W_KENAR  = 0.10  

    frame_merkez_x = FRAME_W / 2
    frame_merkez_y = FRAME_H / 2
    max_mesafe = np.sqrt(frame_merkez_x**2 + frame_merkez_y**2)
    max_alan = FRAME_W * FRAME_H

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            aday_bbox = bbox_sinirla((x1, y1, x2 - x1, y2 - y1))
            aday_w = aday_bbox[2]
            aday_h = aday_bbox[3]
            aday_alan = aday_w * aday_h
            aday_conf = float(box.conf[0])

            skor_alan = min(1.0, aday_alan / (max_alan * 0.15))
            skor_conf = aday_conf
            
            aday_merkez_x = aday_bbox[0] + aday_w / 2
            aday_merkez_y = aday_bbox[1] + aday_h / 2
            mesafe = np.sqrt((aday_merkez_x - frame_merkez_x)**2 + (aday_merkez_y - frame_merkez_y)**2)
            skor_merkez = max(0.0, 1.0 - (mesafe / max_mesafe))

            if (aday_w >= FRAME_W * HEDEF_MIN_YUZDE) or (aday_h >= FRAME_H * HEDEF_MIN_YUZDE):
                skor_kural = 1.0
            else:
                skor_kural = 0.0

            min_kenar_uzakligi = min(aday_bbox[0], aday_bbox[1], 
                                     FRAME_W - (aday_bbox[0] + aday_w), 
                                     FRAME_H - (aday_bbox[1] + aday_h))
            skor_kenar = min(1.0, min_kenar_uzakligi / 50.0)

            toplam_skor = (skor_conf * W_CONF) + \
                          (skor_alan * W_ALAN) + \
                          (skor_merkez * W_MERKEZ) + \
                          (skor_kural * W_KURAL) + \
                          (skor_kenar * W_KENAR)

            if toplam_skor > en_iyi_skor:
                en_iyi_skor = toplam_skor
                en_iyi = aday_bbox
                en_iyi_conf = aday_conf

    return en_iyi, en_iyi_conf

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
    if pts is None:
        return None
    pts[:, 0, 0] += x1
    pts[:, 0, 1] += y1
    return pts

def lk_geri_dogrula(prev_gray, gray, prev_pts):
    next_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **LK_PARAMS)
    if next_pts is None or st_fwd is None: return None, None
    back_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, next_pts, None, **LK_PARAMS)
    if back_pts is None or st_bwd is None: return None, None

    hata = np.linalg.norm(prev_pts - back_pts, axis=2).flatten()
    iyi  = (st_fwd.flatten() == 1) & (st_bwd.flatten() == 1) & (hata < LK_MAX_HATA)

    if np.sum(iyi) < LK_MIN_POINTS: return None, None
    return next_pts[iyi].reshape(-1, 1, 2), hata[iyi].mean()

def lk_bbox_guncelle(pts, eski_bbox):
    cx = np.median(pts[:, 0, 0])
    cy = np.median(pts[:, 0, 1])
    w, h = eski_bbox[2], eski_bbox[3]
    return bbox_sinirla((int(cx - w / 2), int(cy - h / 2), w, h))

# ══════════════════════════════════════════════════════════════════════════════
#  HİBRİT TAKİP SINIFI
# ══════════════════════════════════════════════════════════════════════════════

class HibridTakipci:
    MOD_YOK  = "HEDEF ARANIYOR"
    MOD_LK   = "LK AKIS"
    MOD_CSRT = "CSRT YEDEK"

    def __init__(self, model):
        self.model      = model
        self.mod        = self.MOD_YOK
        self.bbox       = None
        
        self.kalman_bbox = None
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  
        self.kf_basladi = False

        self.prev_gray  = None
        self.prev_pts   = None
        self.lk_hata    = 0.0
        self.csrt       = None
        self.csrt_aktif = False
        self.conf       = 0.0
        self.conf_yasli = 0

    def _baslat(self, frame, gray, bbox):
        bbox = bbox_sinirla(bbox)
        if not bbox_gecerli_mi(bbox): return False
        self.bbox     = bbox
        self.prev_pts = lk_noktalari_al(gray, bbox)
        self.prev_gray = gray.copy()
        self.csrt      = cv2.legacy.TrackerCSRT_create()
        self.csrt.init(frame, bbox)
        self.csrt_aktif = True
        self.mod        = self.MOD_LK
        self.lk_hata    = 0.0
        self.conf_yasli = 0
        self.kf_basladi = False 
        return True

    def _lk_adim(self, gray):
        if self.prev_pts is None or self.prev_gray is None: return False
        iyi_pts, hata = lk_geri_dogrula(self.prev_gray, gray, self.prev_pts)
        if iyi_pts is None: return False
        self.lk_hata  = hata
        self.prev_pts = iyi_pts
        yeni_bbox = lk_bbox_guncelle(self.prev_pts, self.bbox)
        if not bbox_gecerli_mi(yeni_bbox): return False
        self.bbox = yeni_bbox
        
        if len(self.prev_pts) < LK_REINIT_PTS:
            yeni = lk_noktalari_al(gray, self.bbox)
            if yeni is not None: self.prev_pts = yeni
        return True

    def _csrt_adim(self, frame, gray):
        if not self.csrt_aktif or self.csrt is None: return False
        ok, csrt_bbox = self.csrt.update(frame)
        if not ok: return False
        csrt_bbox = bbox_sinirla(tuple(map(int, csrt_bbox)))
        if not bbox_gecerli_mi(csrt_bbox): return False
        self.bbox = csrt_bbox
        yeni = lk_noktalari_al(gray, self.bbox)
        if yeni is not None: self.prev_pts = yeni
        return True

    def guncelle(self, frame, gray, frame_count):
        lk_ok = False
        csrt_ok = False
        self.conf_yasli += 1

        if self.mod != self.MOD_YOK:
            lk_ok = self._lk_adim(gray)
            if lk_ok:
                self.mod = self.MOD_LK
                if self.csrt_aktif and self.csrt is not None:
                    ok, _ = self.csrt.update(frame)
                    if not ok:
                        self.csrt = cv2.legacy.TrackerCSRT_create()
                        self.csrt.init(frame, self.bbox)
            else:
                csrt_ok = self._csrt_adim(frame, gray)
                self.mod = self.MOD_CSRT if csrt_ok else self.MOD_YOK

        yolo_yapildi = False
        if frame_count % RE_DETECT_INTERVAL == 0 or self.mod == self.MOD_YOK:
            mevcut_konum = self.bbox if self.mod != self.MOD_YOK else None
            yolo_bbox, yolo_conf = yolo_tara(self.model, frame, mevcut_bbox=mevcut_konum)
            
            if yolo_bbox is not None:
                if self.bbox is not None and (lk_ok or csrt_ok):
                    self.csrt = cv2.legacy.TrackerCSRT_create()
                    self.csrt.init(frame, yolo_bbox)
                    self.csrt_aktif = True
                else:
                    self._baslat(frame, gray, yolo_bbox)
                
                self.conf       = yolo_conf
                self.conf_yasli = 0
                yolo_yapildi    = True

        if self.conf_yasli > MAX_KAYIP_FRAME:
            self.mod = self.MOD_YOK
            self.bbox = None
            self.csrt_aktif = False
            self.kf_basladi = False
            self.kalman_bbox = None

        self.prev_gray = gray.copy()

        if self.bbox is not None:
            x, y, w, h = self.bbox
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

        return self.mod != self.MOD_YOK, yolo_yapildi

# ══════════════════════════════════════════════════════════════════════════════
#  ANA DÖNGÜ 
# ══════════════════════════════════════════════════════════════════════════════

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Hata: Video/Kamera açılamadı.")
    exit()

takipci     = HibridTakipci(model)
frame_count = 0

kilit_baslangic_zamani = None

print(f"TEKNOFEST KURALLARINA UYGUN HİBRİT TAKİP BAŞLADI! Çıkış için 'q'.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    basarili, yolo_yapildi = takipci.guncelle(frame, gray, frame_count)

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
        
        tamamen_icerde_mi = (x >= av_x1) and (y >= av_y1) and ((x + w) <= av_x2) and ((y + h) <= av_y2)
        yuzde6_sart_saglandi = (w >= FRAME_W * HEDEF_MIN_YUZDE) or (h >= FRAME_H * HEDEF_MIN_YUZDE)

        if tamamen_icerde_mi and yuzde6_sart_saglandi:
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

        conf_prefix = "~" if takipci.conf_yasli > RE_DETECT_INTERVAL * 2 else ""
        etiket = f"IHA  {conf_prefix}%{takipci.conf * 100:.1f}"
        (_, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(frame, etiket, (x + 4, min(y + h + th + 6, FRAME_H - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, gorsel_renk, 2)
        
    else:
        kilit_baslangic_zamani = None
        cv2.putText(frame, "HEDEF ARANIYOR...", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if yolo_yapildi:
        cv2.putText(frame, "YOLO", (FRAME_W - 70, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 0), 2)

    cv2.imshow('Teknofest 2026 Hibrit Takip', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Sistem başarıyla sonlandırıldı.")
