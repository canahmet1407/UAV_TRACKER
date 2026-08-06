import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import os

print("GPU (Ekran Karti) Aktif mi?:", torch.cuda.is_available())

print("\n" + "═"*50)
print(" ✈️  ORIJINAL HIBRIT TAKIP SISTEMI (LK + CSRT) ")
print("═"*50)

# --- KLASÖR VE KAYNAK AYARI ---
VIDEO_VE_MODEL_KLASORU = '/home/ahmet/Desktop/UAV_TRACKER/MODELS_and_VİDEO'

video_secim = input(" Lutfen islenecek kaynagi secin (Kamera icin 0, Video icin 1-6): ")

# Hatalı girişlere karşı koruma ve kamera entegrasyonu
if not video_secim.isdigit() or not (0 <= int(video_secim) <= 6):
    print(" [!] Gecersiz secim! Varsayilan video (5.mp4) yukleniyor...\n")
    SECILEN_KAYNAK = os.path.join(VIDEO_KLASORU, '5.mp4')
elif video_secim == "0":
    print(" [+] Canli Kamera secildi. Sistem baslatiliyor...\n")
    SECILEN_KAYNAK = 0  # 0 id'li varsayılan web kamerası için
else:
    print(f" [+] {video_secim}.mp4 secildi. Sistem baslatiliyor...\n")
    SECILEN_KAYNAK = os.path.join(VIDEO_VE_MODEL_KLASORU, f'{video_secim}.mp4')

# ══════════════════════════════════════════════════════════════════════════════
#  AYARLAR
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH         = os.path.join(VIDEO_VE_MODEL_KLASORU, 'KENDİ_MODELİN.pt')
VIDEO_PATH         = SECILEN_KAYNAK
FRAME_W, FRAME_H   = 640, 480    

RE_DETECT_INTERVAL = 4       # Her kaç karede bir YOLO tarasın
CONF_THRESH        = 0.50    # YOLO güven eşiği
IOU_RESET_ESIGI    = 0.65    

# YENİ: SABIR SINIRI (Hayalet Takibi Önleme)
MAX_KAYIP_FRAME    = 10      # YOLO, 10 kare boyunca kilitlenen yerde uçak göremezse kilidi DÜŞÜR!

HEDEF_MIN_YUZDE    = 0.05    

# Lucas-Kanade parametreleri
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
    return (x, y, bw, bh)

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
    en_iyi, en_buyuk, en_iyi_conf = None, 0, 0.0
    en_iyi_iou = 0.0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            aday_bbox = bbox_sinirla((x1, y1, x2 - x1, y2 - y1))
            aday_alan = aday_bbox[2] * aday_bbox[3]
            aday_conf = float(box.conf[0])

            if mevcut_bbox is not None:
                skor = iou(aday_bbox, mevcut_bbox)
                if skor > en_iyi_iou and skor > 0.05: 
                    en_iyi_iou  = skor
                    en_iyi      = aday_bbox
                    en_iyi_conf = aday_conf
            else:
                if aday_alan > en_buyuk:
                    en_buyuk    = aday_alan
                    en_iyi      = aday_bbox
                    en_iyi_conf = aday_conf

    return en_iyi, en_iyi_conf

def lk_noktalari_al(gray, bbox):
    x, y, w, h = bbox
    x1 = max(0, x);  y1 = max(0, y)
    x2 = min(FRAME_W, x + w);  y2 = min(FRAME_H, y + h)
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

        self.prev_gray = gray.copy()
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

print(f"HAYALET ÖNLEYİCİLİ TAKİP BAŞLADI! Çıkış için 'q'.")

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

    cv2.rectangle(frame, (av_x1, av_y1), (av_x2, av_y2), (0, 255, 255), 2)
    cv2.putText(frame, "Av: Hedef Vurus Alani", (av_x1, av_y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if basarili and takipci.bbox is not None:
        x, y, w, h = takipci.bbox
        
        cx, cy = x + w // 2, y + h // 2
        merkez_icerde_mi = (av_x1 <= cx <= av_x2) and (av_y1 <= cy <= av_y2)
        yuzde5_sart_saglandi = (w >= FRAME_W * HEDEF_MIN_YUZDE) or (h >= FRAME_H * HEDEF_MIN_YUZDE)

        if merkez_icerde_mi and yuzde5_sart_saglandi:
            gorsel_renk = (0, 0, 255)  
            hedef_durumu = "KILITLENDI"
        else:
            gorsel_renk = (255, 0, 0)  
            hedef_durumu = "TAKIP EDILIYOR"

        cv2.rectangle(frame, (x, y), (x + w, y + h), gorsel_renk, 2)
        cv2.circle(frame, (cx, cy), 3, gorsel_renk, -1) 

        cv2.putText(frame, f"{hedef_durumu} [{takipci.mod}]", (x, max(y - 10, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gorsel_renk, 2)

        conf_prefix = "~" if takipci.conf_yasli > RE_DETECT_INTERVAL * 2 else ""
        etiket = f"IHA  {conf_prefix}%{takipci.conf * 100:.1f}"
        (_, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(frame, etiket, (x + 4, min(y + h + th + 6, FRAME_H - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, gorsel_renk, 2)

        if takipci.prev_pts is not None:
            for pt in takipci.prev_pts:
                px, py = map(int, pt.ravel())
                if 0 <= px < FRAME_W and 0 <= py < FRAME_H:
                    cv2.circle(frame, (px, py), 2, (0, 200, 255), -1)
    else:
        cv2.putText(frame, "HEDEF ARANIYOR...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if takipci.prev_pts is not None:
        cv2.putText(frame, f"LK: {len(takipci.prev_pts)} nokta  hata: {takipci.lk_hata:.2f}px",
                    (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    if yolo_yapildi:
        cv2.putText(frame, "YOLO", (FRAME_W - 70, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 0), 2)

    cv2.imshow('Teknofest 2026 Hibrit Takip', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Sistem başarıyla sonlandırıldı.")
