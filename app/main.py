import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# ========= 可改設定 =========
WEIGHTS_PATH = os.path.join("weights", "best.pt")

DEFAULT_CONF = 0.10
DEFAULT_IOU = 0.70
DEFAULT_MAX_DET = 300

# ✅ 如果你想「不管前端送什麼，都固定用預設」就改 True
FORCE_DEFAULT_PARAMS = True

# ✅ 趨勢紀錄檔（JSON Lines：一行一筆 JSON）
HISTORY_PATH = "history.jsonl"
# ===========================


app = FastAPI(title="Tomato Monitoring & Harvest Suggestion API (YOLOv8)")


# 讓前端 HTML 可以直接 fetch API（同網域也行，但保險）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 掛載靜態檔案（如果你 web/ 還有 css/js/圖片，這個很有用）
if os.path.isdir("web"):
    app.mount("/web", StaticFiles(directory="web"), name="web")


# ============ 工具函式：影像品質 ============

def image_quality_warnings(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    回傳影像品質分數與警告（模糊/亮度）
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 模糊檢測：Laplacian variance 越低越模糊
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 亮度檢測：平均灰階
    brightness = float(gray.mean())

    warnings = []
    score = 100.0

    # 門檻可依你資料集微調
    if blur_var < 80:
        warnings.append("影像可能過模糊：建議重新對焦、拿穩、或提高快門")
        score -= 35
    elif blur_var < 120:
        warnings.append("影像略偏糊：建議再拍一張確認")
        score -= 15

    if brightness < 60:
        warnings.append("光線不足（偏暗）：建議補光或移至較亮處")
        score -= 25
    elif brightness > 200:
        warnings.append("影像過亮（可能過曝）：建議降低曝光或避開直射光")
        score -= 15

    score = max(0.0, min(100.0, score))
    return {
        "quality_score": score,
        "blur_var": blur_var,
        "brightness": brightness,
        "warnings": warnings
    }


def coverage_warning(img_w: int, img_h: int, detections: List[Dict[str, Any]]) -> List[str]:
    """
    用框的總面積占比判斷是否拍太近/太遠（拍攝引導）
    """
    if not detections:
        return ["未偵測到番茄：可能角度不對、距離太遠、或光線不足（可試著靠近/補光）"]

    img_area = max(1, img_w * img_h)
    box_area_sum = 0

    for d in detections:
        box_area_sum += max(0, d["x2"] - d["x1"]) * max(0, d["y2"] - d["y1"])

    ratio = box_area_sum / img_area
    warns = []
    if ratio > 0.45:
        warns.append("畫面太近：番茄佔比偏高，建議拉遠一點拍")
    elif ratio < 0.02:
        warns.append("畫面太遠：番茄佔比偏低，建議靠近一點拍")
    return warns


# ============ 工具函式：成熟度(顏色) ============

def maturity_from_crop(img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Dict[str, Any]:
    """
    用 HSV 顏色比例粗估成熟度：red_ratio 越高越成熟
    - ripe:    red_ratio >= 0.45
    - turning: red_ratio >= 0.20
    - unripe:  else
    """
    H, W = img_bgr.shape[:2]
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H, y2))

    if x2 <= x1 or y2 <= y1:
        return {"maturity": "unknown", "red_ratio": 0.0, "green_ratio": 0.0}

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return {"maturity": "unknown", "red_ratio": 0.0, "green_ratio": 0.0}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 過濾低飽和/低亮度（避免把背景、陰影算進來）
    sat = hsv[..., 1]
    val = hsv[..., 2]
    valid = (sat > 40) & (val > 40)

    if int(valid.sum()) < 50:
        return {"maturity": "unknown", "red_ratio": 0.0, "green_ratio": 0.0}

    h = hsv[..., 0]

    # OpenCV H: 0~179
    red = ((h <= 10) | (h >= 170)) & valid
    green = ((h >= 35) & (h <= 85)) & valid

    total = float(valid.sum())
    red_ratio = float(red.sum()) / total
    green_ratio = float(green.sum()) / total

    if red_ratio >= 0.45:
        maturity = "ripe"
    elif red_ratio >= 0.20:
        maturity = "turning"
    else:
        maturity = "unripe"

    return {"maturity": maturity, "red_ratio": red_ratio, "green_ratio": green_ratio}


# ============ 載入 YOLO ============

if not os.path.exists(WEIGHTS_PATH) and WEIGHTS_PATH.endswith(".pt"):
    raise FileNotFoundError(
        f"找不到權重檔：{WEIGHTS_PATH}\n"
        f"請把 best.pt 放到 weights/best.pt，或改成 yolov8n.pt 測試"
    )

model = YOLO(WEIGHTS_PATH)


# ============ 路由 ============

@app.get("/", response_class=HTMLResponse)
def home():
    """直接回傳 web/index.html（省掉額外架 server）"""
    web_path = os.path.join("web", "index.html")
    if os.path.exists(web_path):
        with open(web_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>web/index.html not found</h3>"


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(DEFAULT_CONF, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(DEFAULT_IOU, ge=0.0, le=1.0, description="IoU threshold (NMS)"),
    max_det: int = Query(DEFAULT_MAX_DET, ge=1, le=3000, description="Max detections per image"),
) -> Dict[str, Any]:
    """
    收一張圖片 -> YOLOv8 推論 -> 回傳框 + 成熟度 + 品質提醒 + 採收建議 + 紀錄
    """

    print(f"[detect] received conf={conf} iou={iou} max_det={max_det}")

    if FORCE_DEFAULT_PARAMS:
        conf = DEFAULT_CONF
        iou = DEFAULT_IOU
        max_det = DEFAULT_MAX_DET
        print(f"[detect] FORCE_DEFAULT_PARAMS -> conf={conf} iou={iou} max_det={max_det}")

    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {"error": "讀取圖片失敗，請確認是 jpg/png", "count": 0, "detections": []}

    h, w = img_bgr.shape[:2]

    # ✅ 影像品質（先算，不影響推論）
    quality = image_quality_warnings(img_bgr)

    # ✅ 推論
    t0 = time.time()
    results = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )
    infer_ms = (time.time() - t0) * 1000.0

    r = results[0]

    detections: List[Dict[str, Any]] = []
    if r.boxes is not None and len(r.boxes) > 0:
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names

        for (x1, y1, x2, y2), sc, cid in zip(boxes_xyxy, scores, cls_ids):
            x1i = int(round(float(x1)))
            y1i = int(round(float(y1)))
            x2i = int(round(float(x2)))
            y2i = int(round(float(y2)))

            # ✅ 成熟度判斷（顏色 proxy）
            maturity_info = maturity_from_crop(img_bgr, x1i, y1i, x2i, y2i)

            detections.append({
                "x1": x1i,
                "y1": y1i,
                "x2": x2i,
                "y2": y2i,
                "score": float(sc),
                "class_id": int(cid),
                "class": str(names.get(int(cid), "unknown")),
                "maturity": maturity_info["maturity"],
                "red_ratio": float(maturity_info["red_ratio"]),
                "green_ratio": float(maturity_info["green_ratio"]),
            })

    # ✅ 拍攝引導（太近/太遠/沒拍到）
    quality["warnings"] += coverage_warning(w, h, detections)

    # ✅ 成熟度總結 + 採收建議
    ripe_count = sum(1 for d in detections if d.get("maturity") == "ripe")
    turning_count = sum(1 for d in detections if d.get("maturity") == "turning")
    unripe_count = sum(1 for d in detections if d.get("maturity") == "unripe")
    unknown_count = sum(1 for d in detections if d.get("maturity") == "unknown")

    if len(detections) == 0:
        harvest = "無法判斷：未偵測到番茄（建議調整角度/距離/光線）"
    else:
        if ripe_count >= max(1, int(0.4 * len(detections))):
            harvest = "建議採收：成熟（偏紅）比例偏高"
        elif turning_count > ripe_count:
            harvest = "可分批採收：有不少轉色果（可再觀察 2–3 天）"
        else:
            harvest = "建議再觀察：多數仍偏生（偏綠）"

    maturity_summary = {
        "ripe": ripe_count,
        "turning": turning_count,
        "unripe": unripe_count,
        "unknown": unknown_count,
        "harvest_suggestion": harvest
    }

    # ✅ 趨勢紀錄（寫入 history.jsonl）
    try:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "count": len(detections),
            "ripe": ripe_count,
            "turning": turning_count,
            "unripe": unripe_count,
            "unknown": unknown_count,
            "quality_score": float(quality["quality_score"]),
            "conf": float(conf),
            "iou": float(iou),
            "max_det": int(max_det),
            "inference_ms": float(infer_ms),
        }
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # 不要因為寫檔失敗就讓偵測失敗
        quality["warnings"].append(f"歷史紀錄寫入失敗：{e}")

    return {
        "image_width": int(w),
        "image_height": int(h),
        "count": len(detections),
        "inference_ms": float(infer_ms),

        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),

        # ✅ 新增：品質提醒 + 成熟度總結
        "quality": quality,
        "maturity_summary": maturity_summary,

        "detections": detections,
    }


@app.get("/history")
def history(limit: int = Query(50, ge=1, le=500, description="How many recent records to return")) -> List[Dict[str, Any]]:
    """
    取得最近 N 筆偵測紀錄（給前端畫趨勢）
    """
    if not os.path.exists(HISTORY_PATH):
        return []

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()[-limit:]

    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out
