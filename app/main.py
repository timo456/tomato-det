import os
import time
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
# ===========================


app = FastAPI(title="Tomato Detection API (YOLOv8)")


# 讓前端 HTML 可以直接 fetch API（同網域也行，但保險）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 掛載靜態檔案（如果你 web/ 還有 css/js/圖片，這個很有用）
# 這樣 /web/xxx 就能拿到 web/xxx
if os.path.isdir("web"):
    app.mount("/web", StaticFiles(directory="web"), name="web")


# 載入 YOLO
if not os.path.exists(WEIGHTS_PATH) and WEIGHTS_PATH.endswith(".pt"):
    raise FileNotFoundError(
        f"找不到權重檔：{WEIGHTS_PATH}\n"
        f"請把 best.pt 放到 weights/best.pt，或改成 yolov8n.pt 測試"
    )

model = YOLO(WEIGHTS_PATH)


@app.get("/", response_class=HTMLResponse)
def home():
    """
    直接回傳 web/index.html（省掉額外架 server）
    """
    web_path = os.path.join("web", "index.html")
    if os.path.exists(web_path):
        with open(web_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>web/index.html not found</h3>"


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),

    # ✅ 用 Query 明確指定預設值（/docs 會顯示正確）
    conf: float = Query(DEFAULT_CONF, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(DEFAULT_IOU, ge=0.0, le=1.0, description="IoU threshold (NMS)"),
    max_det: int = Query(DEFAULT_MAX_DET, ge=1, le=3000, description="Max detections per image"),
) -> Dict[str, Any]:
    """
    收一張圖片 -> YOLOv8 推論 -> 回傳框 + 分數 + 數量 + 推論時間
    """

    # 🔍 Debug：抓出到底是誰把值變成 0.25/0.5
    print(f"[detect] received conf={conf} iou={iou} max_det={max_det}")

    # ✅ 若你要固定用後端預設（忽略前端 query 參數）
    if FORCE_DEFAULT_PARAMS:
        conf = DEFAULT_CONF
        iou = DEFAULT_IOU
        max_det = DEFAULT_MAX_DET
        print(f"[detect] FORCE_DEFAULT_PARAMS -> conf={conf} iou={iou} max_det={max_det}")

    # 讀上傳檔案 bytes
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {
            "error": "讀取圖片失敗，請確認是 jpg/png",
            "count": 0,
            "detections": [],
        }

    h, w = img_bgr.shape[:2]

    # 推論
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
        names = r.names  # dict: cls_id -> name

        for (x1, y1, x2, y2), sc, cid in zip(boxes_xyxy, scores, cls_ids):
            detections.append({
                "x1": int(round(float(x1))),
                "y1": int(round(float(y1))),
                "x2": int(round(float(x2))),
                "y2": int(round(float(y2))),
                "score": float(sc),
                "class_id": int(cid),
                "class": str(names.get(int(cid), "unknown")),
            })

    return {
        "image_width": int(w),
        "image_height": int(h),
        "count": len(detections),
        "inference_ms": float(infer_ms),

        # ✅ 回傳實際用到的參數（前端顯示會準）
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),

        "detections": detections,
    }
