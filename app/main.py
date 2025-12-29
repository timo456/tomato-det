import os
import time
import tempfile
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from ultralytics import YOLO



# ========= 可改設定 =========
# 你訓練好的 best.pt 放在 weights/ 裡
WEIGHTS_PATH = os.path.join("weights", "best.pt")
# 如果你還沒放 best.pt，可以先用官方預訓練:
# WEIGHTS_PATH = "yolov8n.pt"

DEFAULT_CONF = 0.1
DEFAULT_IOU = 0.7
MAX_DET = 300
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

# 載入 YOLO
if not os.path.exists(WEIGHTS_PATH) and WEIGHTS_PATH.endswith(".pt"):
    raise FileNotFoundError(
        f"找不到權重檔：{WEIGHTS_PATH}\n"
        f"請把 best.pt 放到 weights/best.pt，或改成 yolov8n.pt 測試"
    )

model = YOLO(WEIGHTS_PATH)

@app.get("/", response_class=HTMLResponse)
def home():
    # 直接回傳前端頁面（省掉額外架 server）
    web_path = os.path.join("web", "index.html")
    if os.path.exists(web_path):
        with open(web_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>web/index.html not found</h3>"


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    max_det: int = MAX_DET,
) -> Dict[str, Any]:
    """
    收一張圖片 -> YOLOv8 推論 -> 回傳框 + 分數 + 數量 + 推論時間
    """
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
    # Ultralytics YOLOv8: model(img, conf=..., iou=..., max_det=...)
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
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "detections": detections,
    }
