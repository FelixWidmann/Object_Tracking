#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

# TBD True MEMOT

try:
    from trackers.memot.tracker_wrapper import MeMOT_Tracker
except Exception as e:
    print("ERROR importing MeMOT_Tracker.")
    raise


# Visual Output

def build_track_grid(track_images, max_cols=4):
    if len(track_images) == 0:
        return None

    cell_h = 240
    cell_w = 240
    resized = [cv2.resize(img, (cell_w, cell_h)) for img in track_images]

    rows = []
    for i in range(0, len(resized), max_cols):
        row_imgs = resized[i:i+max_cols]
        while len(row_imgs) < max_cols:
            row_imgs.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
        rows.append(np.hstack(row_imgs))

    return np.vstack(rows)


# YOLO import

def try_load_yolo(model_path=None):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics YOLO not installed.")
        return None

    if model_path is None:
        model_path = "yolov8n.pt"

    print(f"Loading YOLO model: {model_path}")
    return YOLO(model_path)


def boxes_from_yolo_result(res):
    """Convert YOLO result → [{'bbox':[x1,y1,x2,y2],'score':float}, ...]"""
    dets = []
    for box in res.boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf)
        dets.append({"bbox": xyxy, "score": conf})
    return dets


def main(args):

    # Load vid
    
    if not os.path.exists(args.video):
        print("Video not found:", args.video)
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Opened video: {total_frames} frames, {fps:.2f} FPS, {W}x{H}")

    yolo = None
    if args.use_yolo:
        yolo = try_load_yolo(args.yolo_model)
        if yolo is None:
            print("Failed to load YOLO. Exiting.")
            sys.exit(1)

    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    tracker = MeMOT_Tracker(
        mode="yolo" if args.use_yolo else "memot",
        Ts=3,
        Tl=args.memory_len,
        max_age=args.max_age,
        conf_th=args.confidence,
        device=device
    )

    # CSV output init
    
    csv_file = open(args.output, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "frame", "timestamp", "track_id",
        "x1", "y1", "x2", "y2", "score"
    ])
    csv_writer.writeheader()

    frame_id = 0
    t0 = time.time()


    # Video Processing
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps

        # Detections
        if args.use_yolo:
            yolo_out = yolo(frame, verbose=False)
            detections = boxes_from_yolo_result(yolo_out[0]) if len(yolo_out) > 0 else []
        else:
            detections = []  # Full MeMOT mode should autogenerate detections itself

        # DEBUG TOGGLE HERE?
        # print(f"Frame {frame_id}: YOLO detections = {len(detections)}")

        # Tracking
        active_tracks = tracker.update(detections, frame_id, frame)

        # print("Active tracks:", len(active_tracks))

        # Extract crops for visualization
        track_images = []
        for tr in active_tracks:
            x1, y1, x2, y2 = map(int, tr.bbox)
            x1 = max(0, min(x1, W-1))
            y1 = max(0, min(y1, H-1))
            x2 = max(0, min(x2, W-1))
            y2 = max(0, min(y2, H-1))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            panel = cv2.copyMakeBorder(
                crop, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0)
            )

            cv2.putText(panel, f"ID {tr.id}", (5,18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0),1)
            cv2.putText(panel, f"{tr.score:.2f}", (5,38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255),1)

            track_images.append(panel)

            # CSV row
            csv_writer.writerow({
                "frame": frame_id,
                "timestamp": f"{timestamp:.4f}",
                "track_id": tr.id,
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "score": float(tr.score),
            })

        # Display track crops
        for i, panel in enumerate(track_images):
            cv2.imshow(f"Track {i}", panel)

        grid_img = build_track_grid(track_images)
        if grid_img is not None:
            cv2.imshow("Tracks Grid", grid_img)

        # Draw boxes on main frame
        if args.display:
            show = frame.copy()
            for tr in active_tracks:
                x1, y1, x2, y2 = map(int, tr.bbox)
                cv2.rectangle(show, (x1,y1), (x2,y2), (0,255,0),2)
                cv2.putText(show, f"ID:{tr.id}", (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0),1)

            cv2.imshow("MeMOT", show)
            if cv2.waitKey(1) == 27:
                break

        frame_id += 1


    # Cleanup
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

    dt = time.time() - t0
    print(f"Processed {frame_id} frames in {dt:.1f}s ({frame_id/dt:.2f} FPS)")
    print("CSV saved:", args.output)



# CLI

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--use_yolo", action="store_true")
    p.add_argument("--yolo_model", default=None)
    p.add_argument("--display", action="store_true")
    p.add_argument("--max_age", type=int, default=30)
    p.add_argument("--memory_len", type=int, default=24)
    p.add_argument("--confidence", type=float, default=0.01)
    args = p.parse_args()

    main(args)
