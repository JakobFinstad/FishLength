# workflow.py
import cv2
import numpy as np

RULER_LEN_MM = 313.0   
RULER_WID_MM = 37.0


def largest_mask(masks):
    """Pick the biggest instance from an (N,H,W) boolean array."""
    if len(masks) == 0:
        return None
    return masks[np.argmax(masks.reshape(len(masks), -1).sum(axis=1))]


def largest_contour(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def order_pts(p):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    s, d = p.sum(axis=1), np.diff(p, axis=1).ravel()
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)


def ruler_orientation(ruler_mask):
    """Same logic as the Roboflow custom python block (bounding box width vs height)."""
    _, _, w, h = cv2.boundingRect(ruler_mask.astype(np.uint8))
    return "horizontal" if w > h else "vertical"


def ruler_quad(ruler_mask):
    """4 corners of the ruler (the 'dynamic_zone' step). Tries a true quadrilateral
    first (keeps perspective distortion), falls back to the min-area rectangle."""
    cnt = largest_contour(ruler_mask)
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)


def ruler_homography(quad, ruler_len_mm=RULER_LEN_MM, ruler_wid_mm=RULER_WID_MM):
    """Homography mapping image pixels -> millimetres in the ruler's plane."""
    q = order_pts(quad)
    if np.linalg.norm(q[3] - q[0]) > np.linalg.norm(q[1] - q[0]):
        q = np.roll(q, -1, axis=0)           # make q[0]->q[1] the long edge
    dst = np.array([[0, 0], [ruler_len_mm, 0],
                    [ruler_len_mm, ruler_wid_mm], [0, ruler_wid_mm]], np.float32)
    return cv2.getPerspectiveTransform(q, dst)


def to_mm(contour, M):
    return cv2.perspectiveTransform(contour.astype(np.float32), M)


def measure_fish(fish_mask, M):
    cnt = largest_contour(fish_mask)
    if cnt is None:
        return None
    (_, _), (w, h), _ = cv2.minAreaRect(to_mm(cnt, M))
    return {"length_mm": max(w, h), "width_mm": min(w, h)}


def measure_image(per_class):
    """per_class = {"ruler": (N,H,W) bool, "fish": (M,H,W) bool} from segment_image()."""
    ruler = largest_mask(per_class["ruler"])
    if ruler is None or len(per_class["fish"]) == 0:
        return None

    M = ruler_homography(ruler_quad(ruler))

    # Self-check: the ruler itself, pushed through M, should come out ~ LEN x WID
    (_, _), (rw, rh), _ = cv2.minAreaRect(to_mm(largest_contour(ruler), M))
    ruler_ok = abs(max(rw, rh) - RULER_LEN_MM) / RULER_LEN_MM < 0.05

    fish = [measure_fish(m, M) for m in per_class["fish"]]
    return {"orientation": ruler_orientation(ruler), "ruler_ok": ruler_ok,
            "fish": [f for f in fish if f]}
