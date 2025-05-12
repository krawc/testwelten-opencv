from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from typing import List
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/detect-rectangle")
async def detect_green_rectangle(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert to HSV and apply green mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return JSONResponse({"error": "No green rectangle found"}, status_code=404)

    # Assume largest contour is the rectangle
    largest = max(contours, key=cv2.contourArea)

    # Approximate shape
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    # If not 4 points, use boundingRect as fallback
    if len(approx) != 4:
        x, y, w, h = cv2.boundingRect(largest)
        approx = np.array([
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]]
        ])

    # Convert to list of (x, y) tuples
    corners = [tuple(int(v) for v in pt[0]) for pt in approx]

    def expand_corners(corners, scale=0.05):
        # Compute center of the rectangle
        cx = sum([pt[0] for pt in corners]) / 4
        cy = sum([pt[1] for pt in corners]) / 4

        expanded = []
        for x, y in corners:
            dx = x - cx
            dy = y - cy
            new_x = int(round(cx + dx * (1 + scale)))
            new_y = int(round(cy + dy * (1 + scale)))
            expanded.append((new_x, new_y))
        return expanded

    # Optionally, sort corners if needed (e.g. top-left to bottom-right)
    def sort_clockwise(pts):
        pts = sorted(pts, key=lambda p: p[1])  # sort by y
        top = sorted(pts[:2], key=lambda p: p[0])
        bottom = sorted(pts[2:], key=lambda p: p[0], reverse=True)
        return top + bottom

    sorted_corners = sort_clockwise(corners)
    expanded_corners = expand_corners(sorted_corners, scale=0.05)

    return {"corners": expanded_corners}

