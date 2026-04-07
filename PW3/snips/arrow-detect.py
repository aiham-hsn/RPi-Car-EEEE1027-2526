for cnt in contours:
    area = cv2.contourArea(cnt)
    if 1000 < area < 15000:
        hull = cv2.convexHull(cnt)
        solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        if 0.52 <= solidity <= 0.75:
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                bX, bY = x + (w/2), y + (h/2)
                if abs(cX - bX) > abs(cY - bY):
                    current_identity = "Arrow Right" if cX > bX else "Arrow Left"
                else:
                    current_identity = "Arrow Down" if cY > bY else "Arrow Up"
                break