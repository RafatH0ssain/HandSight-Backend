import cv2
import numpy as np
import math

class HandwritingAnalyzer:
    def __init__(self, image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        self.original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. RESIZE IMAGE (Standardize inputs)
        # If the image is too huge/small, our math breaks. 
        # We resize to a width of 1000px, maintaining aspect ratio.
        h, w = self.original.shape[:2]
        if w > 1000 or w < 500:
            scale = 1000 / w
            new_h = int(h * scale)
            self.original = cv2.resize(self.original, (1000, new_h))

        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # 2. Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(
            self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. Smart Inversion (Paper Detector)
        total_pixels = thresh.size
        white_pixels = cv2.countNonZero(thresh)
        
        if white_pixels > (total_pixels / 2):
            print("🔍 DEBUG: Detected White Background. Inverting colors...")
            self.thresh = cv2.bitwise_not(thresh)
        else:
            print("🔍 DEBUG: Detected Dark Background. Keeping as is.")
            self.thresh = thresh
            
        # Save for you to see
        cv2.imwrite("debug_final.jpg", self.thresh)
        
        self.height, self.width = self.gray.shape

    def extract_features(self) -> dict[str, float]:
        return {
            "Feature_1": self._get_slant(),
            "Feature_2": self._get_word_spacing(),
            "Feature_3": self._get_pressure(),
            "Feature_8": self._get_letter_size()
        }

    def _get_slant(self) -> float:
        """
        Calculates slant using Hough Lines.
        Now that colors are fixed, this works best for ALL handwriting types (cursive or print).
        """
        # 1. Isolate Vertical Strokes
        # Filter out horizontal lines (like crossbars) using a vertical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        eroded = cv2.erode(self.thresh, kernel, iterations=1)
        
        # 2. Detect Edges on the vertical strokes
        edges = cv2.Canny(eroded, 50, 150, apertureSize=3)
        
        # 3. Probabilistic Hough Transform
        # We look for lines at least 30px long.
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

        if lines is None:
            # Fallback: Relax constraints if first pass fails
            print("⚠️ DEBUG: strict scan failed, trying relaxed scan...")
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=5)
            
        if lines is None:
            print("⚠️ DEBUG: No lines found at all. Defaulting to 0.5")
            return 0.5

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Avoid divide by zero
            if x2 - x1 == 0:
                angle = 90.0
            else:
                angle = np.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # Normalize to positive 0-180
            if angle < 0: angle += 180
            
            # Filter for valid handwriting slant (45 to 135 degrees)
            # 45=Extreme Right Slant, 90=Vertical, 135=Extreme Left Slant
            if 45 < angle < 135:
                angles.append(angle)

        if not angles:
            print("⚠️ DEBUG: Lines found, but none were vertical enough.")
            return 0.5

        avg_angle = np.mean(angles)
        print(f"✅ DEBUG: Calculated Slant from {len(angles)} lines. Avg: {avg_angle:.2f}°")

        # Normalize logic:
        # 135 (Left) -> 0.0
        # 90 (Vertical) -> 0.5
        # 45 (Right) -> 1.0
        normalized = (135 - avg_angle) / 90.0
        return float(np.clip(normalized, 0.0, 1.0))

    def _get_pressure(self) -> float:
        # Simple ink density check
        text_pixels = self.gray[self.thresh > 0]
        if text_pixels.size == 0: return 0.5
        avg_intensity = np.mean(text_pixels)
        return float(np.clip(1.0 - (avg_intensity / 255.0), 0.0, 1.0))

    def _get_word_spacing(self) -> float:
        # Dilate to connect letters into words
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        dilated = cv2.dilate(self.thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2: return 0.5
        
        # Sort by X position
        boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
        gaps = []
        for i in range(len(boxes) - 1):
            gap = boxes[i+1][0] - (boxes[i][0] + boxes[i][2])
            if 5 < gap < 100: gaps.append(gap)
            
        if not gaps: return 0.5
        return float(np.clip(np.mean(gaps) / 50.0, 0.0, 1.0))

    def _get_letter_size(self) -> float:
        contours, _ = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return 0.5
        heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 10]
        if not heights: return 0.5
        return float(np.clip(np.mean(heights) / 50.0, 0.0, 1.0))