import cv2
import numpy as np

class MagicWand:

    def __init__(self):
        pass

    def do_the_magic(self, image, x, y):
        """
        Extrahiert eine Kontur von Bereichen, die sich in der Helligkeit stark vom Punkt (x, y) unterscheiden.
        Gibt eine Liste von [x, y]-Koordinaten zurück.
        """
        image = self.pil_to_cv(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        region_size = 20

        # ROI um den Punkt
        x1 = max(0, x - region_size)
        y1 = max(0, y - region_size)
        x2 = min(w, x + region_size)
        y2 = min(h, y + region_size)
        roi = gray[y1:y2, x1:x2]

        # Helligkeit am Klickpunkt
        seed_val = gray[y, x]

        # Differenzbild
        diff = cv2.absdiff(roi, np.full_like(roi, seed_val))

        # Schwelle: Was gilt als „stark abweichend“?
        threshold = 20
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Konturen extrahieren
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # Größte oder nächstgelegene Kontur wählen (hier: größte)
        largest_contour = max(contours, key=cv2.contourArea).squeeze()

        # In globale Bildkoordinaten verschieben
        if len(largest_contour.shape) == 1:
            largest_contour = np.expand_dims(largest_contour, 0)

        global_contour = largest_contour + np.array([[x1, y1]])

        return global_contour.tolist()
    

    # def do_the_magic(self, image, x, y):
    #     """
    #     Extrahiert die Kontur, die den Punkt (x, y) umschließt, basierend auf Helligkeitsdifferenz.
    #     Gibt eine Liste von [x, y]-Punkten zurück.
    #     """
    #     image = self.pil_to_cv(image)

    #     if len(image.shape) == 3:
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray = image.copy()

    #     h, w = gray.shape
    #     region_size = 25

    #     # ROI um den Punkt
    #     x1 = max(0, x - region_size)
    #     y1 = max(0, y - region_size)
    #     x2 = min(w, x + region_size)
    #     y2 = min(h, y + region_size)
    #     roi = gray[y1:y2, x1:x2]

    #     # Helligkeit am Klickpunkt
    #     seed_val = gray[y, x]

    #     # Differenzbild
    #     diff = cv2.absdiff(roi, np.full_like(roi, seed_val))

    #     # Schwelle: Was gilt als „deutlich abweichend“
    #     threshold = 20
    #     _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    #     # Konturen extrahieren
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if not contours:
    #         return []

    #     # Finde die Kontur, die den Punkt enthält
    #     selected_contour = None
    #     for contour in contours:
    #         # Punkt in ROI-Koordinaten umrechnen
    #         test_result = cv2.pointPolygonTest(contour, (x - x1, y - y1), False)
    #         if test_result >= 0:
    #             selected_contour = contour
    #             break  # erste passende reicht

    #     if selected_contour is None:
    #         return []

    #     # In globale Koordinaten umrechnen
    #     selected_contour = selected_contour.squeeze()
    #     if len(selected_contour.shape) == 1:
    #         selected_contour = np.expand_dims(selected_contour, 0)

    #     global_contour = selected_contour + np.array([[x1, y1]])

    #     return global_contour.tolist()



    def pil_to_cv(self, image):
        """Konvertiert PIL.Image zu OpenCV-kompatiblem NumPy-Array (BGR)"""
        image = image.convert("RGB")
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1]  # RGB → BGR
        return open_cv_image