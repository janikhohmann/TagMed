"""
Video Frame Extractor für TagMed
Intelligente Video-Frame-Extraktion "on-the-fly" für den gallery_navigator.
"""

import os
import cv2
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """
    Intelligenter Video-Frame-Extraktor für TagMed.
    
    Features:
    - Automatische Erkennung: Frames vorhanden oder Video extrahieren?
    - Temporäre Speicherung während Patient aktiv ist
    - Kompatibel mit bestehender TagMed-Struktur
    - Automatisches Cleanup beim Patienten-Wechsel
    """
    
    def __init__(self):
        """Initialize VideoFrameExtractor."""
        self.temp_frame_dirs = {}  # Video-Pfad -> temp Verzeichnis
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
    def get_video_frames_intelligent(self, patient_id, selected_exam, selected_video, image_folder):
        """
        Intelligente Frame-Bereitstellung für Videos.
        
        Args:
            patient_id (str): Patient ID
            selected_exam (str): Ausgewähltes Exam
            selected_video (str): Ausgewähltes Video (z.B. "video1.mp4")
            image_folder (str): Basis-Ordner für Bilder
            
        Returns:
            list: Sortierte Liste der verfügbaren Frames
        """
        video_base_name = selected_video.split(".")[0]  # "video1.mp4" -> "video1"
        
        # 1. Prüfe erst, ob bereits Frames vorhanden sind
        existing_frames = self._find_existing_frames(image_folder, video_base_name)
        
        if existing_frames:
            print(f"[INFO] Gefundene vorhandene Frames für {selected_video}: {len(existing_frames)}")
            return existing_frames
        
        # 2. Wenn keine Frames vorhanden, prüfe ob Video existiert
        video_path = os.path.join(image_folder, selected_video)
        
        if not os.path.exists(video_path):
            print(f"[ERROR] Video nicht gefunden: {video_path}")
            return []
            
        if not self._is_video_file(video_path):
            print(f"[ERROR] Nicht unterstütztes Videoformat: {selected_video}")
            return []
            
        print(f"[INFO] Keine Frames gefunden, extrahiere aus Video: {selected_video}")
        
        # 3. Extrahiere Frames aus Video
        extracted_frames = self._extract_frames_from_video(
            video_path=video_path,
            video_base_name=video_base_name,
            frame_interval=1,  # Jeden 1. Frame (konfigurierbar)
            max_frames=2000     # Maximum 2000 Frames (konfigurierbar)
        )
        
        return extracted_frames
    
    def _find_existing_frames(self, image_folder, video_base_name):
        """
        Sucht nach bereits vorhandenen Frames für ein Video.
        
        Args:
            image_folder (str): Ordner mit Bildern/Frames
            video_base_name (str): Basis-Name des Videos (ohne Extension)
            
        Returns:
            list: Sortierte Liste der gefundenen Frames
        """
        try:
            all_files = os.listdir(image_folder)
            
            # Suche nach Dateien, die zum Video gehören und "frame" enthalten
            video_frames = [
                file for file in all_files 
                if video_base_name in file and "frame" in file.lower()
                and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
            ]
            
            return sorted(video_frames)
            
        except Exception as e:
            print(f"[ERROR] Fehler beim Suchen nach Frames: {e}")
            return []
    
    def _is_video_file(self, file_path):
        """Prüft ob Datei ein unterstütztes Videoformat hat."""
        return Path(file_path).suffix.lower() in self.supported_video_formats
    
    def _extract_frames_from_video(self, video_path, video_base_name, frame_interval=1, max_frames=2000):
        """
        Extrahiert Frames aus Video in temporären Ordner.
        
        Args:
            video_path (str): Pfad zur Videodatei
            video_base_name (str): Basis-Name für Frame-Dateien
            frame_interval (int): Extrahiere jeden N-ten Frame
            max_frames (int): Maximale Anzahl Frames
            
        Returns:
            list: Liste der extrahierten Frame-Dateinamen
        """
        # Prüfe ob bereits extrahiert
        if video_path in self.temp_frame_dirs:
            existing_frames = self._get_frames_from_temp_dir(video_path)
            if existing_frames:
                print(f"[INFO] Frames bereits extrahiert für: {Path(video_path).name}")
                return existing_frames
        
        # Erstelle temporären Ordner
        temp_dir = tempfile.mkdtemp(prefix=f"tagmed_frames_{video_base_name}_")
        print(f"[INFO] Extrahiere Frames nach: {temp_dir}")
        
        try:
            # Öffne Video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ERROR] Kann Video nicht öffnen: {video_path}")
                return []
            
            # Video-Eigenschaften
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"[INFO] Video Info: {total_frames} Frames, {fps:.2f} FPS, {duration:.2f}s")
            
            extracted_count = 0
            frame_number = 0
            extracted_frame_names = []
            
            # Progress bar
            expected_frames = min(total_frames // frame_interval, max_frames)
            pbar = tqdm(total=expected_frames, desc=f"Extrahiere {Path(video_path).name}", unit="frames")
            
            while frame_number < total_frames and extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extrahiere Frame in bestimmten Intervallen
                if frame_number % frame_interval == 0:
                    # Frame-Name kompatibel mit bestehender Struktur (beginne mit Frame 1)
                    frame_display_number = (frame_number // frame_interval) + 1
                    frame_filename = f"{video_base_name}_frame_{frame_display_number:06d}.jpg"
                    frame_path = Path(temp_dir) / frame_filename
                    
                    # Speichere Frame
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frame_names.append(frame_filename)
                    extracted_count += 1
                    pbar.update(1)
                
                frame_number += 1
            
            pbar.close()
            cap.release()
            
            print(f"[INFO] {extracted_count} Frames extrahiert aus {Path(video_path).name}")
            
            # Speichere temp Verzeichnis
            self.temp_frame_dirs[video_path] = temp_dir
            
            return sorted(extracted_frame_names)
            
        except Exception as e:
            print(f"[ERROR] Frame-Extraktion fehlgeschlagen: {e}")
            # Cleanup bei Fehler
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            return []
    
    def _get_frames_from_temp_dir(self, video_path):
        """
        Holt Frame-Namen aus bereits extrahiertem temp Verzeichnis.
        
        Args:
            video_path (str): Pfad zur ursprünglichen Videodatei
            
        Returns:
            list: Sortierte Liste der Frame-Dateinamen
        """
        if video_path not in self.temp_frame_dirs:
            return []
        
        temp_dir = Path(self.temp_frame_dirs[video_path])
        if not temp_dir.exists():
            return []
        
        # Finde alle Frame-Dateien
        frame_files = [
            f.name for f in temp_dir.glob('*_frame_*.jpg')
        ]
        
        return sorted(frame_files)
    
    def get_frame_path(self, patient_id, selected_exam, selected_video, frame_filename, image_folder):
        """
        Gibt den vollständigen Pfad zu einem Frame zurück.
        
        Args:
            patient_id (str): Patient ID
            selected_exam (str): Exam ID
            selected_video (str): Video Name
            frame_filename (str): Frame-Dateiname
            image_folder (str): Basis-Bildordner
            
        Returns:
            str: Vollständiger Pfad zum Frame
        """
        video_path = os.path.join(image_folder, selected_video)
        
        # Prüfe ob Frame aus temporärem Verzeichnis kommt
        if video_path in self.temp_frame_dirs:
            temp_dir = self.temp_frame_dirs[video_path]
            frame_path = os.path.join(temp_dir, frame_filename)
            if os.path.exists(frame_path):
                return frame_path
        
        # Sonst normaler Pfad im image_folder
        return os.path.join(image_folder, frame_filename)
    
    def cleanup_temp_frames(self, video_path=None):
        """
        Löscht temporäre Frame-Verzeichnisse.
        
        Args:
            video_path (str, optional): Spezifisches Video (None = alle)
        """
        if video_path:
            # Cleanup für spezifisches Video
            if video_path in self.temp_frame_dirs:
                temp_dir = self.temp_frame_dirs[video_path]
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"[INFO] Temporäre Frames gelöscht für: {Path(video_path).name}")
                    except Exception as e:
                        print(f"[WARNING] Konnte temp Verzeichnis nicht löschen: {e}")
                del self.temp_frame_dirs[video_path]
        else:
            # Cleanup für alle Videos
            for video_path, temp_dir in list(self.temp_frame_dirs.items()):
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"[INFO] Temporäre Frames gelöscht: {temp_dir}")
                    except Exception as e:
                        print(f"[WARNING] Konnte temp Verzeichnis nicht löschen: {e}")
            self.temp_frame_dirs.clear()
    
    def cleanup_on_patient_switch(self):
        """
        Cleanup beim Patienten-Wechsel.
        Löscht alle temporären Frame-Verzeichnisse.
        """
        print("[INFO] Cleanup temporärer Frames beim Patienten-Wechsel...")
        self.cleanup_temp_frames()  # Löscht alle temp Verzeichnisse
    
    def get_video_info(self, video_path):
        """
        Holt Video-Informationen ohne Frame-Extraktion.
        
        Args:
            video_path (str): Pfad zur Videodatei
            
        Returns:
            dict: Video-Eigenschaften
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': 0
            }
            
            if info['fps'] > 0:
                info['duration'] = info['total_frames'] / info['fps']
            
            cap.release()
            return info
            
        except Exception as e:
            print(f"[ERROR] Fehler beim Laden der Video-Info: {e}")
            return None
