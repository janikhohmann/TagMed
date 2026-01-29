# TagMed Manual

## What is TagMed?

TagMed is a comprehensive medical annotation tool designed for annotating medical images and videos. It provides advanced features for creating bounding box and polygon annotations, mask prediction, and intelligent video tracking with SAM2, MedSAM2 and SAM3 integration.

### Key Features
- **Dual Interface**: Separate tabs for images and videos
- **Multiple Annotation Types**: Bounding boxes and polygons
- **Advanced Zoom & Pan**: 100-300% zoom with pan support
- **Intelligent Tracking**: SAM2, SAM3, and MedSAM2 integration for video annotation tracking
- **Mask Prediction**: Automatic mask generation from annotations
- **Video Frame Management**: Intelligent frame extraction and caching
- **Medical Reports**: Integrated display of medical examination reports
- **Progress Tracking**: Color-coded annotation progress indicators

---

## Installation

### Requirements
- Python 3.8 or higher
- CUDA-capable GPU (optional, for SAM2/MedSAM2 tracking)
- 8GB RAM minimum (16GB recommended)

### Python Dependencies
```bash
python setup_TagMed.py
```

Key dependencies:
- tkinter (GUI framework)
- pandas (data management)
- Pillow (image processing)
- torch (for SAM2/MedSAM2)
- opencv-python (video processing)

### Configuration
1. Edit `src/user_settings.json` to configure:
   - Image folder paths
   - Annotation table file location
   - Medical report file location
   - Class list for annotations
   - Image display size (default: 600x600)

Example configuration:
```json
{
  "selected_image_folder": "/path/to/images",
  "selected_anno_table_file": "/path/to/annotations.csv",
  "selected_medical_report_file": "/path/to/reports.csv",
  "class_list": ["Lesion", "Tumor", "Organ"],
  "image_size": [600, 600]
}
```
All configurations can be made within TagMed using the graphical user interface. To get started, it is recommended that you create an automatically generated annotation table via the graphical user interface in order to index your database. 

---

## Start

### Launching TagMed
```bash
cd src
python main.py
```

### Initial Setup
1. **Select Patient**: Click on a patient ID from the main patient list
2. **Select Examination**: Choose an exam from the dropdown menu
3. **Choose Tab**: Switch between "Images" and "Videos" tabs

*INSERT SCREENSHOT HERE*

---

## Interface Overview

### Layout
```
┌─────────────────────────────────────────────────────┐
│ Patient Selection & Exam Dropdown                   │
├───────────┬──────────────────┬──────────────────────┤
│ Image/    │  Medical Report  │  Canvas Display      │
│ Video     │  Description     │  (Image/Video)       │
│ List      │                  │  - Header with Name  │
│           │                  │  - Zoom % Display    │
│           │                  │  - Main Canvas       │
├───────────┴──────────────────┴──────────────────────┤
│ Annotation Controls                                 │
│ - Type/Class Dropdowns                              │
│ - Add/Delete/Save Buttons                           │
│ - Modify Mode Toggle                                │
│ - Objects List                                      │
│ - Video: Slider + Play/Stop                         │
│ - Tracking Controls                                 │
└─────────────────────────────────────────────────────┘
```

*INSERT SCREENSHOT HERE*

### Color Coding

- 🟢 **Green Background**: Image/Video has annotations
- 🔴 **Red Background**: No annotations yet
- 🔵 **Blue Selection**: Currently selected image/Video


---

## Functions

### 1. Zoom and Pan

#### Image & Video Zoom
- **Zoom In/Out**: Hold `Ctrl` + scroll mouse wheel
- **Zoom Range**: 100% - 300%
- **Zoom Display**: Shows current zoom percentage in upper right corner
- **Reset Zoom**: Zoom out to 100% automatically resets pan position

#### Pan (Move Image/Video)
- **Activate Pan**: Hold `Ctrl` + click and drag on canvas
- **Cursor Change**: Cursor changes to "fleur" (move cursor) during panning
- **Live Update**: Annotations move with the image/video in real-time

**Important**: All coordinates remain relative to the original 100% image size, regardless of zoom level.

---

### 2. Creating Annotations

#### Bounding Box Annotations
1. Select "Bounding Box" from Type dropdown
2. Choose a class from Class dropdown
3. Click and drag on canvas to draw rectangle
4. Release mouse to complete
5. Click "Add" button to save annotation

**Features**:
- Crosshair guide appears while drawing (hides on Ctrl press)
- Red outline for bounding boxes
- Temporary outline while drawing

#### Polygon Annotations
1. Select "Polygon" from Type dropdown
2. Choose a class from Class dropdown
3. Click on canvas to add points
4. Continue clicking to add more points (minimum 3 points)
5. Click "Add" button to save annotation

**Features**:
- Red dots mark polygon vertices
- Blue outline connects points
- Auto-closes polygon from last point to first
- **Undo Last Point**: Press `Ctrl+Z` to remove the last added point

**Note**: Cannot mix bounding boxes and polygons on the same image/frame.

---

### 3. Modifying Annotations

#### Activate Modify Mode
- **Toggle**: Check "Modify Mode" checkbox
- **Keyboard Shortcut**: `Ctrl+E` to toggle modify mode

#### Modifying Bounding Boxes
1. Enable Modify Mode
2. Select annotation from Objects list
3. Blue resize handles appear at corners
4. **Resize**: Click and drag corner handles
5. **Move**: Click inside rectangle and drag
6. Annotation updates automatically when done

#### Modifying Polygons
1. Enable Modify Mode
2. Select polygon from Objects list
3. Red points appear at vertices
4. Click and drag any point to move it
5. Polygon outline updates in real-time

**Deactivate**: After modification uncheck Modify Mode or press `Ctrl+E` again. Annotation is updated automatically in database. 

---

### 4. Deleting Annotations

#### Single Deletion
1. Select annotation from Objects list
2. Click "Delete" button
3. Annotation removed from canvas and data

#### Video: Delete All Frames (Double-Click)
- **Double-click** the "Delete" button in video mode
- Confirmation dialog appears
- Deletes all annotations for the entire video
- Use with caution!

---

### 5. Video Playback

#### Playback Controls
- **Play/Stop Button**: Click `▶` to start playback, `⏸` to pause
  - Button changes icon based on state
  - Plays from current frame to end
  - ~20 fps playback speed
- **Frame Navigation**: 
  - Use slider to jump to specific frame
  - `←` button: Previous frame
  - `→` button: Next frame
- **Frame Counter**: Shows "Frame X / Total" above slider

#### Frame Display
- First frame loads automatically when video selected
- Annotations load for each frame
- Masks can be toggled during playback

---

### 6. Video Tracking

#### Tracking Systems Available
1. **Simple**: Basic frame-to-frame tracking
2. **SAM3**: Latest SAM3 model (Workstation must be logged in to HugginFace and access to the Gated SAM3 model must be granted.)
3. **SAM2 Large**: High-accuracy SAM2 model
4. **SAM2 Tiny**: Faster, lighter SAM2 variant
5. **MedSAM2**: Medical-specialized SAM2
6. **MedSAM2 US Heart**: Ultrasound heart specialist
7. **MedSAM2 MRI Liver Lesion**: MRI liver specialist

#### How to Track Annotations
1. **Annotate First Frame**: Create annotation on starting frame
2. **Select Annotation**: Click on annotation in Objects list
3. **Choose Tracking System**: Select from dropdown
4. **Start Tracking**: Click "Start Tracking" button
5. **Wait**: Progress dialog appears during tracking
6. **Review**: Tracked annotations appear on subsequent frames

**Note**: First-time use may prompt to download model weights.

---

### 7. Mask Functionality

#### Creating Masks from Annotations

**For Images**:
1. Check "Create Mask from Annotation" checkbox
2. Create annotation (bounding box or polygon)
3. Click "Add" - mask generates automatically
4. Mask saved to `../masks/` directory

**For Video Frames**:
1. Check "Create Mask for single frame" checkbox
2. Create annotation on frame
3. Click "Add" - mask generates for that frame
4. Use tracking to propagate masks across frames

#### Viewing Masks
- **Toggle Visibility**: Check/uncheck "Show Masks"
- **Overlay Display**: Masks appear as colored overlays
- **Works With Zoom/Pan**: Masks scale and move with image/video
- **Auto-Load**: Masks load automatically when selecting annotated images from `../masks/`

#### Mask Storage
- Masks stored as PNG files in `../masks/` folder
- Filename format: `{image_id}_{annotation_index}.png`
- RGBA format with transparency
- Linked to annotations via paths in DataFrame

---

### 8. Keyboard Shortcuts

| Shortcut          | Function                  | Context               |
|-------------------|---------------------------|-----------------------|
| `Ctrl+Z`          | Delete last polygon point | While drawing polygon |
| `Ctrl+E`          | Toggle Modify Mode        | Anytime               |
| `Ctrl+Scroll`     | Zoom in/out               | On canvas             |
| `Ctrl+Click+Drag` | Pan image/video           | On canvas             |
| `Ctrl` (hold)     | Hide crosshair            | While held            |

---

### 9. Saving Progress

#### Manual Save
- Click green "Save" button in bottom panel
- Progress spinner shows while saving
- Saves to configured annotation table file (csv)

#### Auto-Save Prompt
- Appears when closing patient window
- Options: Yes (save and close), No (close without saving), Cancel (stay open)
- Also cleans up temporary video frames

---

### 10. Data Export

Annotations are saved in csv format with columns:
- `img_ID`: Image/frame identifier
- `x, y, w, h`: Bounding box parameters (center x, center y, width, height)
- `class`: Annotation class label
- `bb_annotype`: Annotation type (manually/automatically)
- `polygon`: List of polygon vertex coordinates
- `class_polygon`: Polygon class labels
- `polygon_annotype`: Polygon annotation type
- `masks`: Paths to generated mask files

You can export your annotations in COCO-Format via the graphical user interface. 

---

## Recommendations

### Best Practices

#### Annotation Workflow
1. **Start with First Frame**: For videos, annotate the first frame carefully
2. **Use Tracking**: Let segmentation model propagate annotations across frames
3. **Review Tracked Results**: Check and correct tracked annotations as needed
4. **Save Frequently**: Use the Save button regularly to prevent data loss
5. **One Type Per Image**: Stick to either bounding boxes OR polygons per image

#### Performance Optimization
- **Video Frames**: Frames are cached - first load may be slow
- **Zoom Before Drawing**: Zoom in for precise annotations
- **SAM Models**: Require GPU for reasonable performance
- **Mask Generation**: Can be slow on large images - use selectively

#### Quality Guidelines
- **Bounding Boxes**: Should tightly fit the region of interest
- **Polygons**: Use enough points for accurate boundary (minimum 3)
- **Modify Mode**: Use to refine annotations rather than redrawing
- **Tracking Review**: Always manually review tracked annotations for high quality annotations

---

## Frequently Asked Questions

**Q: Which tracking system should I use?**  
A: Check --- SAM2 tiny and SAM3

**Q: Why does tracking ask to download a model?**  
A: SAM2 and MedSAM2 models are large (~800Mb). They download once and are cached for future use.

**Q: Can I track multiple objects in one video?**  
A: Yes, annotate and track each object separately. Select each annotation individually before tracking.

**Q: Tracking results aren't accurate - what can I do?**  
A: 
1. Try a different tracking system (e.g., MedSAM2 for medical content)
2. Manually correct errors and re-track from that frame
3. Ensure first annotation is precise

---

### Technical Questions

**Q: Does TagMed require a GPU?**  
A: Check --- No, but SAM2/MedSAM2 tracking features perform much better with a CUDA-capable GPU. 

**Q: What image formats are supported?**  
A: Check ---PNG, JPG, and JPEG for images. MP4 for videos.

**Q: What if my medical reports don't load?**  
A: Check that `selected_medical_report_file` in `user_settings.json` points to a valid csv file with matching exam IDs.

---

## Troubleshooting

### Common Issues

== SEARCH FOR MORE BUGS == 

**Problem**: Annotations don't appear after adding  
**Solution**: Check that you clicked "Add" after drawing. Verify annotation appears in Objects list.

**Problem**: Crosshair doesn't disappear when pressing Ctrl  
**Solution**: This is expected - crosshair only hides while Ctrl is held down.

**Problem**: Video won't play  
**Solution**: Ensure video frames are extracted (happens automatically on first load).

**Problem**: Modify Mode doesn't show handles  
**Solution**: Make sure you selected an annotation from the Objects list first.

**Problem**: Zoom doesn't work  
**Solution**: Hold Ctrl while scrolling. Check that cursor is over the canvas area.

**Problem**: Tracking fails or crashes  
**Solution**: 
1. Check GPU memory (SAM2 models require ~4GB VRAM)
2. Try SAM2 Tiny for lower memory usage
3. Verify model files downloaded correctly

---


*Last Updated: January 28, 2026*  
*Version: 1.0*
