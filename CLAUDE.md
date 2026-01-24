# CLAUDE.md - AI Assistant Guide

**Repository**: human-computer-interface
**Last Updated**: 2026-01-24
**License**: MIT
**Language**: C++
**Platform**: Windows

## Project Overview

This is a **hand gesture recognition system** that provides touchless keyboard and mouse control using computer vision. The system uses a webcam to interpret American one-handed sign language alphabet and number gestures (A-Z, 0-9) and translates them into keyboard/mouse commands.

### Core Purpose
- Provide natural human-computer interface through hand gestures
- Enable touchless control for accessibility (e.g., paralysis patients)
- Recognize 36 gestures (26 alphabets + 10 numbers)
- Control Windows applications (Media Player, MS Word, Paint, Games)

## Repository Structure

```
/
├── core-engine.cpp              # Main application source code (~1182 lines)
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
├── CONTRIBUTING.md              # Empty (placeholder)
├── Algorithm-used               # Algorithm documentation
├── Project-features             # Feature list
├── pre-requisites              # Hardware/software requirements
├── keyboard-mouse-interface    # Interface implementation notes
└── Hand Gesture Recognition using computer vision.ppt
```

## Technology Stack

### Development Environment
- **IDE**: Microsoft Visual Studio 6.0 (legacy)
- **OS**: Windows 98/2000/XP (legacy, but adaptable)
- **Compiler**: MSVC with C++ support
- **Libraries**: OpenCV (legacy version with cv.h, highgui.h)

### Hardware Requirements
- Pentium 3 processor or above
- 128 MB RAM minimum
- Webcam
- Monitor
- Hand gloves (for better recognition)

### Key Dependencies
```cpp
#include "cv.h"           // OpenCV core
#include "highgui.h"      // OpenCV GUI
#include <windows.h>      // Windows API
#include "math.h"         // Mathematical operations
```

## Core Algorithms & Techniques

### Image Processing Pipeline
1. **RGB to Grayscale Conversion** (core-engine.cpp:859-871)
   - Averages RGB values into single-channel image

2. **Thresholding** (core-engine.cpp:873-881)
   - Background subtraction using binary threshold
   - Converts to black/white image (0 or 255)

3. **Edge Detection** (core-engine.cpp:883-892)
   - Uses **Canny Detection Algorithm**
   - Detects intensity changes between pixels
   - Threshold: 100

4. **Morphological Operations**
   - Dilation (core-engine.cpp:894-901): Connects pixels
   - Erosion (core-engine.cpp:903-910): Noise removal

5. **Contour Detection** (core-engine.cpp:915-1135)
   - Finds hand shape boundaries
   - Extracts X,Y coordinates

6. **Shape Signature - Centroid Distance Method**
   - Calculates centroid of hand shape
   - Computes distances from edge points to centroid
   - Makes recognition **translation-invariant**

7. **Fourier Transformation** (core-engine.cpp:1000-1055)
   - Discrete Fourier Transform on shape signature
   - Makes recognition **rotation-invariant**
   - Stores 20 first + 20 last Fourier descriptors

8. **Pattern Matching** (core-engine.cpp:1057-1135)
   - Compares live Fourier descriptors with stored templates
   - Uses minimum distance matching
   - Template files: a.txt, b.txt, c.txt, d.txt, h.txt, e.txt, l.txt, o.txt, end.txt

### Mouse Control Features
- **Movement**: Hand position tracking (core-engine.cpp:759-782)
  - Uses CAMShift algorithm for tracking
  - Converts webcam coordinates to screen coordinates (1.8x scaling)

- **Single Click**: Hand rotation 10-55 degrees (core-engine.cpp:346-399)

- **Double Click**: Hand rotation -50 to -1 degrees (core-engine.cpp:323-344)

### Application Integration
1. **Notepad** (core-engine.cpp:364-398)
   - Detects window "Untitled - Notepad"
   - Switches to gesture recognition mode
   - Types recognized characters

2. **Paint** (core-engine.cpp:553-710)
   - Detects window "untitled - Paint"
   - Hand tracking for drawing
   - G key exits mode

3. **GTA Vice City** (core-engine.cpp:402-551)
   - Game control through hand gestures
   - Maps gestures to game controls
   - Middle mouse button for acceleration

## Code Architecture

### Main Components

#### 1. Main Loop (core-engine.cpp:182-757)
```cpp
int main(int argc, char** argv)
```
- Initializes webcam capture
- Creates windows and trackbars
- Runs CAMShift tracking loop
- Handles application-specific controls

#### 2. Gesture Recognition (core-engine.cpp:791-819)
```cpp
void gesture()
```
- Entry point for keyboard gesture recognition
- Coordinates image processing pipeline
- Loops until exit gesture detected

#### 3. Mouse Movement (core-engine.cpp:759-782)
```cpp
void MoveMouse(int x, int y)
```
- Uses Windows SendInput API
- Sets cursor position

#### 4. Character Input (core-engine.cpp:1138-1182)
```cpp
void sendchar(int ke)
```
- Sends keyboard input to active window
- Uses Windows SendInput API

### Global Variables
```cpp
// Tracking state
int track_object = 0;
int select_object = 0;

// Image buffers
IplImage *image, *hsv, *hue, *mask, *backproject, *histimg;
CvHistogram *hist;

// Mouse state
int mb1=0, mb2=0, moveflag=0, paintflg=0;

// Coordinates
int x=0, y=0, x2=0, y2=0;

// Neural network (declared but appears unused)
double weightsIH[7][7];
double weightsHO[7][2];
```

## Development Guidelines for AI Assistants

### When Working with This Code

#### DO:
- Understand this is **legacy code** (Visual Studio 6.0, old OpenCV)
- Test thoroughly on Windows platform
- Preserve the core algorithm flow
- Maintain compatibility with OpenCV legacy API
- Document any modernization efforts
- Consider accessibility use cases

#### DON'T:
- Break Windows API dependencies without replacement
- Modify Fourier descriptor algorithm without testing
- Change coordinate scaling factors arbitrarily
- Remove application-specific detection logic
- Introduce breaking changes to gesture recognition pipeline

### Modernization Considerations

If updating this codebase:

1. **OpenCV Migration**
   - Legacy headers: `cv.h`, `highgui.h`
   - Modern headers: `opencv2/opencv.hpp`
   - Many legacy functions deprecated

2. **Platform Independence**
   - Current: Windows-only (SendInput, mouse_event, FindWindow)
   - Consider: Cross-platform libraries (Qt, SDL)

3. **Code Quality**
   - Add error handling
   - Remove magic numbers (extract as constants)
   - Separate concerns (UI, processing, recognition)
   - Add unit tests

4. **Neural Network**
   - Declared but not fully implemented (lines 94-123)
   - Could enhance recognition accuracy

### Common Tasks

#### Adding New Gesture Recognition
1. Create gesture shape with hand
2. Capture and process image
3. Extract Fourier descriptors
4. Save to new .txt file (e.g., "x.txt")
5. Add to `arr1` array (core-engine.cpp:926)
6. Add recognition case (core-engine.cpp:1116-1133)

#### Debugging Recognition Issues
- Check threshold value (edge_thresh, line 91)
- Verify lighting conditions
- Examine intermediate images (gray.bmp, threshold.bmp, edge_detection.bmp)
- Compare Fourier descriptors in avi.txt

#### Adding New Application Support
1. Detect window handle with `FindWindow()`
2. Add conditional block in main loop (after line 350)
3. Implement application-specific gesture mapping
4. Handle window focus and input

### Testing Checklist

When making changes:
- [ ] Webcam initializes correctly
- [ ] Hand tracking works in good lighting
- [ ] Mouse movement is smooth and accurate
- [ ] Single/double click gestures detected
- [ ] Character recognition accuracy > 80%
- [ ] No memory leaks (check image releases)
- [ ] Works with target applications (Notepad, Paint)
- [ ] Exit gestures function properly

## Known Limitations

1. **Windows-only**: Uses Windows API extensively
2. **Legacy OpenCV**: Old API (pre-2.x)
3. **No error handling**: Limited validation and error checks
4. **Hard-coded values**: Many magic numbers throughout
5. **Limited gesture set**: Only 9 template files provided
6. **Lighting sensitive**: Performance varies with environment
7. **Single-threaded**: UI and processing in same thread
8. **Memory management**: Some potential leaks

## File I/O

### Input Files (Template Data)
- a.txt, b.txt, c.txt, d.txt, e.txt, h.txt, l.txt, o.txt, end.txt
- Format: Fourier descriptors (one per line)
- Location: Working directory

### Output Files (Debug/Processing)
- color.bmp - Captured webcam frame
- gray.bmp - Grayscale conversion
- threshold.bmp - Binary threshold
- edge_detection.bmp - Canny edge detection
- dilate.bmp - After dilation
- erosion.bmp - After erosion
- avi.txt - Live Fourier descriptors

## Hot Keys

- **ESC**: Quit program
- **c**: Stop tracking
- **b**: Toggle backprojection view
- **h**: Show/hide histogram
- **g**: Exit Paint mode
- **Numpad 1**: Toggle mouse movement flag

## Contributing

The CONTRIBUTING.md file is currently empty. Before contributing:

1. Understand the legacy codebase constraints
2. Test on Windows platform
3. Maintain backward compatibility unless explicitly modernizing
4. Document all changes
5. Consider accessibility impact
6. Provide before/after comparison for algorithm changes

## Security Considerations

This code uses Windows API to:
- Control mouse movement
- Send keyboard input
- Detect and focus windows

**Warning**: This could be misused for:
- Unauthorized input injection
- Keystroke logging
- UI manipulation

Only use this software:
- On systems you own
- With proper authorization
- For legitimate accessibility or research purposes

## Resources

### Understanding the Algorithms
- Canny Edge Detection: https://docs.opencv.org/
- Fourier Descriptors for Shape Recognition
- CAMShift Algorithm: Continuously Adaptive Mean Shift
- Centroid Distance Method for shape signatures

### OpenCV Documentation
- Legacy API (1.x): https://docs.opencv.org/2.4/
- Modern API (4.x): https://docs.opencv.org/4.x/

### Windows API
- SendInput: Keyboard/mouse input synthesis
- FindWindow: Window detection
- SetForegroundWindow: Window activation

## Questions to Ask Users

When assisting with this codebase:

1. **Which Windows version are you targeting?**
2. **Do you want to modernize or maintain legacy compatibility?**
3. **What gesture recognition accuracy are you achieving?**
4. **Which applications need gesture control?**
5. **Is this for accessibility, research, or general use?**
6. **Do you have the template .txt files?**
7. **What webcam resolution are you using?**

## Quick Reference: Key Functions

| Function | Line | Purpose |
|----------|------|---------|
| main() | 182 | Entry point, webcam tracking loop |
| gesture() | 791 | Keyboard gesture recognition mode |
| MoveMouse() | 759 | Mouse cursor control |
| sendchar() | 1138 | Send keyboard input |
| capturef() | 820 | Capture webcam frame |
| color_to_gray() | 859 | RGB to grayscale conversion |
| gray_to_thresh() | 873 | Thresholding |
| thresh_to_edge() | 883 | Canny edge detection |
| edge_to_dilate() | 894 | Morphological dilation |
| dilate_to_erosion() | 903 | Morphological erosion |
| contour() | 915 | Contour detection & recognition |
| on_mouse() | 126 | Mouse callback for ROI selection |
| hsv2rgb() | 165 | Color space conversion |

---

**End of CLAUDE.md**

This document should be updated when significant changes are made to the codebase or development practices.
