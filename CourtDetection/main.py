import numpy as np

# Standard Badminton Court Dimensions in Meters
# Width: 6.10m (Doubles), Length: 13.40m
# Origin (0,0) is at the top-left outer corner.
# source: https://www.cits.wa.gov.au/sport-and-recreation/sports-dimensions-guide/badminton
COURT_TEMPLATE = {
    # --- ROW 1: Top Outer Baseline (Y = 0.00) ---
    0:  [0.00,  0.00],   # Top-Left Outer Corner
    1:  [0.46,  0.00],   # Top Singles-Left Intersection
    2:  [5.64,  0.00],   # Top Singles-Right Intersection
    3:  [6.10,  0.00],   # Top-Right Outer Corner

    # --- ROW 2: Top Long / Back Service Line (Y = 0.76) ---
    4:  [0.00,  0.76],   # Top Back Service Left
    5:  [6.10,  0.76],   # Top Back Service Right

    # --- ROW 3: Top Front / Short Service Line (Y = 4.72) ---
    6:  [0.00,  4.72],   # Top Front Service Left
    7:  [3.05,  4.72],   # Top Front Center T-Junction
    8:  [6.10,  4.72],   # Top Front Service Right

    # --- ROW 4: Bottom Front / Short Service Line (Y = 8.68) ---
    9:  [0.00,  8.68],   # Bottom Front Service Left
    10: [3.05,  8.68],   # Bottom Front Center T-Junction
    11: [6.10,  8.68],   # Bottom Front Service Right

    # --- ROW 5: Bottom Long / Back Service Line (Y = 12.64) ---
    12: [0.00, 12.64],   # Bottom Back Service Left
    13: [6.10, 12.64],   # Bottom Back Service Right

    # --- ROW 6: Bottom Outer Baseline (Y = 13.40) ---
    14: [0.00, 13.40],   # Bottom-Left Outer Corner
    15: [0.46, 13.40],   # Bottom Singles-Left Intersection
    16: [5.64, 13.40],   # Bottom Singles-Right Intersection
    17: [6.10, 13.40],   # Bottom-Right Outer Corner
}