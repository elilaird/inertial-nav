"""
KITTI odometry benchmark sequence mapping.

Maps between odometry sequence numbers (0-10) and raw drive names.
Extracted from KITTIDataset.odometry_benchmark.
"""

from collections import OrderedDict

# Odometry sequence number → raw drive name (ordered 00-10)
ODOM_TO_DRIVE = OrderedDict(
    [
        (0, "2011_10_03_drive_0027_extract"),
        (1, "2011_10_03_drive_0042_extract"),
        (2, "2011_10_03_drive_0034_extract"),
        (3, "2011_09_26_drive_0067_extract"),
        (4, "2011_09_30_drive_0016_extract"),
        (5, "2011_09_30_drive_0018_extract"),
        (6, "2011_09_30_drive_0020_extract"),
        (7, "2011_09_30_drive_0027_extract"),
        (8, "2011_09_30_drive_0028_extract"),
        (9, "2011_09_30_drive_0033_extract"),
        (10, "2011_09_30_drive_0034_extract"),
    ]
)

# Reverse mapping: raw drive name → odometry sequence number
DRIVE_TO_ODOM = {v: k for k, v in ODOM_TO_DRIVE.items()}

# Odometry benchmark frame ranges (from KITTIDataset.odometry_benchmark)
ODOM_FRAME_RANGES = {
    0: [0, 45692],
    1: [0, 12180],
    2: [0, 47935],
    3: [0, 8000],
    4: [0, 2950],
    5: [0, 28659],
    6: [0, 11347],
    7: [0, 11545],
    8: [11231, 53650],
    9: [0, 16589],
    10: [0, 12744],
}
