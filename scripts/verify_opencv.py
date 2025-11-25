#!/usr/bin/env python3
"""
Simple environment verification for OpenCV / cv2.face availability.
Run: python scripts/verify_opencv.py

This prints helpful messages and a suggested fix if cv2.face is missing.
"""
import sys
import textwrap

print("Python:", sys.version.splitlines()[0])

try:
    import cv2
except Exception as e:
    print("\nERROR: Failed to import cv2: {}\n".format(e))
    print(textwrap.dedent("""
        Common fixes:
        - Make sure you are using the same Python environment you expect.
        - If you see import errors from conflicting OpenCV builds, run:

            pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
            pip install opencv-contrib-python

        Or for headless/server environments:

            pip install opencv-contrib-python-headless

        After installing, re-run this script.
    """))
    sys.exit(2)

print("cv2 version:", getattr(cv2, "__version__", "unknown"))

has_face = hasattr(cv2, "face")
print("cv2.face available:", has_face)

if has_face:
    face_mod = cv2.face
    has_lbph = hasattr(face_mod, "LBPHFaceRecognizer_create") or hasattr(face_mod, "createLBPHFaceRecognizer")
    print("Contains LBPH face recognizer factory:", has_lbph)
    if has_lbph:
        print("Great — your environment provides the cv2.face LBPH recognizer API used by this project.")
    else:
        print("cv2.face exists but the LBPH recognizer factory name wasn't found.")
else:
    print(textwrap.dedent("""
        cv2.face is NOT available in this Python environment.

        Likely cause: you have a non-contrib OpenCV package installed (e.g., opencv-python or headless opencv-python-headless)

        Fix suggestion (PowerShell / Windows example):

            pip uninstall -y opencv-python opencv-python-headless
            pip install opencv-contrib-python

        For headless/server environments:

            pip uninstall -y opencv-python opencv-python-headless
            pip install opencv-contrib-python-headless

        After changing packages, re-run this script to confirm.
    """))

# Exit codes: 0 = success & cv2.face present, 1 = cv2.imported but cv2.face missing
sys.exit(0 if has_face else 1)
