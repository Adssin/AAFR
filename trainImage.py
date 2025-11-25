import csv
import os, cv2
import numpy as np
import pandas as pd
import datetime
import time
from PIL import ImageTk, Image


# Train Image
def TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path, message,text_to_speech):
    # Ensure the cv2.face module is available (requires opencv-contrib-python)
    try:
        if hasattr(cv2, "face"):
            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
            except AttributeError:
                # older/alternate API name
                recognizer = getattr(cv2.face, "createLBPHFaceRecognizer", None)
                if recognizer is None:
                    raise
                recognizer = recognizer()
        else:
            raise AttributeError
    except AttributeError:
        msg = (
            "cv2.face is not available. Install 'opencv-contrib-python' (and remove conflicting opencv packages).\n"
            "For example:\n    pip uninstall opencv-python opencv-python-headless\n    pip install opencv-contrib-python"
        )
        # show a friendly message via UI hooks
        message.configure(text=msg)
        text_to_speech(msg)
        raise RuntimeError(msg)
    detector = cv2.CascadeClassifier(haarcasecade_path)
    faces, Id = getImagesAndLables(trainimage_path)
    recognizer.train(faces, np.array(Id))
    recognizer.save(trainimagelabel_path)
    res = "Image Trained successfully"  # +",".join(str(f) for f in Id)
    message.configure(text=res)
    text_to_speech(res)


def getImagesAndLables(path):
    # imagePath = [os.path.join(path, f) for d in os.listdir(path) for f in d]
    newdir = [os.path.join(path, d) for d in os.listdir(path)]
    imagePath = [
        os.path.join(newdir[i], f)
        for i in range(len(newdir))
        for f in os.listdir(newdir[i])
    ]
    faces = []
    Ids = []
    for imagePath in imagePath:
        pilImage = Image.open(imagePath).convert("L")
        imageNp = np.array(pilImage, "uint8")
        Id = int(os.path.split(imagePath)[-1].split("_")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids
