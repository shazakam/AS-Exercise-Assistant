from DataProcessor import DataProcessor as dp
from DataProcessorMulti import MultiDP as mdp
from VideoReciever import VideoMessenger as vm
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from firebase_admin import credentials
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time
from sklearn.metrics import accuracy_score # Accuracy metrics 
#import pickle 
import math
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import subprocess as sp
import multiprocessing as mup
from os import remove


if __name__ == '__main__':
    
    cred = credentials.Certificate("key.json")
    firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    
    while(True):
        
        result = db.collection("exerciseType").document("Exercise").get()
        
        if result.exists:
            dataP = dp("myVideo.mp4", db)
            vM = vm()
            vM.recieveVideo()
            
            if result.to_dict()["exercise"] == "lower":
                dataP.analyse_video_lower()
                
            else:
                dataP.analyse_video_lie()
            
            vM.sendVideo()
            db.collection("exerciseType").document("Exercise").delete()
            print(result.to_dict()["exercise"])
        else:
            continue

        