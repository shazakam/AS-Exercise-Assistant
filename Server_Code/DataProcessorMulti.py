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

class MultiDP:
    
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils # Drawing helpers
        self.mp_holistic = mp.solutions.holistic 
        #with open('model.pkl', 'rb') as f:
        #    self.model = pickle.load(f)
        pass
    
    def draw_landmarks(self, results,image):
        # 2. Right hand
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                                self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                )

        # 3. Left Hand
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                                self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                )

        # 4. Pose Detections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, 
                                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )
        return image
            
    
    def send_lower_data(self,angles, rep_times, db):
        average_angle = sum(angles)/len(angles)
        max_angle = max(angles)
        
        
        average_rep = sum(rep_times)/len(rep_times)
        longest_rep = max(rep_times)
    
    
        print(average_angle)
        print(max_angle)
        print(average_rep)
        print(longest_rep)
        
        data = {
            u'average_angle': average_angle,
            u'max_angle': max_angle,
            u'average_rep': average_rep,
            u'longest_rep':longest_rep
        }
        
        db.collection(u'RESULTS').document(u'lower_results').set(data)

    
    def get_angle(self, firstPoint, midPoint, lastPoint):
        result = math.degrees(math.atan2(lastPoint.y - midPoint.y,lastPoint.x - midPoint.x)- math.atan2(firstPoint.y - midPoint.y, firstPoint.x - midPoint.x))
        result = abs(result)
        if (result > 180):
            result = (360.0 - result)
        return result       
            
    #MULTITHREADING
            
            
    def multi_analyse_lower(self, image):
        timer = 0
        in_rep = False
        rep_times = []
        angles = []
        
        try:
            # Recolor Feed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False       
            
            with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                # Make Detections
                results = holistic.process(image)
                # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            image = self.draw_landmarks(image=image,results=results)
            
            return image
            
            #self.send_lower_data(angles=angles, rep_times=rep_times, db=db)

        except:
            pass
        
        

"""
                try:
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    row = pose_row
                    
                    X = pd.DataFrame([row])
                    body_language_class = self.model.predict(X)[0]
                    body_language_prob = self.model.predict_proba(X)[0]
                    
                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))
                    
                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    
                    height, width, _ = image.shape
                    # Get status box
                    cv2.rectangle(image, (0,0), (width, 95), (245, 117, 16), -1)
                    
                    
                    # Display Probability
                    cv2.putText(image, 'Quality Score'
                                , (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.putText(image, 'Class'
                                    , (300,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    #Checks if most likely pose is lower stretch
                    if body_language_class != "Lower_Stretch":
                        cv2.putText(image, str(round(body_language_prob[1 - np.argmax(body_language_prob)],2))
                                    , (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        cv2.putText(image, "Bad Form"
                                    , (300,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                    , (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, body_language_class.split('_')[0]
                                    , (300,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                        right_shoulder_prob = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER].visibility
                        left_shoulder_prob = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_SHOULDER].visibility
                        angle = 0
                        
                        if right_shoulder_prob > left_shoulder_prob:
                            left_knee = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_KNEE]
                            left_ankle = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_ANKLE]
                            left_hip = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_HIP]
                            
                            angle = self.get_angle(left_ankle,left_knee,left_hip)
                            
                            if angle > 160:
                                cv2.putText(image, "PUSH LEFT LEG FORWARD"
                                    ,(width - 450,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                
                                if in_rep:
                                    rep_times.append(time.time() - timer)
                                    in_rep = False
                            else:
                                if in_rep:
                                    cv2.putText(image, "GOOD JOB"
                                        ,(15,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                    angles.append(angle)
                                else:
                                    in_rep = True
                                    timer = time.time()
                        else:
                            right_knee = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_KNEE]
                            right_ankle = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_ANKLE]
                            right_hip = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_HIP]
                            
                            angle = self.get_angle(right_ankle,right_knee,right_hip)
                            
                            if angle > 160:
                                cv2.putText(image, "PUSH RIGHT FORWARD"
                                    ,(15,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                
                                if in_rep:
                                    rep_times.append(time.time() - timer)
                                    in_rep = False
                                    
                            else:
                                if in_rep:
                                    cv2.putText(image, "GOOD JOB"
                                        ,(15,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                    angles.append(angle)
                                else:
                                    in_rep = True
                                    timer = time.time()
                  
                                
                    video_result.write(image)
                    proc_frames += 1
                                   
                except:
                    pass
                
                 """ 
    

            
            