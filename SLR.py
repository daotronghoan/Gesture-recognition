#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# In[11]:


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
thresshold = 0.98
lenvec = 1530


# In[12]:


def extracting_key_point(results) :
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()                                if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()                             if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()                            if results.right_hand_landmarks else np.zeros(21*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()                if results.pose_landmarks else np.zeros(21*3)
    return np.concatenate([face,lh, rh])


# In[13]:


def extracting_key_point_from_image(image) :
    image, results = mediapipe_detection(image, model)
    return extracting_key_point(results)  


# In[14]:


def draw_land_marks(image, results) :
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# In[15]:


def mediapipe_detection(image, model) :
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# In[16]:


def draw_styled_landmarks(image, results) :
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS
                             ,mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121),thickness=1, circle_radius=1)
                             )
    #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# In[17]:


actions = np.array(['hello', 'how are you', 'i am fine', 'thank you', 'i love you'])
no_sequence = 30
sequence_length = 30


# In[18]:


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,lenvec)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.load_weights('model.h5')


# In[19]:


cap = cv2.VideoCapture(0)
feature_frame = []
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence = 0.5) as hol :
    while cap.isOpened():
        ret, frame = cap.read()
        frame = np.flip(frame, axis = 1)

        image, results = mediapipe_detection(frame, hol)
        
        draw_styled_landmarks(image, results)
        
        feature = extracting_key_point(results)
        feature_frame.append(feature)
        if len(feature_frame) == 31 :
            feature_frame.pop(0)
            feature = np.array(feature_frame)
            #feature = np.array([feature for i in range(30)])
            feature = feature.reshape((1, 30, lenvec))


            pre = model.predict(feature)
            pre.shape = (actions.shape[0], 1)
            pre_label = actions[np.argmax(pre)]
            percent = pre[np.argmax(pre)]
            if percent > thresshold:
                cv2.putText(image,'{}'.format(percent),(15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 4, cv2.LINE_AA)
                cv2.putText(image,pre_label,(120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q') :
            break
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:




