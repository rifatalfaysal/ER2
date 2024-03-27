# Importing Necessary Libraries
import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, VideoTransformerBase
from PIL import Image



css_example = '''                                           
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">    
    
    <style>
        .bodyP1{
            font-size: 20px;
        }
        .footer{
            display: flex;
            justify-content:center;
            align-items: center;
            font-size: 20px;
            font-weight: 300;
            margin-top: 50px;
        }
        .aboutUs p{
            font-size: 18px;
            text-align: justify;
        }
        .header{
            display: flex;
            flex-direction:column;
            justify-content: center;
            align-items: center;
        }
    </style>
'''
st.write(css_example, unsafe_allow_html=True)

# Declaring Classes
emotion_classes = {
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Happy", 
    4: "Neutral", 
    5: "Sad", 
    6: "Surprise"}

# Loading Trained Model:
json_file = open(r"model/model_v2.json", 'r')

# Loading model.json file into model
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Loading Weights:
model.load_weights(r"model/new_model_v2.h5")

print("Model lodded scussesfully")

# Loading Face Cascade
try: 
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.error("Unable to load Cascade Classifier", icon="⚠️")


class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        # Converting frame into 2 array of RGB format.
        img = np.array(frame.to_ndarray(format = "bgr24"))

        #Converting the Captured frame to gray scale:
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces available on camera:
        num_face = face_detector.detectMultiScale(gray_frame, scaleFactor = 1.3, minNeighbors = 5)

        # Take each fave available on the camera and preprocess it:
        for (x, y, w, h) in num_face:
            cv2.rectangle(img, (x,y-50), (x+w, y+h+10), (0,255,0), 4)
            roi_gray_frame = gray_frame[y:y+h, x: x+w]
            cropped_img = np.expand_dims(cv2.resize(roi_gray_frame, (48,48), -1), 0)

            #Predict the emotion:
            if np.sum([roi_gray_frame])!=0:
                emotion_prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                label_position = (x,y)
                output = str(emotion_classes[maxindex])
                cv2.putText(img,output,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(img,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        return img


def main():
    

    
        header = """
                    <div class = "header">
                        <h1>Smile</h1>
                    </div>
                """
        st.markdown(header,unsafe_allow_html=True)
       
        webrtc_streamer(key="example", video_transformer_factory=EmotionDetector)
        
    
    


if __name__ == "__main__":
    main()


