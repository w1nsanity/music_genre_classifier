from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import librosa
import math

app = FastAPI()

origins = [
    'http://localhost',
    'http://localhost:3000',
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

MODEL = tf.keras.models.load_model('../models/Music_Genre_10_CNN')

CLASS_NAMES = ["blues","classical","country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

@app.get('/ping')
async def ping():
    return 'hello im here'

# Audio files pre-processing
def process_input(audio_file, track_duration):

  SAMPLE_RATE = 22050
  NUM_MFCC = 13
  N_FTT=2048
  HOP_LENGTH=512
  TRACK_DURATION = track_duration # measured in seconds
  SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
  NUM_SEGMENTS = 10

  samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
  num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

  signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
  
  for d in range(10):

    # calculate start and finish sample for current segment
    start = samples_per_segment * d
    finish = start + samples_per_segment

    # extract mfcc
    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
    mfcc = mfcc.T

    return mfcc

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    audio = process_input(file.file, 30)
    
    x_to_predict = audio[np.newaxis, ..., np.newaxis]
    
    prediction = MODEL.predict(x_to_predict)
    
    pred_class = CLASS_NAMES[int(np.argmax(prediction, axis=1))]
    confidence = np.max(prediction[0])
    
    return {
        'pred_class': pred_class,
        'confidence': float(confidence)
    }

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port='8000')