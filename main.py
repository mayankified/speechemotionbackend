from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import numpy as np
import librosa
from vmdpy import VMD
import io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Initialize FastAPI app
app = FastAPI()

# Helper function to compute energy over frames
def energy(signal, frame_length, hop_length):
    energy_vals = []
    for i in range(0, len(signal), hop_length):
        frame = signal[i : i + frame_length]
        frame_energy = np.sum(frame**2)
        energy_vals.append(frame_energy)
    return np.mean(np.array(energy_vals))

import io
from pydub import AudioSegment

def convert_to_wav(audio_bytes):
    # Convert the file to WAV using pydub
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

# Feature extraction function
def extract_features(
    wav_file,
    sr=16000,
    frame_length=800,
    hop_length=400,
    alpha=5000,
    tau=0,
    K=3,
    DC=0,
    init=1,
    tol=1e-7,
):
    """
    Extract features from a WAV file-like object.
    """
    # Load audio from file-like object
    signal, _ = librosa.load(wav_file, sr=sr)

    # Pre-emphasis and normalization
    signal = librosa.effects.preemphasis(signal, coef=0.97)
    signal = librosa.util.normalize(signal)

    # Compute total energy of the signal
    total_energy_val = energy(signal, frame_length, hop_length)

    # Define Hanning window
    window = np.hanning(frame_length)
    features_per_frame = []

    # Process the signal in frames
    for start in range(0, len(signal) - frame_length + 1, hop_length):
        current_frame = signal[start : start + frame_length] * window
        # Apply Variational Mode Decomposition (VMD)
        u, u_hat, omega = VMD(current_frame, alpha, tau, K, DC, init, tol)
        frame_features = []
        # For each mode, compute MFCCs
        for mode in u:
            mfccs = librosa.feature.mfcc(
                y=mode, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mfcc=30
            )
            mfccs_mean = np.mean(mfccs, axis=1)
            frame_features.extend(mfccs_mean.tolist())
        features_per_frame.append(frame_features)

    if features_per_frame:
        feature_avg = np.mean(np.array(features_per_frame), axis=0)
    else:
        feature_avg = np.array([])

    return np.expand_dims(feature_avg, axis=0)


# Label mapping and reverse mapping for emotions
label_mapping = {
    "happyness": 0,
    "neutral": 1,
    "anger": 2,
    "sadness": 3,
    "fear": 4,
    "boredom": 5,
    "disgust": 6,
}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Define the ANN model architecture
ANN_model = Sequential(
    [
        Dense(999, input_shape=(90,), activation="elu"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(785, activation="elu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(865, activation="elu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(672, activation="elu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(7, activation="softmax"),  # 7 emotion classes
    ]
)

# Compile the model
optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
ANN_model.compile(
    optimizer=optimiser,
    loss="sparse_categorical_crossentropy",
    metrics=["SparseCategoricalAccuracy"],
)

# Load the pretrained weights
checkpoint_path = "Ann_EMODB_90feature.weights.h5"
try:
    ANN_model.load_weights(checkpoint_path)
    print("Model weights loaded successfully!")
except Exception as e:
    print("Error loading model weights:", e)


# Define the predict endpoint
@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if file.content_type not in ["audio/wav", "audio/x-wav"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a WAV file.")
    try:
        contents = await file.read()
        # Convert the uploaded file to WAV
        wav_file = convert_to_wav(contents)
        # Optional: Ensure pointer is at the start
        wav_file.seek(0)
        
        # Extract features from the WAV file
        data = extract_features(wav_file)
        
        if data.shape[1] != 90:
            raise HTTPException(status_code=500, detail="Feature extraction failed.")
        
        predictions = ANN_model.predict(data)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_emotion = reverse_label_mapping.get(predicted_index, "Unknown")
        return {"predicted_emotion": predicted_emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
