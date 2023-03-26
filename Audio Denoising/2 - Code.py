# Import required libraries
import numpy as np
import librosa
import noisereduce as nr

# Load audio file
audio_file = 'example_audio.wav'
data, sample_rate = librosa.load(audio_file)

# Plot audio file
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
librosa.display.waveplot(data, sr=sample_rate)
plt.title('Original Audio Waveform')
plt.show()

# Add noise to audio
noise_file = 'example_noise.wav'
noise_data, noise_sample_rate = librosa.load(noise_file)
noise_data = noise_data[:len(data)]
noisy_data = data + noise_data

# Plot noisy audio
plt.figure(figsize=(14, 5))
librosa.display.waveplot(noisy_data, sr=sample_rate)
plt.title('Noisy Audio Waveform')
plt.show()

# Apply noise reduction
reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noise_data, verbose=False)

# Plot denoised audio
plt.figure(figsize=(14, 5))
librosa.display.waveplot(reduced_noise, sr=sample_rate)
plt.title('Denoised Audio Waveform')
plt.show()

# Save denoised audio to file
librosa.output.write_wav('denoised_audio.wav', reduced_noise, sample_rate)
