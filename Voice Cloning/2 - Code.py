import tensorflow as tf
import numpy as np
import librosa
import argparse

from tacotron.hparams import hparams
from tacotron.models.tacotron import Tacotron
from tacotron.utils.text import text_to_sequence
from tacotron.utils.audio import inv_mel_spectrogram, save_wav

def generate_voice(model, text):
    # Prepare text input
    sequence = np.array(text_to_sequence(text, hparams.text_cleaners))
    sequence = np.reshape(sequence, [1, -1])
    feed_dict = {
        "inputs:0": np.zeros([1, 1, 80]),
        "input_lengths:0": np.array([len(sequence)]),
        "text:0": sequence
    }

    # Generate mel spectrogram
    mel_output = sess.run(model.mel_outputs, feed_dict=feed_dict)

    # Convert mel spectrogram to audio
    audio_output = inv_mel_spectrogram(mel_output[0], hparams)
    audio_output = audio_output[:-(audio_output.shape[0] % hparams.hop_length)]
    audio_output = audio_output / np.abs(audio_output).max() * hparams.rescaling_max

    return audio_output

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate voice from input text')
    parser.add_argument('--checkpoint', type=str, default='logs/latest_model.ckpt',
                        help='Path to checkpoint file')
    parser.add_argument('--text', type=str, default='Hello, how are you?',
                        help='Text to generate voice for')
    parser.add_argument('--output', type=str, default='output.wav',
                        help='Output file name')
    args = parser.parse_args()

    # Load model
    model = Tacotron()
    model.load(args.checkpoint)

    # Generate voice
    audio_output = generate_voice(model, args.text)

    # Save as WAV file
    save_wav(audio_output, args.output, hparams)
