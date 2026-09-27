import torch
import numpy as np

class AddNoise:
    """Add random noise to audio data."""
    def __init__(self, noise_level=0.02):
        self.noise_level = noise_level

    def __call__(self, audio):
        noise = np.random.normal(0, self.noise_level, audio.shape)
        return audio + noise

class VolumeAdjust:
    """Randomly adjust volume within a given range."""
    def __init__(self, min_gain=0.1, max_gain=1.3):
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, audio):
        gain = np.random.uniform(self.min_gain, self.max_gain)
        return audio * gain


class FlipPolarity:
    """Invert the polarity of the audio."""
    def __call__(self, audio):
        return -audio


class DCOffset:
    """Add a random DC bias to the audio."""
    def __init__(self, max_offset=0.05):
        self.max_offset = max_offset

    def __call__(self, audio):
        dc_bias = np.random.uniform(-self.max_offset, self.max_offset)
        return audio + dc_bias


class Normalize:
    """Normalize the audio signal to [-1, 1]."""
    def __call__(self, audio):
        return audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) != 0 else audio


class ClippingDistortion:
    """Apply amplitude clipping to simulate audio distortion."""
    def __init__(self, clipping_threshold=0.3):
        self.clipping_threshold = clipping_threshold

    def __call__(self, audio):
        return np.clip(audio, -self.clipping_threshold, self.clipping_threshold)


class DynamicRangeCompression:
    """Apply dynamic range compression to audio."""
    def __init__(self, threshold=0.2, ratio=4):
        self.threshold = threshold
        self.ratio = ratio

    def __call__(self, audio):
        return np.where(
            np.abs(audio) > self.threshold,
            np.sign(audio) * (self.threshold + (np.abs(audio) - self.threshold) / self.ratio),
            audio
        )


class HarmonicDistortion:
    """Add harmonic distortion by mixing harmonics at specific frequencies."""
    def __init__(self, distortion_level=0.1, harmonic_freq_ratio=2):
        self.distortion_level = distortion_level
        self.harmonic_freq_ratio = harmonic_freq_ratio

    def __call__(self, audio):
        harmonic = self.distortion_level * np.sin(2 * np.pi * np.arange(len(audio)) * self.harmonic_freq_ratio / len(audio))
        return audio + harmonic


class ReverberationSimulation:
    """Simulate reverberation by mixing delayed versions of audio."""
    def __init__(self, delay_samples=4000, decay=0.3):
        self.delay_samples = delay_samples
        self.decay = decay

    def __call__(self, audio):
        reverb_audio = np.copy(audio)
        for i in range(self.delay_samples, len(audio)):
            reverb_audio[i] += self.decay * audio[i - self.delay_samples]
        return reverb_audio


class BitDepthReduction:
    """Reduce bit depth to simulate quantization noise."""
    def __init__(self, bit_depth=8):
        self.bit_depth = bit_depth

    def __call__(self, audio):
        max_val = 2 ** (self.bit_depth - 1)
        return np.round(audio * max_val) / max_val

from scipy.signal import butter, filtfilt

class LowPassFilter:
    """Apply low-pass filter to remove high-frequency content."""
    def __init__(self, cutoff_freq, sample_rate, order=5):
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.order = order

    def __call__(self, audio):
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, audio)


class HighPassFilter:
    """Apply high-pass filter to remove low-frequency content."""
    def __init__(self, cutoff_freq, sample_rate, order=5):
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.order = order

    def __call__(self, audio):
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = butter(self.order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, audio)


class Equalizer:
    """Apply a band-pass filter for an equalizer effect."""
    def __init__(self, low_cutoff, high_cutoff, sample_rate, order=5):
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.sample_rate = sample_rate
        self.order = order

    def __call__(self, audio):
        nyquist = 0.5 * self.sample_rate
        low_normal_cutoff = self.low_cutoff / nyquist
        high_normal_cutoff = self.high_cutoff / nyquist
        b, a = butter(self.order, [low_normal_cutoff, high_normal_cutoff], btype='band')
        return filtfilt(b, a, audio)
