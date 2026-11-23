#!/usr/bin/env python3
"""Mithi v2 Flash: text to speech in 11 Indian languages.

Command line (inside this folder):
    python run.py --lang hi --voice charvi --text "आज मौसम कितना अच्छा है!" --out hello.wav
    python run.py --lang en --voice satvik --text "Hello! How are you today?" --out hello_en.wav
    python run.py --device cpu --threads 4 --lang hi --text "नमस्ते! आप कैसे हैं?" --out hello_cpu.wav

Python (inside this folder):
    from run import Mithi
    tts = Mithi()                                     # loads the model from this folder
    tts.save(tts.synthesize("नमस्ते! आप कैसे हैं?", lang="hi", voice="charvi"), "namaste.wav")

Languages: hi en bn gu kn ml mr or pa ta te
Voices:    satvik anirban nakul vasant siddhant (male), charvi ahana vaidehi malini jahnavi (female)

Includes code from Kokoro (github.com/hexgrad/kokoro, Apache-2.0) and the Maya text normaliser (Apache-2.0).
"""
import argparse
import importlib.abc
import importlib.util
import linecache
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- embedded package (loaded from memory)
_SOURCES = {
    'mithi_tts': (True, r'''"""Mithi: Indian-language text-to-speech (11 languages, 10 voices)."""
from .tts import SR, Mithi  # noqa: F401

__version__ = "1.0.0"
'''),
    'mithi_tts.frontend': (False, r'''"""Text front end: Maya text normaliser -> espeak-ng / misaki G2P -> Kokoro phoneme string, plus the words indic-bert
reads and the word index of every phoneme token.

    norm, ps = phonemes("आज मौसम कितना अच्छा है!", "hi")
"""
import json
import os
import re
import unicodedata

from .text_norm import normalize_text

LANGS = {"hi": "Hindi", "en": "English", "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
         "mr": "Marathi", "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu"}

# Kokoro-82M phoneme vocabulary (char -> id) plus the breath token the writer inserts (id 7, never emitted by G2P)
from ._here import HERE as _HERE
VOCAB = json.load(open(os.path.join(_HERE, "config.json"), encoding="utf-8"))["kokoro"]["vocab"]
BREATH, BREATH_ID = "⁂", 7
VOCAB = {**VOCAB, BREATH: BREATH_ID}

NUKTA = "़"
PUNCT = ',.!?;:—…"“”()'
SPLIT_PUNCT = re.compile(r"([,.!?;:—…\"“”()])")
LATIN_RUN = re.compile(r"([A-Za-z][A-Za-z']*(?:\s+[A-Za-z][A-Za-z']*)*)")
# espeak-ng phones some Indic languages use that Kokoro's vocabulary lacks -> the closest Kokoro phone
IPA_ALIASES = {"ɭ": "l", "ʉ": "u", "̩": "", "+": ""}
PAUSE_IDS = {VOCAB[c] for c in " ,.!?;:—…" if c in VOCAB}
SPACE = VOCAB[" "]
PUNCT_IDS = {VOCAB[c] for c in PUNCT if c in VOCAB}

# ---------------------------------------------------------------- G2P
_g2p = {}


def _short_espeak_path():
    """espeak-ng cannot open a data folder whose path is longer than about 160 characters (deep virtualenvs). In that
    case copy the data (19 MB, once) to a short folder and point espeak there."""
    try:
        import shutil
        import tempfile
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        data = espeakng_loader.get_data_path()
        if len(data) < 140:
            return
        for base in (os.path.expanduser("~/.cache/mithi"), os.path.join(tempfile.gettempdir(), "mithi")):
            short = os.path.join(base, "espeak-ng-data")
            if len(short) >= 140:
                continue
            try:
                if not os.path.exists(os.path.join(short, "phontab")):
                    os.makedirs(base, exist_ok=True)
                    tmp = short + f".tmp{os.getpid()}"
                    shutil.copytree(data, tmp)
                    os.replace(tmp, short)
            except OSError:
                continue
            EspeakWrapper.set_data_path(short)
            return
    except Exception:
        pass


def _init_g2p(lang):
    from misaki import en, espeak
    _short_espeak_path()
    if lang not in LANGS:
        raise ValueError(f"unsupported language: {lang} (one of {', '.join(LANGS)})")
    if lang != "en":
        _g2p[lang] = espeak.EspeakG2P(language=lang)
    _g2p["en"] = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))


def restore_nukta(text, text_norm):
    """The Maya normaliser drops the nukta (पेड़ -> पेड); put it back wherever the original word had one."""
    nfd = lambda w: unicodedata.normalize("NFD", w)  # noqa: E731
    table = {}
    for w in re.findall(r"[ऀ-ॿ]+", text):
        if NUKTA in nfd(w):
            table[unicodedata.normalize("NFC", nfd(w).replace(NUKTA, ""))] = unicodedata.normalize("NFC", w)
    if not table:
        return text_norm
    return re.sub(r"[ऀ-ॿ]+", lambda m: table.get(m.group(), m.group()), text_norm)


# Legacy-font typing leaves two vowel signs in a row (छाेटा, काॅक्स); espeak-ng crashes on some of them.
MATRA_FIX = [("ाॅ", "ॉ"), ("ॅा", "ॉ"), ("ाे", "ो"), ("ेा", "ो"), ("ाै", "ौ"), ("ैा", "ौ")]
DOUBLE_SIGN = re.compile(r"([ा-ौ])[ा-ौ]+")


def fix_matras(t):
    for bad, good in MATRA_FIX:
        t = t.replace(bad, good)
    return DOUBLE_SIGN.sub(r"\1", t)


def _phonemize_hi(chunk):
    chunk = re.sub(r"['\-–]+", " ", chunk).strip()        # espeak-ng hi crashes on a run starting with "'-"
    if not chunk:
        return ""
    ps = _g2p["hi"](chunk)[0]
    # espeak-ng hi writes ड़/ढ़ as "r."; inside a punctuation-free chunk any "." is that marker
    return ps.replace("r.", "ɽ").replace(".", "")


def _phonemize(text, text_norm, lang):
    t = restore_nukta(text, text_norm) if lang in ("hi", "mr") else text_norm
    if lang in ("hi", "mr"):
        t = fix_matras(t)
    t = t.replace("।", ".").replace("॥", ".")
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t)
    out = ""
    for piece in SPLIT_PUNCT.split(t):
        if not piece:
            continue
        if piece in PUNCT:
            out = out.rstrip() + piece + " "
            continue
        words = []
        for run in LATIN_RUN.split(piece):
            run = run.strip()
            if not run:
                continue
            if LATIN_RUN.fullmatch(run):
                words.append(_g2p["en"](run)[0])
            elif lang == "hi":
                words.append(_phonemize_hi(run))
            else:
                words.append(_g2p[lang](run)[0])
        if words:
            out += " ".join(w for w in words if w) + " "
    out = re.sub(r"\s+", " ", out).strip()
    for old, new in IPA_ALIASES.items():
        out = out.replace(old, new)
    return "".join(c for c in out if c in VOCAB)


def phonemes(text, lang):
    """text -> (normalised text, Kokoro phoneme string)."""
    if lang not in _g2p:
        _init_g2p(lang)
    norm = normalize_text(text, lang)
    return norm, _phonemize(text, norm, lang)


# ---------------------------------------------------------------- words for indic-bert
def words_of(text, text_norm):
    """The words the meaning encoder reads (same transformations as _phonemize), punctuation attached to the word
    before it. The n-th space-separated phoneme group is the n-th word."""
    t = fix_matras(restore_nukta(text, text_norm)).replace("।", ".").replace("॥", ".")
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t)
    t = re.sub(r"['\-–]+", " ", t)
    out = ""
    for piece in SPLIT_PUNCT.split(t):
        if not piece:
            continue
        out = out.rstrip() + piece + " " if piece in PUNCT else out + piece.strip() + " "
    words = []
    for g in (w for w in re.sub(r"\s+", " ", out).split(" ") if w):
        if all(c in PUNCT for c in g):          # a punctuation-only group joins the word before it
            if words:
                words[-1] += g
        else:
            words.append(g)
    return words


def token_words(ids):
    """Word index per token (BOS/EOS and spaces -1, punctuation / breath -> the word before) and the word count."""
    out, w, in_word, last = [], -1, False, -1
    for k, t in enumerate(ids):
        if k == 0 or k == len(ids) - 1 or t == SPACE:
            out.append(-1)
            in_word = False
        elif t == BREATH_ID or t in PUNCT_IDS:
            out.append(last)
        else:
            if not in_word:
                w += 1
                in_word = True
            out.append(w)
            last = w
    return out, w + 1


# ---------------------------------------------------------------- long text
def split_text(text, lang, limit=400):
    """The whole text if its phonemes fit, else sentences (and, if still too long, comma pieces) under the limit.
    Odia ୱ is written ବ (espeak-ng spells ୱ words letter by letter)."""
    text = " ".join(text.split())
    if lang == "or":
        text = text.replace("ୱ", "ବ")
    if len(phonemes(text, lang)[1]) <= limit:
        return [text]
    parts = [s.strip() for s in re.split(r"(?<=[.!?।॥])\s+", text) if s.strip()]
    out = []
    for s in parts:
        if len(phonemes(s, lang)[1]) <= limit:
            out.append(s)
            continue
        cur = ""
        for piece in re.split(r"(?<=[,;:])\s+", s):
            if cur and len(phonemes(cur + " " + piece, lang)[1]) > limit:
                out.append(cur)
                cur = piece
            else:
                cur = (cur + " " + piece).strip()
        if cur:
            out.append(cur)
    return out
'''),
    'mithi_tts.indic_encoder': (False, r'''"""Meaning encoder for the prosody writer: ai4bharat/indic-bert (ALBERT) with its word table trimmed to the 105,229
pieces our languages use (21.3 M parameters, outputs identical to the full model).

Preprocessing matches indic-bert's pretraining (ALBERT's preprocess_text): collapse spaces, NFKD, drop combining marks
(in Indic scripts: virama and nukta; vowel signs stay), lowercase. SentencePiece is called directly (the transformers
AlbertTokenizer also strips the vowel signs) and its ids are remapped to the trimmed table.
"""
import unicodedata

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from transformers import AlbertModel

CLS, SEP = 2, 3


def preprocess(text):
    text = unicodedata.normalize("NFKD", " ".join(text.split()))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


class IndicTextEncoder(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.sp = spm.SentencePieceProcessor(model_file=f"{model_dir}/spiece.model")
        self.register_buffer("old2new", torch.from_numpy(np.load(f"{model_dir}/old2new.npy")), persistent=False)
        self.bert = AlbertModel.from_pretrained(model_dir, dtype=torch.float32)
        self.dim = self.bert.config.hidden_size

    def pieces(self, words):
        ids, owner = [CLS], [-1]
        for w, word in enumerate(words):
            p = self.sp.encode(preprocess(word))
            ids += p
            owner += [w] * len(p)
        ids.append(SEP)
        owner.append(-1)
        return self.old2new[torch.tensor(ids)], torch.tensor(owner)

    def word_vectors(self, text):
        """One vector per space-separated word of `text` (mean of its pieces' last hidden states)."""
        words = text.split()
        ids, owner = self.pieces(words)
        h = self.bert(input_ids=ids[None].to(self.old2new.device)).last_hidden_state[0]
        out = torch.zeros(len(words), self.dim, device=h.device, dtype=h.dtype)
        cnt = torch.zeros(len(words), 1, device=h.device, dtype=h.dtype)
        sel = owner >= 0
        out.index_add_(0, owner[sel].to(h.device), h[sel.to(h.device)])
        cnt.index_add_(0, owner[sel].to(h.device), torch.ones(int(sel.sum()), 1, device=h.device, dtype=h.dtype))
        return out / cnt.clamp(min=1), words
'''),
    'mithi_tts.kokoro': (True, r'''"""Kokoro-82M text encoder and iSTFTNet decoder (hexgrad/kokoro, Apache-2.0), vendored unchanged except for imports."""
'''),
    'mithi_tts.kokoro.custom_stft': (False, r'''import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomSTFT(nn.Module):
    """
    STFT/iSTFT without unfold/complex ops, using conv1d and conv_transpose1d.

    - forward STFT => Real-part conv1d + Imag-part conv1d
    - inverse STFT => Real-part conv_transpose1d + Imag-part conv_transpose1d + sum
    - avoids F.unfold, so easier to export to ONNX
    - uses replicate or constant padding for 'center=True' to approximate 'reflect' 
      (reflect is not supported for dynamic shapes in ONNX)
    """

    def __init__(
        self,
        filter_length=800,
        hop_length=200,
        win_length=800,
        window="hann",
        center=True,
        pad_mode="replicate",  # or 'constant'
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.center = center
        self.pad_mode = pad_mode

        # Number of frequency bins for real-valued STFT with onesided=True
        self.freq_bins = self.n_fft // 2 + 1

        # Build window
        assert window == 'hann', window
        window_tensor = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        if self.win_length < self.n_fft:
            # Zero-pad up to n_fft
            extra = self.n_fft - self.win_length
            window_tensor = F.pad(window_tensor, (0, extra))
        elif self.win_length > self.n_fft:
            window_tensor = window_tensor[: self.n_fft]
        self.register_buffer("window", window_tensor)

        # Precompute forward DFT (real, imag)
        # PyTorch stft uses e^{-j 2 pi k n / N} => real=cos(...), imag=-sin(...)
        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)
        angle = 2 * np.pi * np.outer(k, n) / self.n_fft  # shape (freq_bins, n_fft)
        dft_real = np.cos(angle)
        dft_imag = -np.sin(angle)  # note negative sign

        # Combine window and dft => shape (freq_bins, filter_length)
        # We'll make 2 conv weight tensors of shape (freq_bins, 1, filter_length).
        forward_window = window_tensor.numpy()  # shape (n_fft,)
        forward_real = dft_real * forward_window  # (freq_bins, n_fft)
        forward_imag = dft_imag * forward_window

        # Convert to PyTorch
        forward_real_torch = torch.from_numpy(forward_real).float()
        forward_imag_torch = torch.from_numpy(forward_imag).float()

        # Register as Conv1d weight => (out_channels, in_channels, kernel_size)
        # out_channels = freq_bins, in_channels=1, kernel_size=n_fft
        self.register_buffer(
            "weight_forward_real", forward_real_torch.unsqueeze(1)
        )
        self.register_buffer(
            "weight_forward_imag", forward_imag_torch.unsqueeze(1)
        )

        # Precompute inverse DFT
        # Real iFFT formula => scale = 1/n_fft, doubling for bins 1..freq_bins-2 if n_fft even, etc.
        # For simplicity, we won't do the "DC/nyquist not doubled" approach here. 
        # If you want perfect real iSTFT, you can add that logic. 
        # This version just yields good approximate reconstruction with Hann + typical overlap.
        inv_scale = 1.0 / self.n_fft
        n = np.arange(self.n_fft)
        angle_t = 2 * np.pi * np.outer(n, k) / self.n_fft  # shape (n_fft, freq_bins)
        idft_cos = np.cos(angle_t).T  # => (freq_bins, n_fft)
        idft_sin = np.sin(angle_t).T  # => (freq_bins, n_fft)

        # Multiply by window again for typical overlap-add
        # We also incorporate the scale factor 1/n_fft
        inv_window = window_tensor.numpy() * inv_scale
        backward_real = idft_cos * inv_window  # (freq_bins, n_fft)
        backward_imag = idft_sin * inv_window

        # We'll implement iSTFT as real+imag conv_transpose with stride=hop.
        self.register_buffer(
            "weight_backward_real", torch.from_numpy(backward_real).float().unsqueeze(1)
        )
        self.register_buffer(
            "weight_backward_imag", torch.from_numpy(backward_imag).float().unsqueeze(1)
        )
        


    def transform(self, waveform: torch.Tensor):
        """
        Forward STFT => returns magnitude, phase
        Output shape => (batch, freq_bins, frames)
        """
        # waveform shape => (B, T).  conv1d expects (B, 1, T).
        # Optional center pad
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)

        x = waveform.unsqueeze(1)  # => (B, 1, T)
        # Convolution to get real part => shape (B, freq_bins, frames)
        real_out = F.conv1d(
            x,
            self.weight_forward_real,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        # Imag part
        imag_out = F.conv1d(
            x,
            self.weight_forward_imag,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )

        # magnitude, phase
        magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
        phase = torch.atan2(imag_out, real_out)
        # Handle the case where imag_out is 0 and real_out is negative to correct ONNX atan2 to match PyTorch
        # In this case, PyTorch returns pi, ONNX returns -pi
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase[correction_mask] = torch.pi
        return magnitude, phase


    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, length=None):
        """
        Inverse STFT => returns waveform shape (B, T).
        """
        # magnitude, phase => (B, freq_bins, frames)
        # Re-create real/imag => shape (B, freq_bins, frames)
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)

        # conv_transpose wants shape (B, freq_bins, frames). We'll treat "frames" as time dimension
        # so we do (B, freq_bins, frames) => (B, freq_bins, frames)
        # But PyTorch conv_transpose1d expects (B, in_channels, input_length)
        real_part = real_part  # (B, freq_bins, frames)
        imag_part = imag_part

        # real iSTFT => convolve with "backward_real", "backward_imag", and sum
        # We'll do 2 conv_transpose calls, each giving (B, 1, time),
        # then add them => (B, 1, time).
        real_rec = F.conv_transpose1d(
            real_part,
            self.weight_backward_real,  # shape (freq_bins, 1, filter_length)
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        imag_rec = F.conv_transpose1d(
            imag_part,
            self.weight_backward_imag,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        # sum => (B, 1, time)
        waveform = real_rec - imag_rec  # typical real iFFT has minus for imaginary part

        # If we used "center=True" in forward, we should remove pad
        if self.center:
            pad_len = self.n_fft // 2
            # Because of transposed convolution, total length might have extra samples
            # We remove `pad_len` from start & end if possible
            waveform = waveform[..., pad_len:-pad_len]

        # If a specific length is desired, clamp
        if length is not None:
            waveform = waveform[..., :length]

        # shape => (B, T)
        return waveform

    def forward(self, x: torch.Tensor):
        """
        Full STFT -> iSTFT pass: returns time-domain reconstruction.
        Same interface as your original code.
        """
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])
'''),
    'mithi_tts.kokoro.istftnet': (False, r'''# ADAPTED from https://github.com/yl4579/StyleTTS2/blob/main/Modules/istftnet.py
from .custom_stft import CustomSTFT
from torch.nn.utils.parametrizations import weight_norm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/yl4579/StyleTTS2/blob/main/Modules/utils.py
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        # affine should be False, however there's a bug in the old torch.onnx.export (not newer dynamo) that causes the channel dimension to be lost if affine=False. When affine is true, there's additional learnably parameters. This shouldn't really matter setting it to True, since we're in inference mode
        self.norm = nn.InstanceNorm1d(num_features, affine=True)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super(AdaINResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                  padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                                  padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                                  padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        self.adain1 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        self.adain2 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x


class TorchSTFT(nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        assert window == 'hann', window
        self.window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device),
            return_complex=True)
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))
        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class SineGen(nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(torch.pi) or cos(0)
    """
    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * torch.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1/self.upsample_scale, mode="linear").transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
            phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
            sines = torch.sin(phase)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation
            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            # get the sines
            sines = torch.cos(i_phase * 2 * torch.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        # fundamental component
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp
        # generate uv signal
        # uv = torch.ones(f0.shape)
        # uv = uv * (f0 > self.voiced_threshold)
        uv = self._f02uv(f0)
        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        #        for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """
    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)
        # to merge source harmonics into a single excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class Generator(nn.Module):
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, disable_complex=False):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
                    sampling_rate=24000,
                    upsample_scale=math.prod(upsample_rates) * gen_istft_hop_size,
                    harmonic_num=8, voiced_threshod=10)
        self.f0_upsamp = nn.Upsample(scale_factor=math.prod(upsample_rates) * gen_istft_hop_size)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                   k, u, padding=(k-u)//2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes,resblock_dilation_sizes)):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(
                    gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1,3,5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1,3,5], style_dim))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(nn.Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = (
            CustomSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)
            if disable_complex
            else TorchSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)
        )

    def forward(self, x, s, f0):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
            har_source, noi_source, uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1) 
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        return self.stft.inverse(spec, phase)


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2))
        return out


class Decoder(nn.Module):
    def __init__(self, dim_in, style_dim, dim_out, 
                 resblock_kernel_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 resblock_dilation_sizes,
                 upsample_kernel_sizes,
                 gen_istft_n_fft, gen_istft_hop_size,
                 disable_complex=False):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList()
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(512, 64, kernel_size=1)))
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, 
                                   upsample_initial_channel, resblock_dilation_sizes, 
                                   upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, disable_complex=disable_complex)

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        x = self.generator(x, s, F0_curve)
        return x
'''),
    'mithi_tts.kokoro.modules': (False, r'''# https://github.com/yl4579/StyleTTS2/blob/main/models.py
from .istftnet import AdainResBlk1d
from torch.nn.utils.parametrizations import weight_norm
from transformers import AlbertModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # [B, T, chn]
        lengths = input_lengths if input_lengths.device == torch.device('cpu') else input_lengths.to('cpu')
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad
        x.masked_fill_(m, 0.0)
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid,nlayers=nlayers, dropout=dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        m = m.unsqueeze(1)
        lengths = text_lengths if text_lengths.device == torch.device('cpu') else text_lengths.to('cpu')
        x = nn.utils.rnn.pack_padded_sequence(d, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, :x.shape[1], :] = x
        x = x_pad
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
        en = (d.transpose(-1, -2) @ alignment)
        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, d_model // 2, num_layers=1, batch_first=True, bidirectional=True))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                lengths = text_lengths if text_lengths.device == torch.device('cpu') else text_lengths.to('cpu')
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad

        return x.transpose(-1, -2)


# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state
'''),
    'mithi_tts.prosody': (False, r'''"""Prosody writer: reads a sentence (phoneme tokens + indic-bert word meaning + speaker + language) and writes Maya-style
prosody one phoneme token at a time, as discrete codes; decode() turns the codes into the per-token durations and the
frame-level F0 / energy the Kokoro decoder takes (2 frames per 25 ms unit).

Codes per token (P5): dur, uv0, uv1, p1..p5 (pitch, 0.5-semitone bins re 100 Hz), e1..e3 (energy, 0.25 bins),
then breath_next (1 = the writer inserts a breath token after this one).
The writer predicts voicing explicitly and pitch / energy as changes from the previous point (v2 design), and its depth
heads predict the codes of one token in order, each seeing the ones before it.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .frontend import BREATH_ID

# ---------------------------------------------------------------- code tables
DUR_MAX, UV_MAX = 63, 31
ST_LO, ST_STEP, N_ST = -10.0, 0.5, 81            # -10 .. +30 semitones
PITCH_NONE = N_ST                                # extra class: no voiced frame
E_LO, E_STEP, N_E = -6.0, 0.25, 65               # -6 .. +10
K_PITCH, J_ENERGY = 5, 3                         # P5: 5 pitch + 3 energy points
P5_COLS = 3 + K_PITCH + J_ENERGY                 # 11 = index of breath_next in what sample() returns


def _interp(points, n):
    """n frame values through k points placed at the centres of k equal parts."""
    k = len(points)
    if n == 0:
        return np.zeros(0)
    pos = (np.arange(k) + 0.5) / k * n - 0.5
    return np.interp(np.arange(n), pos, points)


def decode(codes):
    """P5 codes [T, 11] -> dur [T] units, f0 [2*sum(dur)] Hz, energy [2*sum(dur)]."""
    k = K_PITCH
    dur = codes[:, 0].astype(np.int64)
    n = 2 * int(dur.sum())
    f0, en, s = np.zeros(n, np.float32), np.zeros(n, np.float32), 0
    for t, d in enumerate(dur):
        if d == 0:
            continue
        a, b = 2 * s, 2 * (s + d)
        s += d
        en[a:b] = _interp(E_LO + E_STEP * codes[t, 3 + k:].astype(np.float64), b - a)
        p = codes[t, 3:3 + k]
        if (p == PITCH_NONE).any():
            continue
        v0, v1 = a + min(int(codes[t, 1]), b - a), b - min(int(codes[t, 2]), b - a)
        if v1 > v0:
            st = _interp(ST_LO + ST_STEP * p.astype(np.float64), v1 - v0)
            f0[v0:v1] = 100 * 2 ** (st / 12)
    return dur, f0, en


# ---------------------------------------------------------------- writer
NAMES = ["dur", "voiced", "uv0", "uv1", "d1", "d2", "d3", "d4", "d5", "f1", "f2", "f3", "breath"]
WRITER_LANGS = ["hi", "en", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "as"]
D_P, D_E = 80, 64
REF_P = int(round((0 - ST_LO) / ST_STEP))           # 0 st re 100 Hz: the reference before the first voiced token
REF_E = int(round((0 - E_LO) / E_STEP))
ABS_CLASSES = [DUR_MAX + 1, 2, UV_MAX + 1, UV_MAX + 1] + [N_ST + 1] * 5 + [N_E] * 3 + [2]
REL_CLASSES = [DUR_MAX + 1, 2, UV_MAX + 1, UV_MAX + 1] + [2 * D_P + 1] * 5 + [2 * D_E + 1] * 3 + [2]
VOICED, BREATH_NEXT = 1, 12                         # columns of the 13-column absolute layout


def abs_to_p5(codes_abs):
    """absolute layout [N, 13] -> (P5 codes [N, 11], breath_next [N])."""
    return torch.cat([codes_abs[:, :1], codes_abs[:, 2:12]], 1), codes_abs[:, BREATH_NEXT]


class ProsodyWriter(nn.Module):
    def __init__(self, n_tok=178, word_dim=768, d=384, heads=6, enc_layers=4, dec_layers=4, n_spk=2,
                 n_lang=len(WRITER_LANGS), depth_hid=256, dropout=0.1):
        super().__init__()
        self.d = d
        self.tok = nn.Embedding(n_tok, d)
        self.word = nn.Linear(word_dim, d)
        self.no_word = nn.Parameter(torch.zeros(d))
        self.spk = nn.Embedding(n_spk, d)
        self.lang = nn.Embedding(n_lang, d)
        self.enc_pos = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True),
                                         enc_layers, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)
        self.code_emb = nn.ModuleList([nn.Embedding(c, d) for c in ABS_CLASSES])
        self.start = nn.Parameter(torch.zeros(d))
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True),
                                         dec_layers)
        self.dec_norm = nn.LayerNorm(d)
        self.depth = nn.ModuleList([nn.Sequential(nn.Linear(d, depth_hid), nn.GELU(), nn.Linear(depth_hid, c)) for c in REL_CLASSES])

    def read(self, enc_ids, word_vec, has_word, spk, lang, enc_pad):
        x = (self.tok(enc_ids) + torch.where(has_word[..., None], self.word(word_vec), self.no_word)
             + (self.spk(spk) + self.lang(lang))[:, None])
        x = x.masked_fill(enc_pad[..., None], 0)
        x = x + self.enc_pos(x.transpose(1, 2)).transpose(1, 2)
        return self.enc_norm(self.enc(x, src_key_padding_mask=enc_pad))

    def _prev_codes(self, ctx):
        e = sum(emb(ctx[..., k]) for k, emb in enumerate(self.code_emb))
        return torch.cat([self.start.expand(ctx.shape[0], 1, self.d), e[:, :-1]], 1)

    def write(self, H, enc_pad, dec_ids, dec_map, ctx, dec_pad):
        x = torch.gather(H, 1, dec_map[..., None].expand(-1, -1, self.d)) + self.tok(dec_ids) + self._prev_codes(ctx)
        causal = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), 1)
        h = self.dec(x, H, tgt_mask=causal, tgt_key_padding_mask=dec_pad, memory_key_padding_mask=enc_pad, tgt_is_causal=True)
        return self.dec_norm(h)

    @staticmethod
    def _abs_from_rel(k, c, abs_row, prev_p5, prev_e3):
        if k == 4:
            return (prev_p5 + c - D_P).clamp(0, N_ST - 1)
        if 5 <= k <= 8:
            return (abs_row[k - 1] + c - D_P).clamp(0, N_ST - 1)
        if k == 9:
            return (prev_e3 + c - D_E).clamp(0, N_E - 1)
        if 10 <= k <= 11:
            return (abs_row[k - 1] + c - D_E).clamp(0, N_E - 1)
        return c

    @torch.no_grad()
    def sample(self, H, enc_ids, pause_ids, temp=0.8, top_p=0.9, max_breaths=None, head_temp=None):
        """Write one sentence (batch of 1), token by token. head_temp = {code name: temperature} (0 = most likely).
        Returns (dec_ids [Nd] incl. inserted breaths, P5 codes + breath_next [Nd, 12])."""
        dev = H.device
        dec_ids, dec_map, rows = [], [], []
        enc_pad = torch.zeros(1, H.shape[1], dtype=torch.bool, device=dev)
        queue = [(int(t), i) for i, t in enumerate(enc_ids.tolist())]
        n_breath, prev_p5, prev_e3 = 0, REF_P, REF_E
        while queue:
            tok, i = queue.pop(0)
            dec_ids.append(tok)
            dec_map.append(i)
            ctx = torch.tensor(rows + [[0] * len(NAMES)], device=dev)[None]
            ids = torch.tensor(dec_ids, device=dev)[None]
            h = self.write(H, enc_pad, ids, torch.tensor(dec_map, device=dev)[None], ctx,
                           torch.zeros_like(ids, dtype=torch.bool))[0, -1]
            row, acc = [], h
            for k, head in enumerate(self.depth):
                t = (head_temp or {}).get(NAMES[k], temp)
                if 2 <= k <= 8 and row[VOICED] == 0:
                    a = PITCH_NONE if k >= 4 else 0
                else:
                    c = _sample(head(acc), t, 1.0 if k == BREATH_NEXT else top_p)
                    a = int(self._abs_from_rel(k, torch.tensor(c), torch.tensor(row + [0]), torch.tensor(prev_p5), torch.tensor(prev_e3))) \
                        if 4 <= k <= 11 else c
                if k == BREATH_NEXT and (tok not in pause_ids or tok == BREATH_ID or not queue
                                         or (max_breaths is not None and n_breath >= max_breaths)):
                    a = 0
                row.append(a)
                acc = acc + self.code_emb[k](torch.tensor(a, device=dev))
            rows.append(row)
            if row[VOICED] == 1:
                prev_p5 = row[8]
            prev_e3 = row[11]
            if row[BREATH_NEXT] == 1:
                queue.insert(0, (BREATH_ID, i))
                n_breath += 1
        out = torch.tensor(rows)
        p5, breath = abs_to_p5(out)
        return torch.tensor(dec_ids), torch.cat([p5, breath[:, None]], 1)


def _sample(logits, temp, top_p):
    if temp <= 0:
        return int(logits.argmax())
    p = F.softmax(logits / temp, -1)
    sp, idx = p.sort(descending=True)
    keep = sp.cumsum(-1) - sp < top_p
    sp = sp * keep
    return int(idx[torch.multinomial(sp / sp.sum(), 1)])
'''),
    'mithi_tts.text_norm': (True, r'''"""Text normalisation for Maya3 (numbers, currency, dates, OTPs, URLs, ... -> spoken words, per language).

    from mithi_tts.text_norm import normalise_text
    normalise_text("Your OTP is 482913 [laughs]", "en")   # 'Your OTP is four, eight, two, nine, one, three [laughs]'

Same behaviour as the normalise_text wrapper the release evals used:
  * inline [tags] (e.g. [laughs], [sighs]) are protected: the text between them is normalised piece by piece;
  * ALL-CAPS Latin words that look like names (>= 5 letters with >= 2 vowels, or 4 letters with >= 2 vowels) are
    title-cased first, so the normaliser and the model read them as words; short / vowel-poor tokens (OTP, NEFT,
    HDFC, KYC) stay acronyms;
  * each piece goes through the bundled normaliser mithi_tts.text_norm.tn (pure Python, standard library only);
  * if the normaliser raises on a piece, that piece is kept unchanged (the eval wrapper's second fallback, a
    separate benchmark pipeline, is not part of the release);
  * whitespace is collapsed and the result stripped;
  * `rules=True` (default) adds the targeted pre/post rules of mithi_tts.text_norm.rules (two-digit-year dates, card expiry,
    FY ranges, spaced card/phone digits, PIN codes, version strings, rupees+paise in every language, the comma
    before an inline [tag]). `rules=False` is the exact behaviour of the release evals before these rules were added.
`lang` is a Maya3 language code (as bn en gu hi kn ml mr or pa ta te); for an unknown code the normaliser raises,
so the text comes back with only the caps fold and whitespace collapse applied.
"""
from __future__ import annotations

import logging
import re

from mithi_tts.text_norm import lexicon as _lexicon
from mithi_tts.text_norm import rules as _rules

__all__ = ["normalise_text", "normalize_text", "SOURCE"]

SOURCE = "Maya Research text normaliser, upstream commit fe5e6ec"

_TAG = re.compile(r"(\[[a-z ]+\])")
_CAPS = re.compile(r"\b[A-Z]{4,}\b")
_VOWELS = set("AEIOU")
_TN = None

log = logging.getLogger("mithi_tts.text_norm")


def _caps_fold(t):
    def f(m):
        w = m.group(0)
        if len(w) >= 5 and sum(ch in _VOWELS for ch in w) >= 2 or (len(w) == 4 and sum(ch in _VOWELS for ch in w) >= 2):
            return w.title()
        return w
    return _CAPS.sub(f, t)


def _load_tn():
    global _TN
    if _TN is None:
        from mithi_tts.text_norm.tn.normalize import normalize
        _TN = normalize
    return _TN


def normalise_text(text: str, lang: str, rules: bool = True) -> str:
    tn = _load_tn()
    _rules.install(rules)
    _lexicon.install(rules)
    out = []
    segs = _TAG.split(text)
    for i, seg in enumerate(segs):
        if not seg or _TAG.fullmatch(seg):
            out.append(seg)
            continue
        s = _caps_fold(seg)
        try:
            s2 = tn(_rules.pre(s, lang) if rules else s, lang)
            if rules:
                s2 = _rules.post(s2, seg, lang, before_tag=any(_TAG.fullmatch(x) for x in segs[i + 1:i + 2]))
        except Exception as exc:   # never fail a request on normalisation: keep this piece as written
            log.debug("normaliser fallback (%s: %s)", type(exc).__name__, exc)
            s2 = s
        lead = seg[:len(seg) - len(seg.lstrip())]
        trail = seg[len(seg.rstrip()):]
        out.append(lead + s2.strip() + trail)
    return re.sub(r"\s+", " ", "".join(out)).strip()


normalize_text = normalise_text
'''),
    'mithi_tts.text_norm.lexicon': (False, r'''"""Language-table corrections applied around the vendored normaliser (mithi_tts.text_norm.tn), found by a multi-language
normaliser benchmark.

The vendored copy stays byte-identical. `install(True)` (called by mithi_tts.text_norm.normalise_text with rules on) patches
the module-level language tables / one Normalizer method; `install(False)` restores the originals exactly, so
`normalise_text(..., rules=False)` keeps the eval-time behaviour.

  L1 as_lexicon  Assamese inherited Bengali grammar tables: Bengali र (U+09B0) instead of Assamese ৰ (U+09F0) in the
                 number words (ত্রিশ -> ত্ৰিশ), Bengali ordinals / symbol words / letter names / clock words
                 (চতুর্থ, অ্যাট দ্য রেট, আর, সকাল, দুপুর, রাত, "{h}টা", "এর বর্গ" ...). Replaced with Assamese forms;
                 Assamese clock markers (বজাত / বজাৰ / বজালৈ) and period words (পুৱা, আবেলি, বিয়লি, ৰাতি ...)
                 are recognised, and postpositions glue to the number noun (পঞ্চাশ টকা ৰ -> পঞ্চাশ টকাৰ).
  L2 per_unit    "km/h", "MB/s": the per word was the English "per" in bn as gu kn ml mr pa ta, and the
                 denominator was a spelled letter ("কে এম per এইচ"); now the language's per word + hour / second.
  L3 uri_hosts   bare domains with sub-domains (e-receipt.hdfcbank.com, netbanking.nationalbank.in) were only
                 matched from the last label, so "e-receipt.hdfcbank" stayed raw; a bare domain that ends a sentence
                 ("irctc.co.in.") was not matched at all; the local part of a UPI ID (rahul.deshmukh@okaxis) stayed
                 raw. All three are now spelled like other URLs.
"""
from __future__ import annotations

import re
import types

__all__ = ["install", "AS_ONES"]

_R, _RR = "র", "ৰ"      # Bengali ra, Assamese ra


def _as_script(s):
    return s.replace(_R, _RR) if isinstance(s, str) else s


# Assamese 0-99: AI4Bharat indic-numtowords data (tn/assamese_numbers.py) with the Bengali ra fixed to ৰ
# (ঊনত্রিশ..আঠত্রিশ -> ঊনত্ৰিশ..আঠত্ৰিশ) and two malformed entries corrected: 45 পঁঞ্চল্লিশ -> পঞ্চল্লিশ,
# 71 এসত্তৰ -> একসত্তৰ.
def _as_ones():
    from mithi_tts.text_norm.tn.assamese_numbers import ONES
    ones = [_as_script(w) for w in ONES]
    ones[45] = "পঞ্চল্লিশ"
    ones[71] = "একসত্তৰ"
    return ones


AS_ONES = None   # filled on first install

_AS_SYM = {
    "at": "এট দ্য ৰেট", "underscore": "আণ্ডাৰস্ক'ৰ", "slash": "শ্লেচ", "hash": "হেচ", "amp": "এণ্ড",
    "squared": "বৰ্গ", "sqrt": "বৰ্গমূল", "lt": "তকৈ কম", "gt": "তকৈ বেছি",
    "half": "আধা", "quarter": "এক চতুৰ্থাংশ", "three_quarters": "তিনি চতুৰ্থাংশ",
    "degree": "ডিগ্ৰী", "celsius": "ডিগ্ৰী চেলছিয়াছ", "fahrenheit": "ডিগ্ৰী ফাৰেনহাইট",
    "second": "ছেকেণ্ড", "cent": "চেণ্ট", "per": "প্ৰতি",
}
_AS_PERIOD_DETECT = {
    "পুৱা": "morning", "পুৱাৰ": "morning", "পুৱাতে": "morning", "ৰাতিপুৱা": "morning",
    "দুপৰীয়া": "afternoon", "দুপৰীয়াৰ": "afternoon", "দুপৰীয়াত": "afternoon",
    "আবেলি": "evening", "আবেলিৰ": "evening", "আবেলিতে": "evening", "বিয়লি": "evening", "বিয়লিৰ": "evening",
    "সন্ধিয়া": "evening", "সন্ধিয়াৰ": "evening", "সন্ধিয়াতে": "evening",
    "ৰাতি": "night", "ৰাতিৰ": "night", "ৰাতিতে": "night",
}
_AS_ATTACH = {"টকা", "পইচা", "ডলাৰ", "শতাংশ", "মিনিট", "ঘণ্টা", "ছেকেণ্ড", "বজা", "ডিগ্ৰী", "কিলোমিটাৰ"}
_AS_PP = ("ৰ", "ত", "লৈ", "তকৈ", "ৰে", "তে", "লৈকে", "ৰপৰা")


def _as_attach(self, rep, tok):
    """Assamese case endings glue to the noun without sandhi: টকা+ৰ -> টকাৰ, মিনিট+ত -> মিনিটত."""
    t = tok.lstrip("-")
    last = rep.rsplit(" ", 1)[-1]
    if last not in _AS_ATTACH or t not in _AS_PP:
        return None
    return rep + t


_PER = {"bn": "প্রতি", "gu": "પ્રતિ", "mr": "प्रति", "pa": "ਪ੍ਰਤੀ",
        "ta": "பெர்", "kn": "ಪರ್", "ml": "പെർ"}            # hi or te already have one; as: _AS_SYM
_PER_UNITS = {   # denominators of X/h, X/s (H, S are not units on their own, so only "km/h"-style reads change)
    "en": {"H": "hour", "S": "second"},
    "hi": {"H": "घंटा", "S": "सेकंड"}, "mr": {"H": "तास", "S": "सेकंद"},
    "bn": {"H": "ঘণ্টা", "S": "সেকেন্ড"}, "as": {"H": "ঘণ্টা", "S": "ছেকেণ্ড"},
    "gu": {"H": "કલાક", "S": "સેકન્ડ"}, "pa": {"H": "ਘੰਟਾ", "S": "ਸਕਿੰਟ"}, "or": {"H": "ଘଣ୍ଟା", "S": "ସେକେଣ୍ଡ"},
    "ta": {"H": "ஹவர்", "S": "செகண்ட்"}, "te": {"S": "సెకండ్"},          # te already has H
    "kn": {"H": "ಅವರ್", "S": "ಸೆಕೆಂಡ್"}, "ml": {"H": "അവർ", "S": "സെക്കൻഡ്"},
}

_saved = []          # (restore callable) in apply order
_installed = False


def _set_attr(obj, name, value):
    had = name in vars(obj)
    old = vars(obj).get(name)
    setattr(obj, name, value)
    _saved.append(lambda: setattr(obj, name, old) if had else delattr(obj, name))


def _set_key(d, key, value):
    had = key in d
    old = d.get(key)
    d[key] = value
    _saved.append(lambda: d.__setitem__(key, old) if had else d.pop(key, None))


def _patch_url(N):
    orig = N._m_url

    def _m_url(self, text):
        # as the vendored _m_url, but a bare host may have sub-domain labels (e-receipt.hdfcbank.com) and may end a
        # sentence ("... irctc.co.in." was left unspelled: the vendored lookahead rejected any following ".")
        pat = (r"https?://[^\s<>]+|www\.[^\s<>]+"
               r"|\b[A-Za-z][\w\-]*(?:\.[A-Za-z0-9][\w\-]*)*\.(?:com|org|net|ai|io|co|dev|app|edu|gov|in)"
               r"(?:\.(?:in|uk|us))?(?:[/?#][^\s<>]*)?(?![\w]|\.\w)")
        for m in re.finditer(pat, text):
            token = m.group(0).rstrip(".,!?;)")
            body = re.sub(r"^https?://", "", token)
            yield m.start(), m.start() + len(token), self._spell_uri_part(body)
        # UPI IDs with a dotted local part (rahul.deshmukh@okaxis) were left raw (the "." unspoken); spell the ID
        for m in re.finditer(r"(?<![\w.@])[A-Za-z][\w\-]*\.[\w.\-]*@[A-Za-z][A-Za-z0-9]{1,15}(?![\w@]|\.\w)", text):
            yield m.start(), m.end(), self._spell_uri_part(m.group(0))

    N._m_url = _m_url
    _saved.append(lambda: setattr(N, "_m_url", orig))


def install(on: bool = True):
    global _installed, AS_ONES
    if on == _installed:
        return
    import sys
    import mithi_tts.text_norm.tn.normalize  # noqa: F401
    mod = sys.modules["mithi_tts.text_norm.tn.normalize"]
    if on:
        L = mod.language_table("as")
        AS_ONES = AS_ONES or _as_ones()
        _set_attr(L, "ONES", AS_ONES)
        _set_attr(L, "ORDINAL_EXCEPTIONS", {1: "প্ৰথম", 2: "দ্বিতীয়", 3: "তৃতীয়", 4: "চতুৰ্থ", 5: "পঞ্চম",
                                            6: "ষষ্ঠ", 7: "সপ্তম", 8: "অষ্টম", 9: "নৱম", 10: "দশম"})
        _set_attr(L, "LETTERS", {**{k: _as_script(v) for k, v in L.LETTERS.items()}, "S": "এছ", "Y": "ৱাই"})
        for k, v in list(L.SYM.items()):
            if _as_script(v) != v:
                _set_key(L.SYM, k, _as_script(v))
        for k, v in _AS_SYM.items():
            _set_key(L.SYM, k, v)
        _set_attr(L, "PERIODS", {"morning": "পুৱা", "afternoon": "দুপৰীয়া", "evening": "সন্ধিয়া", "night": "ৰাতি"})
        _set_attr(L, "PERIOD_DETECT", {**L.PERIOD_DETECT, **_AS_PERIOD_DETECT})
        _set_attr(L, "TIME_ON_HOUR", "{h} বজা")
        # Bengali "টা" is a counting classifier in Assamese (2023 টা অভিযোগ), not an o'clock marker
        _set_attr(L, "TIME_MARKERS", ["বাজে", "বজাত", "বজাৰ", "বজালৈ", "বজাতকৈ", "বজা"])
        _set_attr(L, "MARKER_SUFFIXES", {"বজাত": "ত", "বজাৰ": "ৰ", "বজালৈ": "লৈ", "বজাতকৈ": "তকৈ"})
        _set_attr(L, "POWER", "{base}ৰ ঘাত {exp}")
        _set_attr(L, "DECADE_SHORT", "{tens}ৰ দশক")
        _set_attr(L, "DECADE_FULL", "{tens}ৰ দশক")
        _set_attr(L, "ABBREV_UNITS", {"MINS": "মিনিট", "MIN": "মিনিট", "HRS": "ঘণ্টা", "HR": "ঘণ্টা",
                                      "SECS": "ছেকেণ্ড", "SEC": "ছেকেণ্ড", "KMS": "কিলোমিটাৰ"})
        _set_attr(L, "attach_postposition", types.MethodType(_as_attach, L))
        for code, word in _PER.items():
            sym = mod.language_table(code).SYM
            if "per" not in sym:
                _set_key(sym, "per", word)
        for code, words in _PER_UNITS.items():
            T = mod.language_table(code)
            uw = dict(T.UNIT_WORDS)
            for k, w in words.items():
                uw.setdefault(k, w)
            _set_attr(T, "UNIT_WORDS", uw)
        _patch_url(mod.Normalizer)
    else:
        while _saved:
            _saved.pop()()
    mod._engine.cache_clear()      # regexes are built from the tables
    _installed = on
'''),
    'mithi_tts.text_norm.rules': (False, r'''"""Targeted pre/post rules around the vendored normaliser (mithi_tts.text_norm.tn), derived from the eval rows that still
failed with text normalisation on.

The vendored copy stays byte-identical. Each rule rewrites one input shape that the strict normaliser either rejects
(it then falls back to reading the span character by character: "15-08-25" -> "one five minus zero eight minus two
five") or reads in a way listeners do not expect (a PIN code as "five lakh sixty thousand thirty eight"):

  R1 date_2y      DD-MM-YY, DD/MM/YY, DD.MM.YY            -> four-digit year (pivot: 00-50 -> 20xx, 51-99 -> 19xx),
                                                              then the normaliser's own date reading
  R2 expiry       MM/YY or MM/YYYY next to an expiry / validity / date word or a CVV -> "<month> <year>"
  R3 fy_range     2023-24 (second part = next year)       -> "<year> <two-digit number>"
  R4 digit_groups 4111 2233 4455 6677, 98765 43210, 1800 123 4567 (>= 10 digits in space-separated groups)
                                                           -> one digit run, which the normaliser reads digit by digit
  R5 pin_keyword  PIN / पिन / ... followed by 4-6 digits    -> digit by digit
  R6 pincode      a bare 6-digit number (no currency or unit next to it, no grouping commas) -> digit by digit
  R7 version      v4.5.2                                  -> "v four point five point two" (language's decimal word)
  R8 ta_date_year Tamil "ஆகஸ்ட் 15, 2025"                  -> "15 ஆகஸ்ட் 2025" (the normaliser keeps the year)
  R10 pct_range   3.5%-4%                                 -> 3.5-4% (the normaliser's own percent range)
  R11 per_price   ₹500/kg                                 -> "<amount> <per word> K G"
  R12 per_annum   8.5% p.a.                               -> "P A" (en: "per annum")
  R9 paise        rupee amounts with two decimals read "<n> rupees <m> paise" in every language (the tables of
                  bn en gu kn ml mr pa ta had no paise word, so they read "point five zero rupees")
  post:
  P1 tag_comma    a piece that ends in "," before an inline [tag] keeps its comma (the normaliser turns a trailing
                  comma into a full stop, and the model then ends the sentence at the tag)
  P2 uri_dash     "-" inside an e-mail / URL is spoken in the request language, not as the English word "dash"
  P3 uri_acronym  lowercase acronym labels in an e-mail / URL (sbi, kyc, hdfc in hdfcbank, sbi in onlinesbi) are
                  spelled letter by letter in the request language; other labels keep their English spelling
Language-table corrections (Assamese lexicon, "per" units, sub-domain hosts) live in mithi_tts.text_norm.lexicon.
"""
from __future__ import annotations

import re

__all__ = ["pre", "post", "install", "RULES"]

RULES = ("date_2y", "expiry", "fy_range", "digit_groups", "pin_keyword", "pincode", "version", "ta_date_year",
         "paise", "pct_range", "per_price", "per_annum", "tag_comma", "uri_dash", "uri_acronym")

PAISE = {"en": "paise", "bn": "পয়সা", "gu": "પૈસા", "kn": "ಪೈಸೆ", "ml": "പൈസ", "mr": "पैसे", "pa": "ਪੈਸੇ",
         "ta": "பைசா"}            # hi as or te already have one
DASH = {"hi": "डैश", "mr": "डॅश", "bn": "ড্যাশ", "as": "ডেশ্ব", "gu": "ડૅશ", "pa": "ਡੈਸ਼", "or": "ଡ୍ୟାସ୍",
        "ta": "டாஷ்", "te": "డాష్", "kn": "ಡ್ಯಾಶ್", "ml": "ഡാഷ്"}

# expiry / validity / date words (any language; matched case-insensitively in the 40 characters before the span)
_EXPIRY_WORDS = [
    "expiry", "expires", "expire", "exp", "valid", "validity", "thru", "expiration",
    "एक्सपायरी", "समाप्ति", "वैधता", "मुदत", "समाप्ती",
    "মেয়াদ", "মেয়াদ", "এক্সপায়ারি", "এক্সপায়ারি", "এক্সপায়াৰী", "ম্যাদ",
    "એક્સપાયરી", "સમાપ્તિ", "માન્યતા", "ਮਿਆਦ", "ਐਕਸਪਾਇਰੀ",
    "காலாவதி", "எக்ஸ்பைரி", "గడువు", "ఎక్స్‌పైరీ", "ఎక్స్పైరీ",
    "ಮುಕ್ತಾಯ", "ಅಂತಿಮ ದಿನಾಂಕ", "ಎಕ್ಸ್‌ಪೈರಿ", "കാലാവധി", "എക്സ്പയറി", "ମିଆଦ", "ଏକ୍ସପାୟାରୀ", "ସମାପ୍ତି",
]
_DATE_WORDS = ["date", "तारीख", "तिथि", "दिनांक", "डेट", "তারিখ", "তাৰিখ", "તારીખ", "ડેટ", "ਤਾਰੀਖ਼", "ਤਾਰੀਖ", "ਡੇਟ",
               "தேதி", "டேட்", "తేదీ", "డేట్", "ದಿನಾಂಕ", "ಡೇಟ್", "തീയതി", "ഡേറ്റ്", "ତାରିଖ", "ଡେଟ୍"]
_alt = lambda ws: "|".join(sorted(map(re.escape, ws), key=len, reverse=True))
_EXPIRY_RE = re.compile(_alt(_EXPIRY_WORDS), re.I)
_DATE_RE = re.compile(_alt(_DATE_WORDS), re.I)
_PIN_WORDS = ["pin", "पिन", "पिनकोड", "ಪಿನ್", "పిన్", "பின்", "പിൻ", "পিন", "ପିନ", "ਪਿੰਨ", "ਪਿਨ", "પિન"]
_PIN_RE = re.compile(r"(?<![\w])(?:" + "|".join(map(re.escape, _PIN_WORDS)) + r")(?:[\s-]*(?:code|कोड|কোড|કોડ|ਕੋਡ|କୋଡ|ಕೋಡ್|కోడ్|கோட்|കോഡ്))?"
                     r"(?:\s*(?:no\.?|number|नंबर|नम्बर))?\s*[:\-]?\s*(\d{4,6})(?![\d,]|\.\d)", re.I)
# words right after a number that make it an amount / quantity (R6 must not read those digit by digit)
_AMOUNT_AFTER = re.compile(
    r"\s*(?:rupees?|rs\.?|inr|dollars?|usd|paise|lakh|crore|thousand|people|users|customers|km|kg|kms|"
    r"रुपये|रुपए|रुपया|रु\.?|लाख|करोड़|हज़ार|हजार|लोग|লাখ|টাকা|টকা|কোটি|રૂપિયા|લાખ|ਰੁਪਏ|ਲੱਖ|ரூபாய்|ரூ\.?|லட்சம்|"
    r"రూపాయలు|రూ\.?|లక్షల|ರೂಪಾಯಿ|ರೂ\.?|ಲಕ್ಷ|രൂപ|ലക്ഷം|ଟଙ୍କା|ଲକ୍ଷ|ਕਰੋੜ|कोटी)(?![\w])", re.I)
_CURRENCY_BEFORE = re.compile(r"(?:₹|\$|€|£|\brs\.?|\binr|रु\.?|ரூ\.?|రూ\.?|ರೂ\.?)\s*$", re.I)
_NUMBER_WORD_BEFORE = re.compile(
    r"(?:\bno\.?|\bnumber|\bplot|\bflat|\bhouse|\bdoor|\bh\.\s?no\.?|नंबर|नम्बर|क्रमांक|नं\.?|নং|নম্বর|নম্বৰ|ਨੰਬਰ|નંબર|ନମ୍ବର|"
    r"எண்|నంబర్|ಸಂಖ್ಯೆ|ನಂಬರ್|നമ്പർ)\s*[:\-]?\s*$", re.I)
_TA_MONTHS = ["ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்", "ஜூலை", "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்",
              "நவம்பர்", "டிசம்பர்"]
_URI_RE = re.compile(r"https?://\S+|www\.\S+|\b[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}|\b[A-Za-z][\w.\-]*@[A-Za-z][A-Za-z0-9]{1,15}\b|\b[A-Za-z][\w\-]*(?:\.[A-Za-z0-9][\w\-]*)*\.(?:com|org|net|ai|io|co|dev|app|edu|gov|in)(?:\.(?:in|uk|us))?(?:[/?#][^\s<>]*)?")

# P3: lowercase acronym labels inside e-mail / URL hosts and paths are spelled letter by letter (sbi.co.in ->
# "S B I dot co dot in", hdfcbank.com -> "H D F C bank dot com"); other labels keep their English spelling.
_URI_ACRONYMS = {"sbi", "hdfc", "icici", "kyc", "irctc", "pnb", "rbi", "idbi", "npci", "uidai", "epfo", "mf", "lic",
                 "upi", "gst", "sms", "atm", "emi", "bob", "boi", "uco", "iob", "nsdl", "cdsl", "pnr", "otp", "ifsc"}
_URI_ACR_PREFIX = ("icici", "irctc", "uidai", "hdfc", "idbi", "npci", "epfo", "sbi", "pnb", "kyc", "rbi")
_URI_ACR_SUFFIX = ("sbi", "mf")

_SPELL_UNITS = {"kg", "km", "gm", "g", "ml", "ltr", "l"}   # R11: read letter by letter (as the normaliser reads "15 kg")

_installed = False


def install(on: bool = True):
    """R9: add the paise word to the language tables that lack it (bn en gu kn ml mr pa ta), so the normaliser's own
    main+sub-unit path reads ₹95.41 as "<n> rupees <m> paise" (with its native scale words and case suffixes).
    The tables are module-level; install(False) removes the added words again (used only by rules=False, the
    eval-parity mode; do not mix rules=False with concurrent rules=True calls)."""
    global _installed
    if on == _installed:
        return
    from mithi_tts.text_norm.tn.normalize import language_table
    for code, word in PAISE.items():
        sym = language_table(code).SYM
        if on:
            sym.setdefault("paisa", word)
        elif sym.get("paisa") == word:
            del sym["paisa"]
    _installed = on


def _table(lang):
    from mithi_tts.text_norm.tn.normalize import language_table
    return language_table(lang)


def _tn_id_context(L, before):
    """True when the normaliser itself will read the next number digit by digit (an ID keyword such as PIN / OTP /
    number in the three tokens before it); the rule then leaves the span alone so the normaliser's postposition
    sandhi ("...ಒಂದರಲ್ಲಿ") is kept."""
    from mithi_tts.text_norm.tn import normalize as _N   # the function; its module holds the keyword sets
    import sys
    mod = sys.modules["mithi_tts.text_norm.tn.normalize"]
    kws = mod._ID_KEYWORDS_EN | mod._OTP_INPUT_WORDS | {w.lower() for w in L.ID_KEYWORDS}
    toks = re.findall(r"[\w#\u0900-\u0DFF']+", before.replace("\u200c", "").lower())[-3:]
    return any(t in kws for t in toks)


def _digits(L, s):
    return ", ".join(L.digit(int(c)) for c in s)


def _year4(yy):
    y = int(yy)
    return 2000 + y if y <= 50 else 1900 + y


def _decimal_word(L):
    w = L.decimal(1, "2")
    a, b = L.cardinal(1), L.digit(2)
    return w[len(a):len(w) - len(b)].strip() if w.startswith(a) and w.endswith(b) else "point"


def pre(text: str, lang: str) -> str:
    """Apply R1-R8 outside e-mail / URL spans (those are identifiers, spelled by the normaliser)."""
    try:
        L = _table(lang)
    except Exception:
        return text
    out, pos = [], 0
    for m in _URI_RE.finditer(text):
        out.append(_pre(text[pos:m.start()], lang, L)); out.append(m.group(0)); pos = m.end()
    out.append(_pre(text[pos:], lang, L))
    return "".join(out)


def _pre(text, lang, L):
    t = text

    # R10 percent ranges written with the sign twice: 3.5%-4% (the normaliser read "...percentminus four percent")
    t = re.sub(r"(?<![\d.])(\d+(?:\.\d+)?)\s?%\s?[-–]\s?(\d+(?:\.\d+)?)\s?%", r"\1-\2%", t)

    # R11 price per unit: ₹500/kg (the "/" was dropped and "kg" left in Latin letters)
    def r11(m):
        unit, per = m.group(2), L.SYM.get("per") or "per"
        if lang != "en" and unit.lower() in _SPELL_UNITS:
            unit = " ".join(L.letter(c) for c in unit.upper())
        return f"{m.group(1)} {per} {unit}"
    t = re.sub(r"((?:[₹$€£]|\b(?:Rs\.?|INR)\s?)\s?\d[\d,]*(?:\.\d+)?)\s?/\s?(kg|km|gm|g|ml|ltr|l|day|month|night|year|person)\b", r11, t,
               flags=re.I)

    # R12 "p.a." (per annum) after a rate was left as the raw token "p.a."
    t = re.sub(r"(?<![\w.])[pP]\.\s?[aA]\.?(?![\w])", lambda m: "per annum" if lang == "en" else f"{L.letter('P')} {L.letter('A')}", t)

    # R7 version strings
    def r7(m):
        parts = m.group(1).split(".")
        dw = _decimal_word(L)
        return f"{L.letter('v')} " + f" {dw} ".join(L.cardinal(int(p)) for p in parts)
    t = re.sub(r"(?<![\w.])[vV](\d+(?:\.\d+){1,3})(?![\w]|\.\d)", r7, t)

    # R4 space-separated digit groups (card / phone numbers): join into one run (>= 10 digits in total)
    def r4(m):
        groups = m.group(0).split()
        return "".join(groups) if sum(len(g) for g in groups) >= 10 else m.group(0)
    t = re.sub(r"(?<![\d,.])(?:\d{4,5}(?: \d{3,5}){1,3}|\d{2,4} \d{3,4} \d{4})(?![\d,]|\.\d)", r4, t)

    # R1 dates with a two-digit year
    def r1(m):
        d, sep, mo, yy = m.group(1), m.group(2), m.group(3), m.group(4)
        if not (1 <= int(d) <= 31 and 1 <= int(mo) <= 12):
            return m.group(0)
        if sep == "-" and len(d) == 1 and len(mo) == 1:          # 8-2-45: a door number, not a date
            return m.group(0)
        if _NUMBER_WORD_BEFORE.search(t[max(0, m.start() - 16):m.start()]):   # "House No. 12-5-24", "ঘৰ নং ..."
            return m.group(0)
        return f"{d}{sep}{mo}{sep}{_year4(yy)}"
    t = re.sub(r"(?<![\d/.\-])(\d{1,2})([/.\-])(\d{1,2})\2(\d{2})(?![\d/\-]|\.\d)", r1, t)

    # R2 card expiry / month-year next to an expiry, date or CVV word
    def r2(m):
        mo, y = int(m.group(1)), m.group(2)
        before = t[max(0, m.start() - 40):m.start()]
        after = t[m.end():m.end() + 40]
        strong = _EXPIRY_RE.search(before) or re.search(r"\bcvv\b", before + " " + after, re.I)
        weak = _DATE_RE.search(before) and (len(y) == 4 or int(y) > 12)    # second part > 12: not DD/MM, so MM/YY
        if not (strong or weak):
            return m.group(0)
        year = int(y) if len(y) == 4 else 2000 + int(y)
        return f"{L.month(mo)} {L.year(year)}"
    t = re.sub(r"(?<![\d/.\-])(0?[1-9]|1[0-2])/(\d{2}|20\d{2})(?![\d/]|\.\d)", r2, t)

    # R3 financial-year ranges 2023-24
    def r3(m):
        y, yy = int(m.group(1)), int(m.group(2))
        if (y + 1) % 100 != yy:
            return m.group(0)
        return f"{L.year(y)} {L.cardinal(yy)}"
    t = re.sub(r"(?<![\d\-])((?:19|20)\d{2})[-–](\d{2})(?![\d\-])", r3, t)

    # R5 PIN keyword + digits
    def r5(m):
        if _tn_id_context(L, t[max(0, m.start(1) - 60):m.start(1)]):
            return m.group(0)
        return m.group(0)[:m.start(1) - m.start(0)] + _digits(L, m.group(1))
    t = _PIN_RE.sub(r5, t)

    # R6 bare six-digit numbers (Indian PIN codes): digit by digit unless an amount / quantity
    def r6(m):
        before = t[max(0, m.start() - 8):m.start()]
        if (_CURRENCY_BEFORE.search(before) or _AMOUNT_AFTER.match(t, m.end())
                or re.search(r"[A-Za-z0-9]\s?-\s?$|[#÷×*/+=<>]\s*$", before)       # ID suffix (CUST-567890), #ref, maths
                or re.match(r"\s*[÷×*/+=<>^]", t[m.end():])
                or _tn_id_context(L, re.split(r"[.!?;।॥\n]", t[max(0, m.start() - 60):m.start()])[-1])):
            return m.group(0)
        return _digits(L, m.group(0))
    t = re.sub(r"(?<![\d,.₹$#\w])[1-9]\d{5}(?![\d,]|\.\d|\s*%)", r6, t)

    # R8 Tamil "Month D, YYYY" -> "D Month YYYY"
    if lang == "ta":
        months = "|".join(_TA_MONTHS)
        t = re.sub(rf"(?<!\w)({months})\s+(\d{{1,2}}),?\s+(\d{{4}})(?!\d)", r"\2 \1 \3", t)
    return t


TAG_COMMA = False   # P1 off: no measured gain (6 vs 6 backstop cuts on the 14 affected emotion rows x2 seeds)


def post(out: str, seg: str, lang: str, before_tag: bool) -> str:
    # P1 keep a trailing comma that precedes an inline [tag] (disabled, see TAG_COMMA)
    if TAG_COMMA and before_tag and seg.rstrip().endswith(",") and out.rstrip().endswith("."):
        out = out.rstrip()[:-1] + ","
    # P2 spoken hyphen inside e-mail / URL in the request language
    uris = [m.group(0) for m in _URI_RE.finditer(seg)]
    if lang in DASH and uris:
        out = re.sub(r"(?<=\s)dash(?=\s)", DASH[lang], out)
    # P3 acronym labels inside e-mail / URL
    labels = {w for u in uris for w in re.findall(r"[a-z]+", re.sub(r"^https?://", "", u))
              if w in _URI_ACRONYMS or w.startswith(_URI_ACR_PREFIX) or w.endswith(_URI_ACR_SUFFIX)}
    if labels:
        try:
            L = _table(lang)
        except Exception:
            return out
        spell = lambda w: " ".join(L.letter(c.upper()) for c in w)

        def p3(m):
            w = m.group(0)
            if w in _URI_ACRONYMS:
                return spell(w)
            for a in _URI_ACR_PREFIX:
                if w.startswith(a) and len(w) - len(a) >= 3:
                    return spell(a) + " " + w[len(a):]
            for a in _URI_ACR_SUFFIX:
                if w.endswith(a) and len(w) - len(a) >= 4:
                    return w[:-len(a)] + " " + spell(a)
            return w
        out = re.sub(r"(?<![\w.@/])(?:" + "|".join(sorted(map(re.escape, labels), key=len, reverse=True)) + r")(?![\w]|[.@/\-]\w)",
                     p3, out)
    return out
'''),
    'mithi_tts.text_norm.tn': (True, r'''"""Vendored text normaliser (upstream commit fe5e6ec).

The modules in this directory are byte-identical copies of the source (see maya3/text/README.md); only this
__init__.py was added so they import as a package.  Use mithi_tts.text_norm.normalise_text rather than this package directly.
"""
from .normalize import NormalizationError, normalize, resolve_language  # noqa: F401
'''),
    'mithi_tts.text_norm.tn.assamese_numbers': (False, r'''"""Assamese number data from AI4Bharat/indic-numtowords (MIT).
Source: https://github.com/AI4Bharat/indic-numtowords/blob/master/indic_numtowords/asm/data/nums.py
License: assamese_numbers.LICENSE. Data only; the upstream parser is not used.
"""
ONES = ['শূন্য', 'এক', 'দুই', 'তিনি', 'চাৰি', 'পাঁচ', 'ছয়', 'সাত', 'আঠ', 'ন', 'দহ', 'এঘাৰ', 'বাৰ', 'তেৰ', 'চৈধ্য', 'পোন্ধৰ', 'ষোল্ল', 'সোতৰ', 'ওঠৰ', 'ঊনৈশ', 'বিশ', 'একৈশ', 'বাইশ', 'তেইশ', 'চৌবিশ', 'পঁচিশ', 'ছাব্বিশ', 'সাতাইশ', 'আঠাইশ', 'ঊনত্রিশ', 'ত্রিশ', 'একত্রিশ', 'বত্রিশ', 'তেত্রিশ', 'চৌত্রিশ', 'পঁয়ত্রিশ', 'ছয়ত্রিশ', 'সাতত্রিশ', 'আঠত্রিশ', 'ঊনচল্লিশ', 'চল্লিশ', 'একচল্লিশ', 'বিয়াল্লিশ', 'তিয়াল্লিশ', 'চৌচল্লিশ', 'পঁঞ্চল্লিশ', 'ছয়চল্লিশ', 'সাতচল্লিশ', 'আঠচল্লিশ', 'ঊনপঞ্চাশ', 'পঞ্চাশ', 'একাৱন', 'বাৱন', 'তেপন', 'চৌৱন', 'পঁচপন', 'ছাপন', 'সাতাৱন', 'আঠাৱন', 'ঊনষাঠি', 'ষাঠি', 'একষষ্ঠি', 'বাষষ্ঠি', 'তেষষ্ঠি', 'চৌষষ্ঠি', 'পঁয়ষষ্ঠি', 'ছয়ষষ্ঠি', 'সাতষষ্ঠি', 'আঠষষ্ঠি', 'ঊনসত্তৰ', 'সত্তৰ', 'এসত্তৰ', 'বাসত্তৰ', 'তেসত্তৰ', 'চৌসত্তৰ', 'পঁয়সত্তৰ', 'ছয়সত্তৰ', 'সাতসত্তৰ', 'আঠসত্তৰ', 'ঊনাশী', 'আশী', 'একাশী', 'বিৰাশী', 'তিৰাশী', 'চৌৰাশী', 'পঁচাশী', 'ছিয়াশী', 'সাতাশী', 'আঠাশী', 'ঊননব্বৈ', 'নব্বৈ', 'একানব্বৈ', 'বিৰানব্বৈ', 'তিৰানব্বৈ', 'চৌৰানব্বৈ', 'পঁচানব্বৈ', 'ছিয়ানব্বৈ', 'সাতানব্বৈ', 'আঠানব্বৈ', 'নিৰানব্বৈ']
HUNDREDS = ['', 'এশ', 'দুশ', 'তিনিশ', 'চাৰিশ', 'পাঁচশ', 'ছয়শ', 'সাতশ', 'আঠশ', 'নশ']
'''),
    'mithi_tts.text_norm.tn.identifier_rules': (False, r'''"""Opt-in identifier rendering; no model calls or prose transliteration.

Uppercase dictionary components prescribe character spelling. In particular,
AXIS is spelled as an identifier rather than guessed as the word 'access'.
Ordinary words stay words; existing locale letter/symbol tables supply sounds.
"""
import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def vocabulary():
    return json.loads(Path(__file__).with_name("identifier_words.json").read_text(encoding="utf-8"))


def spell_uri(part, lang, spell_digits):
    data = vocabulary()

    def word(token, expand=True):
        if expand and token.lower() in data["brands"]:
            return " ".join(word(p, False) for p in data["brands"][token.lower()])
        if (len(token) == 1 or token.isupper() or token.upper() in data["acronyms"]
                or token.upper() in lang.ABBREV_UNITS):
            return " ".join(lang.letter(c) for c in token)
        pieces = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[A-Z]+", token)
        if len(pieces) > 1:
            return " ".join(word(p, False) for p in pieces)
        return token

    symbols = {"@": "at", ".": "dot", "_": "underscore", "-": "minus",
               "/": "slash", "+": "plus", "=": "equals", "&": "amp",
               "#": "hash", "%": "percent"}
    out = []
    # Retain non-ASCII runs intact, including Indic combining marks.
    for token in re.findall(r"[A-Za-z]+|[0-9]+|[@._/\-+?=&#%:~]|[^A-Za-z0-9@._/\-+?=&#%:~]+", part):
        if token.isascii() and token.isalpha():
            out.append(word(token))
        elif token.isdecimal():
            out.append(spell_digits(token, comma=False))
        elif token in symbols:
            out.append(lang.SYM[symbols[token]])
        else:
            out.append({"?": "question mark", ":": "colon", "~": "tilde"}.get(token, token))
    return " ".join(p for p in out if p.strip())


def group_digits(digits, lang, sizes=None):
    """Spaces inside groups, comma pause between groups; every digit survives."""
    if sizes is None:
        if len(digits) == 16:
            sizes = [4] * 4
        elif len(digits) == 10:
            sizes = [3, 3, 4]
        else:
            sizes = [4] * (len(digits) // 4) + [len(digits) % 4]
    out, start = [], 0
    for size in sizes:
        if size:
            out.append(" ".join(lang.digit(int(c)) for c in digits[start:start + size]))
            start += size
    assert start == len(digits)
    return ", ".join(out)


# These shapes identify card/telephone numbers even without English keywords.
# Dates and currency retain higher matcher priority in the normalizer.
LONG_RUN = re.compile(
    r"(?<![A-Za-z0-9.,])(?:\+\d{1,3}[ -]+|91[ -]+)?"
    r"(?:\d{4}(?:[ -]\d{4}){3}|\d{5}[ -]\d{5}|\d{3}[ -]\d{3}[ -]\d{4}|\d{4}[ -]\d{7}|\d{8,})"
    r"(?![A-Za-z0-9]|[.,]\d)"
)


def long_runs(text, lang):
    for match in LONG_RUN.finditer(text):
        token = match.group().lstrip("+")
        groups = re.split(r"[ -]", token)
        digits = "".join(groups)
        sizes = [len(g) for g in groups] if len(groups) > 1 else None
        spoken = group_digits(digits, lang, sizes)
        if match.group().startswith("+"):
            spoken = lang.SYM["plus"] + " " + spoken
        yield match.start(), match.end(), spoken
'''),
    'mithi_tts.text_norm.tn.indic_languages': (False, r'''"""Native language tables recovered from MayaTeam/enterprise.

Source: 7384e91b02848ed791060c2eb368d52c9b92203c:gpu-origin/normalize_all.py
Only LangBase subclasses were extracted; the legacy parser was NOT copied.
The review branch also repairs singular units and Hindi irregular ordinals, and
adds explicit Hindi/Telugu month-spelling aliases for calendar validation.
These are inherited candidate lexicons, not independently native-speaker approved.
See docs/NORMALIZATION_REVIEW.md for provenance, limitations and release gates.
"""
if __package__:
    from .normalize import LangBase
else:
    from normalize import LangBase


class Hindi(LangBase):
    code = "hi"
    name = "Hindi"
    native_digits = "०१२३४५६७८९"
    SINGULAR_NOUNS = {"घंटे": "घंटा", "रुपये": "रुपया", "पैसे": "पैसा"}

    ONES = [
        "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
        "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
        "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
        "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
        "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
        "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
        "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
        "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी",
        "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अठासी", "नवासी",
        "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पंचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
    ]
    HUNDRED = "सौ"
    THOUSAND = "हज़ार"
    LAKH = "लाख"
    CRORE = "करोड़"

    LETTERS = {
        "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ़",
        "G": "जी", "H": "एच", "I": "आई", "J": "जे", "K": "के", "L": "एल",
        "M": "एम", "N": "एन", "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर",
        "S": "एस", "T": "टी", "U": "यू", "V": "वी", "W": "डब्ल्यू",
        "X": "एक्स", "Y": "वाई", "Z": "ज़ेड",
    }

    MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
              "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
    MONTH_ALIASES = {"फरवरी": 2, "सितम्बर": 9, "नवम्बर": 11, "दिसम्बर": 12}

    SYM = {
        "rupee": "रुपये", "dollar": "डॉलर", "euro": "यूरो", "pound": "पाउंड",
        "cent": "सेंट", "paisa": "पैसे",
        "percent": "प्रतिशत",
        "point": "दशमलव",
        "minus": "माइनस", "plus": "प्लस", "equals": "बराबर",
        "times": "गुणा", "divide": "भाग",
        "squared": "का वर्ग", "sqrt": "वर्गमूल",
        "lt": "से कम", "gt": "से ज़्यादा",
        "plus_minus": "धन-ऋण",
        "at": "एट द रेट", "underscore": "अंडरस्कोर", "dot": "डॉट",
        "slash": "स्लैश", "hash": "हैश", "amp": "और",
        "half": "आधा", "quarter": "चौथाई", "three_quarters": "तीन चौथाई",
        "degree": "डिग्री", "celsius": "डिग्री सेल्सियस", "fahrenheit": "डिग्री फ़ारेनहाइट",
        "minute": "मिनट", "second": "सेकंड", "per": "प्रति",
    }

    TIME_ON_HOUR = "{h} बजे"
    TIME_HM = "{h} बजकर {m} मिनट"
    RANGE = "{a} से {b}"
    RANGE_TO = "{a} से {b} तक"
    RATIO = "{a} अनुपात {b}"
    POWER = "{base} की घात {exp}"
    FRACTION = "{num} बटा {den}"
    DECADE_SHORT = "{tens} के"
    DECADE_FULL = "{tens} का दशक"

    PERIODS = {"morning": "सुबह", "afternoon": "दोपहर", "evening": "शाम", "night": "रात"}

    TIME_MARKERS = ["बजे", "बजकर"]
    PERIOD_DETECT = {"सुबह": "morning", "दोपहर": "afternoon", "शाम": "evening",
                     "रात": "night", "सवेरे": "morning"}
    YEAR_WORDS = ["साल", "वर्ष", "सन्", "सन"]
    ID_KEYWORDS = ["नंबर", "नम्बर", "खाता", "खाते", "कोड", "ऑर्डर", "फ़ोन", "फोन", "मोबाइल"]
    LABEL_WORDS = ["अनुच्छेद", "धारा", "कमरा", "कक्षा", "वार्ड", "अध्याय", "भाग"]
    ID_AFTER_WORDS = ["खत्म", "डायल", "कॉल", "संपर्क"]
    NUMBER_WORD = "नंबर"
    ABBREV_UNITS = {"MINS": "मिनट", "HRS": "घंटे", "SECS": "सेकंड",
                    "KMS": "किलोमीटर"}
    MINUTE_WORDS = ["मिनट"]
    SCALE_WORDS = {"करोड़": "crore", "लाख": "lakh", "हज़ार": "thousand", "हजार": "thousand",
                   "अरब": "arab"}
    CURRENCY_WORDS = {"रुपये": "rupee", "रुपए": "rupee", "रुपया": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "पहला", 2: "दूसरा", 3: "तीसरा", 4: "चौथा", 6: "छठा"}
    ORDINAL_SUFFIX = "वाँ"
    ORDINAL_INPUT_SUFFIXES = ["वीं", "वाँ", "वां", "वें", "वे"]

    def _ordinal_compose(self, n, src_suffix):
        # 12वीं -> बारहवीं : cardinal + preserved suffix
        if src_suffix:
            if n in self.ORDINAL_EXCEPTIONS:
                # Irregular stems retain gender/case: 1वीं -> पहली, not एकवीं.
                ending = {"वीं": "ी", "वें": "े", "वे": "े"}.get(src_suffix, "ा")
                return self.ORDINAL_EXCEPTIONS[n][:-1] + ending
            return f"{self.cardinal(n)}{src_suffix}"
        if n in self.ORDINAL_EXCEPTIONS:
            return self.ORDINAL_EXCEPTIONS[n]
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"


class Bengali(LangBase):
    code = "bn"
    name = "Bengali"
    native_digits = "০১২৩৪৫৬৭৮৯"

    ONES = [
        "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়",
        "দশ", "এগারো", "বারো", "তেরো", "চৌদ্দ", "পনেরো", "ষোলো", "সতেরো", "আঠারো", "উনিশ",
        "বিশ", "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ", "ছাব্বিশ", "সাতাশ", "আঠাশ", "ঊনত্রিশ",
        "ত্রিশ", "একত্রিশ", "বত্রিশ", "তেত্রিশ", "চৌত্রিশ", "পঁয়ত্রিশ", "ছত্রিশ", "সাঁইত্রিশ", "আটত্রিশ", "ঊনচল্লিশ",
        "চল্লিশ", "একচল্লিশ", "বিয়াল্লিশ", "তেতাল্লিশ", "চুয়াল্লিশ", "পঁয়তাল্লিশ", "ছেচল্লিশ", "সাতচল্লিশ", "আটচল্লিশ", "ঊনপঞ্চাশ",
        "পঞ্চাশ", "একান্ন", "বায়ান্ন", "তিপ্পান্ন", "চুয়ান্ন", "পঞ্চান্ন", "ছাপ্পান্ন", "সাতান্ন", "আটান্ন", "ঊনষাট",
        "ষাট", "একষট্টি", "বাষট্টি", "তেষট্টি", "চৌষট্টি", "পঁয়ষট্টি", "ছেষট্টি", "সাতষট্টি", "আটষট্টি", "ঊনসত্তর",
        "সত্তর", "একাত্তর", "বাহাত্তর", "তিয়াত্তর", "চুয়াত্তর", "পঁচাত্তর", "ছিয়াত্তর", "সাতাত্তর", "আটাত্তর", "ঊনআশি",
        "আশি", "একাশি", "বিরাশি", "তিরাশি", "চুরাশি", "পঁচাশি", "ছিয়াশি", "সাতাশি", "আটাশি", "ঊননব্বই",
        "নব্বই", "একানব্বই", "বিরানব্বই", "তিরানব্বই", "চুরানব্বই", "পঁচানব্বই", "ছিয়ানব্বই", "সাতানব্বই", "আটানব্বই", "নিরানব্বই",
    ]
    HUNDRED = "শো"
    THOUSAND = "হাজার"
    LAKH = "লাখ"
    CRORE = "কোটি"

    LETTERS = {
        "A": "এ", "B": "বি", "C": "সি", "D": "ডি", "E": "ই", "F": "এফ",
        "G": "জি", "H": "এইচ", "I": "আই", "J": "জে", "K": "কে", "L": "এল",
        "M": "এম", "N": "এন", "O": "ও", "P": "পি", "Q": "কিউ", "R": "আর",
        "S": "এস", "T": "টি", "U": "ইউ", "V": "ভি", "W": "ডব্লিউ",
        "X": "এক্স", "Y": "ওয়াই", "Z": "জেড",
    }

    MONTHS = ["জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
              "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"]

    SYM = {
        "rupee": "টাকা", "dollar": "ডলার", "euro": "ইউরো", "pound": "পাউন্ড",
        "cent": "সেন্ট",
        "percent": "শতাংশ",
        "point": "দশমিক",
        "minus": "মাইনাস", "plus": "প্লাস", "equals": "সমান",
        "times": "গুণ", "divide": "ভাগ",
        "squared": "এর বর্গ", "sqrt": "বর্গমূল",
        "lt": "এর চেয়ে কম", "gt": "এর চেয়ে বেশি",
        "plus_minus": "প্লাস-মাইনাস",
        "at": "অ্যাট দ্য রেট", "underscore": "আন্ডারস্কোর", "dot": "ডট",
        "slash": "স্ল্যাশ", "hash": "হ্যাশ", "amp": "অ্যান্ড",
        "half": "অর্ধেক", "quarter": "এক চতুর্থাংশ", "three_quarters": "তিন চতুর্থাংশ",
        "degree": "ডিগ্রি", "celsius": "ডিগ্রি সেলসিয়াস", "fahrenheit": "ডিগ্রি ফারেনহাইট",
        "minute": "মিনিট", "second": "সেকেন্ড",
    }

    TIME_ON_HOUR = "{h}টা"
    TIME_HM = "{h}টা বেজে {m} মিনিট"
    RANGE = "{a} থেকে {b}"
    RANGE_TO = "{a} থেকে {b} পর্যন্ত"
    RATIO = "{a} অনুপাত {b}"
    POWER = "{base} এর ঘাত {exp}"
    FRACTION = "{num} বাই {den}"
    DECADE_SHORT = "{tens} এর দশক"
    DECADE_FULL = "{tens} এর দশক"

    PERIODS = {"morning": "সকাল", "afternoon": "দুপুর",
               "evening": "সন্ধ্যা", "night": "রাত"}

    TIME_MARKERS = ["টা", "টায়", "টার", "বেজে", "বাজে", "বাজবে"]
    PERIOD_DETECT = {"সকাল": "morning", "সকালে": "morning",
                     "ভোর": "morning", "ভোরে": "morning",
                     "দুপুর": "afternoon", "দুপুরে": "afternoon",
                     "বিকেল": "evening", "বিকেলে": "evening", "বিকাল": "evening",
                     "সন্ধ্যা": "evening", "সন্ধ্যায়": "evening",
                     "রাত": "night", "রাতে": "night"}
    YEAR_WORDS = ["সাল", "সালে", "সালের", "সন", "বছর", "বছরে"]
    ID_KEYWORDS = ["নম্বর", "নাম্বার", "অ্যাকাউন্ট", "একাউন্ট", "কোড", "অর্ডার", "ফোন", "মোবাইল", "ওটিপি", "পিন", "আইডি", "পাসওয়ার্ড", "ট্রেন", "এক্সপ্রেস", "পিএনআর", "কার্ড", "রেফারেন্স", "ট্রানজেকশন", "টিকিট", "অ্যাকাউন্টের", "নম্বরে", "আইডিতে", "ইউজার", "ইনস্টলমেন্ট", "কিস্তি", "ডিজিট", "অঙ্ক", "লেনদেন"]
    ID_AFTER_WORDS = ["ডায়াল", "কল", "যোগাযোগ"]
    NUMBER_WORD = "নম্বর"
    ABBREV_UNITS = {"MINS":"মিনিট", "MIN":"মিনিট", "HRS":"ঘণ্টা", "HR":"ঘণ্টা", "SECS":"সেকেন্ড", "SEC":"সেকেন্ড", "KMS":"কিলোমিটার"}
    LABEL_WORDS = ["অনুচ্ছেদ", "ধারা", "রুম", "ঘর", "কক্ষ", "ওয়ার্ড",
                   "অধ্যায়", "গেট", "কাউন্টার"]
    MINUTE_WORDS = ["মিনিট"]
    SCALE_WORDS = {"কোটি": "crore", "লাখ": "lakh", "লক্ষ": "lakh",
                   "হাজার": "thousand"}
    CURRENCY_WORDS = {"টাকা": "rupee", "টাকার": "rupee", "টাকায়": "rupee",
                      "রুপি": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "প্রথম", 2: "দ্বিতীয়", 3: "তৃতীয়", 4: "চতুর্থ",
                          5: "পঞ্চম", 6: "ষষ্ঠ", 7: "সপ্তম", 8: "অষ্টম",
                          9: "নবম", 10: "দশম"}
    ORDINAL_SUFFIX = "তম"
    ORDINAL_INPUT_SUFFIXES = ["তম", "লা", "রা", "ঠা", "ই", "শে"]

    # Bengali date-of-month ordinals for ১লা / ২রা / ৩রা / ৪ঠা
    _DATE_ORDINALS = {1: "পহেলা", 2: "দোসরা", 3: "তেসরা", 4: "চৌঠা"}

    # ------------------------------------------------------------------
    # Hundreds fuse: এক+শো -> একশো, চার+শো -> চারশো (written solid).
    # ------------------------------------------------------------------
    def hundreds(self, h, rest):
        out = f"{self.ONES[h]}{self.HUNDRED}"
        if rest:
            out += f" {self.two_digit(rest)}"
        return out

    def year(self, y):
        """1947 -> উনিশশো সাতচল্লিশ (fused শো); 2026 -> দুই হাজার ছাব্বিশ."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            out = f"{self.two_digit(hi)}{self.HUNDRED}"
            if lo:
                out += f" {self.two_digit(lo)}"
            return out
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # Date-style ordinals: ১লা -> পহেলা, ২রা -> দোসরা, ৪ঠা -> চৌঠা,
        # ৫ই -> পাঁচই, ১৯শে -> উনিশে. General: ১২তম -> বারোতম.
        if src_suffix in ("লা", "রা", "ঠা") and n in self._DATE_ORDINALS:
            return self._DATE_ORDINALS[n]
        if src_suffix == "ই":
            return f"{self.cardinal(n)}ই"
        if src_suffix == "শে":
            return f"{self.cardinal(n)}ে"
        if src_suffix == "তম":
            return f"{self.cardinal(n)}তম"
        if n in self.ORDINAL_EXCEPTIONS:
            return self.ORDINAL_EXCEPTIONS[n]
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"

    # postpositions attach to noun-words with Bengali sandhi
    # (টাকা+এর->টাকার, শতাংশ+এ->শতাংশে, টা+মধ্যে->টার মধ্যে)
    _ATTACH_NOUNS = {"টাকা", "পয়সা", "ডলার", "শতাংশ", "কিলোমিটার",
                     "কিলোগ্রাম", "মিটার", "লিটার", "গ্রাম", "ডিগ্রি",
                     "মিনিট", "সেকেন্ড", "টা", "টো"}

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if last not in self._ATTACH_NOUNS:
            return None
        if t == last:
            return rep                          # duplicate marker
        if t in ("এ", "তে", "য়"):               # locative
            if last.endswith(("া","ো","ে")):
                return rep + "য়"                # টাকায়, এগারোটায়
            return rep + "ে"                     # শতাংশে, মিনিটে
        if t in ("এর", "র"):                     # genitive
            if last.endswith(("া","ো","ে")):
                return rep + "র"                 # টাকার, টার
            return rep + "ের"                    # মিনিটের
        if t in ("মধ্যে", "নাগাদ", "থেকে", "পর্যন্ত", "দিয়ে"):
            g = rep + ("র" if last.endswith(("া","ো","ে")) else "ের")
            return g + " " + t                   # এগারোটার মধ্যে
        return None


class Gujarati(LangBase):
    code = "gu"
    name = "Gujarati"
    native_digits = "૦૧૨૩૪૫૬૭૮૯"

    ONES = [
        "શૂન્ય", "એક", "બે", "ત્રણ", "ચાર", "પાંચ", "છ", "સાત", "આઠ", "નવ",
        "દસ", "અગિયાર", "બાર", "તેર", "ચૌદ", "પંદર", "સોળ", "સત્તર", "અઢાર", "ઓગણીસ",
        "વીસ", "એકવીસ", "બાવીસ", "ત્રેવીસ", "ચોવીસ", "પચ્ચીસ", "છવ્વીસ", "સત્તાવીસ", "અઠ્ઠાવીસ", "ઓગણત્રીસ",
        "ત્રીસ", "એકત્રીસ", "બત્રીસ", "તેત્રીસ", "ચોત્રીસ", "પાંત્રીસ", "છત્રીસ", "સાડત્રીસ", "આડત્રીસ", "ઓગણચાળીસ",
        "ચાળીસ", "એકતાળીસ", "બેતાળીસ", "તેંતાળીસ", "ચુંમાળીસ", "પિસ્તાળીસ", "છેંતાળીસ", "સુડતાળીસ", "અડતાળીસ", "ઓગણપચાસ",
        "પચાસ", "એકાવન", "બાવન", "ત્રેપન", "ચોપન", "પંચાવન", "છપ્પન", "સત્તાવન", "અઠ્ઠાવન", "ઓગણસાઠ",
        "સાઠ", "એકસઠ", "બાસઠ", "ત્રેસઠ", "ચોસઠ", "પાંસઠ", "છાસઠ", "સડસઠ", "અડસઠ", "અગણોસિત્તેર",
        "સિત્તેર", "એકોતેર", "બોતેર", "તોંતેર", "ચુંમોતેર", "પંચોતેર", "છોતેર", "સિત્તોતેર", "અઠ્ઠોતેર", "ઓગણએંસી",
        "એંસી", "એક્યાસી", "બ્યાસી", "ત્યાસી", "ચોરાસી", "પંચાસી", "છયાસી", "સત્યાસી", "અઠયાસી", "નેવ્યાસી",
        "નેવું", "એકાણું", "બાણું", "ત્રાણું", "ચોરાણું", "પંચાણું", "છન્નું", "સત્તાણું", "અઠ્ઠાણું", "નવ્વાણું",
    ]
    HUNDRED = "સો"
    THOUSAND = "હજાર"
    LAKH = "લાખ"
    CRORE = "કરોડ"

    # Gujarati hundreds are fused compounds (200 = બસો, not "બે સો").
    HUNDREDS_FUSED = {
        1: "એકસો", 2: "બસો", 3: "ત્રણસો", 4: "ચારસો", 5: "પાંચસો",
        6: "છસો", 7: "સાતસો", 8: "આઠસો", 9: "નવસો",
    }

    LETTERS = {
        "A": "એ", "B": "બી", "C": "સી", "D": "ડી", "E": "ઈ", "F": "એફ",
        "G": "જી", "H": "એચ", "I": "આઈ", "J": "જે", "K": "કે", "L": "એલ",
        "M": "એમ", "N": "એન", "O": "ઓ", "P": "પી", "Q": "ક્યુ", "R": "આર",
        "S": "એસ", "T": "ટી", "U": "યુ", "V": "વી", "W": "ડબલ્યુ",
        "X": "એક્સ", "Y": "વાય", "Z": "ઝેડ",
    }

    MONTHS = ["જાન્યુઆરી", "ફેબ્રુઆરી", "માર્ચ", "એપ્રિલ", "મે", "જૂન",
              "જુલાઈ", "ઓગસ્ટ", "સપ્ટેમ્બર", "ઓક્ટોબર", "નવેમ્બર", "ડિસેમ્બર"]

    SYM = {
        "rupee": "રૂપિયા", "dollar": "ડોલર", "euro": "યુરો", "pound": "પાઉન્ડ",
        "cent": "સેન્ટ",
        "percent": "ટકા",
        "point": "દશાંશ",
        "minus": "માઇનસ", "plus": "પ્લસ", "equals": "બરાબર",
        "times": "ગુણ્યા", "divide": "ભાગ્યા",
        "squared": "નો વર્ગ", "sqrt": "વર્ગમૂળ",
        "lt": "થી ઓછું", "gt": "થી વધારે",
        "plus_minus": "વત્તા-ઓછા",
        "at": "એટ ધ રેટ", "underscore": "અન્ડરસ્કોર", "dot": "ડોટ",
        "slash": "સ્લેશ", "hash": "હેશ", "amp": "અને",
        "half": "અડધો", "quarter": "પા", "three_quarters": "પોણો",
        "degree": "ડિગ્રી", "celsius": "ડિગ્રી સેલ્સિયસ", "fahrenheit": "ડિગ્રી ફેરનહીટ",
        "minute": "મિનિટ", "second": "સેકન્ડ",
    }

    TIME_ON_HOUR = "{h} વાગ્યે"
    TIME_HM = "{h} વાગીને {m} મિનિટ"
    RANGE = "{a} થી {b}"
    RANGE_TO = "{a} થી {b} સુધી"
    RATIO = "{a} ગુણોત્તર {b}"
    POWER = "{base} ની ઘાત {exp}"
    FRACTION = "{num} બટા {den}"
    DECADE_SHORT = "{tens} ના"
    DECADE_FULL = "{tens} નો દાયકો"

    PERIODS = {"morning": "સવારે", "afternoon": "બપોરે",
               "evening": "સાંજે", "night": "રાત્રે"}

    TIME_MARKERS = ["વાગ્યે", "વાગે", "વાગ્યા", "વાગ્યું", "વાગીને", "કલાકે"]
    PERIOD_DETECT = {"સવારે": "morning", "સવાર": "morning", "સવારના": "morning",
                     "બપોરે": "afternoon", "બપોર": "afternoon", "બપોરના": "afternoon",
                     "સાંજે": "evening", "સાંજ": "evening", "સાંજના": "evening",
                     "રાત્રે": "night", "રાતે": "night", "રાત": "night", "રાતના": "night"}
    YEAR_WORDS = ["વર્ષ", "વર્ષે", "સાલ", "સન", "સને"]
    ID_KEYWORDS = ["નંબર", "નમ્બર", "ખાતા", "ખાતું", "ખાતાં", "કોડ", "ઓર્ડર", "ફોન", "મોબાઇલ", "મોબાઈલ", "એકાઉન્ટ", "એકાઉન્ટમાં", "એકાઉન્ટનો", "પિન", "આઈડી", "પાસવર્ડ", "ટ્રેન", "એક્સપ્રેસ", "પીએનઆર", "કાર્ડ", "રેફરન્સ", "ટ્રાન્ઝેક્શન", "ટિકિટ", "ઓટીપી", "નંબરમાં", "યુઝર", "હપ્તો", "અંક", "ડિજિટ"]
    ID_AFTER_WORDS = ["ડાયલ", "કૉલ", "સંપર્ક"]
    NUMBER_WORD = "નંબર"
    ABBREV_UNITS = {"MINS":"મિનિટ", "MIN":"મિનિટ", "HRS":"કલાક", "HR":"કલાક", "SECS":"સેકન્ડ", "SEC":"સેકન્ડ", "KMS":"કિલોમીટર"}
    LABEL_WORDS = ["અનુચ્છેદ", "કલમ", "ઓરડો", "રૂમ", "વોર્ડ", "ધોરણ",
                   "પ્રકરણ", "ભાગ", "વર્ગ"]
    MINUTE_WORDS = ["મિનિટ", "મિનીટ"]
    SCALE_WORDS = {"કરોડ": "crore", "લાખ": "lakh", "હજાર": "thousand",
                   "અબજ": "arab"}
    CURRENCY_WORDS = {"રૂપિયા": "rupee", "રૂપિયો": "rupee", "રૂપિયે": "rupee",
                      "રૂ": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "પહેલો", 2: "બીજો", 3: "ત્રીજો", 4: "ચોથો", 6: "છઠ્ઠો"}
    # Stems for gender/case agreement with a source suffix (1લી -> પહેલી).
    ORDINAL_STEMS = {1: "પહેલ", 2: "બીજ", 3: "ત્રીજ", 4: "ચોથ", 6: "છઠ્ઠ"}
    ORDINAL_SUFFIX = "મો"
    ORDINAL_INPUT_SUFFIXES = ["મી", "મો", "મું", "મા", "મે",
                              "લી", "લો", "લું", "લા",
                              "જી", "જો", "જું", "જા",
                              "થી", "થો", "થું", "થા"]

    # ------------------------------------------------------------------
    # Gujarati-specific composition
    # ------------------------------------------------------------------

    def hundreds(self, h, rest):
        """Fused hundreds: 250 -> 'બસો પચાસ', 147 -> 'એકસો સુડતાળીસ'."""
        out = self.HUNDREDS_FUSED[h]
        if rest:
            out += f" {self.two_digit(rest)}"
        return out

    def year(self, y):
        """1100-1999: fused hundred style — 1947 -> 'ઓગણીસો સુડતાળીસ',
        1857 -> 'અઢારસો સત્તાવન'. A trailing સ merges with સો
        (ઓગણીસ + સો -> ઓગણીસો). Everything else: full cardinal."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            w = self.two_digit(hi)
            w = w[:-1] + self.HUNDRED if w.endswith("સ") else w + self.HUNDRED
            if lo:
                return f"{w} {self.two_digit(lo)}"
            return w
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # 1-4, 6 have suppletive stems: 1લી -> પહેલી, 2જી -> બીજી, 4થા -> ચોથા.
        if n in self.ORDINAL_STEMS:
            if src_suffix and len(src_suffix) > 1:
                return self.ORDINAL_STEMS[n] + src_suffix[1:]
            return self.ORDINAL_EXCEPTIONS[n]
        # Regular: cardinal + મ-suffix (12મી -> બારમી, 5મો -> પાંચમો).
        if src_suffix and src_suffix.startswith("મ"):
            return f"{self.cardinal(n)}{src_suffix}"
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"

    # Gujarati postpositions agglutinate directly onto the preceding word
    # (રૂપિયા + માં -> રૂપિયામાં). Oblique of the plural currency word before
    # a genitive: રૂપિયા stays (already the citation form here).
    _PP = {"માં", "માંથી", "નું", "નો", "ની", "ના", "ને", "થી", "સુધી",
           "પર", "પરથી", "વડે", "કરતાં", "જેટલા", "જેટલું"}

    # attach postpositions only to noun-words we insert (currency/unit/
    # percent), never to a bare cardinal (Terminal બે + પર stays "બે પર")
    _ATTACH_NOUNS = {"રૂપિયા", "પૈસા", "ડોલર", "ટકા", "ટકાવારી",
                     "કિલોમીટર", "કિલોગ્રામ", "મીટર", "લિટર", "ગ્રામ",
                     "ડિગ્રી", "કલાક", "મિનિટ", "સેકન્ડ"}

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if t == last and last in self._ATTACH_NOUNS:
            return rep
        if t in self._PP and last in self._ATTACH_NOUNS:
            return rep + t
        return None


class Kannada(LangBase):
    code = "kn"
    name = "Kannada"
    native_digits = "೦೧೨೩೪೫೬೭೮೯"

    ONES = [
        "ಸೊನ್ನೆ", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು",
        "ಹತ್ತು", "ಹನ್ನೊಂದು", "ಹನ್ನೆರಡು", "ಹದಿಮೂರು", "ಹದಿನಾಲ್ಕು", "ಹದಿನೈದು", "ಹದಿನಾರು", "ಹದಿನೇಳು", "ಹದಿನೆಂಟು", "ಹತ್ತೊಂಬತ್ತು",
        "ಇಪ್ಪತ್ತು", "ಇಪ್ಪತ್ತೊಂದು", "ಇಪ್ಪತ್ತೆರಡು", "ಇಪ್ಪತ್ತಮೂರು", "ಇಪ್ಪತ್ತನಾಲ್ಕು", "ಇಪ್ಪತ್ತೈದು", "ಇಪ್ಪತ್ತಾರು", "ಇಪ್ಪತ್ತೇಳು", "ಇಪ್ಪತ್ತೆಂಟು", "ಇಪ್ಪತ್ತೊಂಬತ್ತು",
        "ಮೂವತ್ತು", "ಮೂವತ್ತೊಂದು", "ಮೂವತ್ತೆರಡು", "ಮೂವತ್ತಮೂರು", "ಮೂವತ್ತನಾಲ್ಕು", "ಮೂವತ್ತೈದು", "ಮೂವತ್ತಾರು", "ಮೂವತ್ತೇಳು", "ಮೂವತ್ತೆಂಟು", "ಮೂವತ್ತೊಂಬತ್ತು",
        "ನಲವತ್ತು", "ನಲವತ್ತೊಂದು", "ನಲವತ್ತೆರಡು", "ನಲವತ್ತಮೂರು", "ನಲವತ್ತನಾಲ್ಕು", "ನಲವತ್ತೈದು", "ನಲವತ್ತಾರು", "ನಲವತ್ತೇಳು", "ನಲವತ್ತೆಂಟು", "ನಲವತ್ತೊಂಬತ್ತು",
        "ಐವತ್ತು", "ಐವತ್ತೊಂದು", "ಐವತ್ತೆರಡು", "ಐವತ್ತಮೂರು", "ಐವತ್ತನಾಲ್ಕು", "ಐವತ್ತೈದು", "ಐವತ್ತಾರು", "ಐವತ್ತೇಳು", "ಐವತ್ತೆಂಟು", "ಐವತ್ತೊಂಬತ್ತು",
        "ಅರವತ್ತು", "ಅರವತ್ತೊಂದು", "ಅರವತ್ತೆರಡು", "ಅರವತ್ತಮೂರು", "ಅರವತ್ತನಾಲ್ಕು", "ಅರವತ್ತೈದು", "ಅರವತ್ತಾರು", "ಅರವತ್ತೇಳು", "ಅರವತ್ತೆಂಟು", "ಅರವತ್ತೊಂಬತ್ತು",
        "ಎಪ್ಪತ್ತು", "ಎಪ್ಪತ್ತೊಂದು", "ಎಪ್ಪತ್ತೆರಡು", "ಎಪ್ಪತ್ತಮೂರು", "ಎಪ್ಪತ್ತನಾಲ್ಕು", "ಎಪ್ಪತ್ತೈದು", "ಎಪ್ಪತ್ತಾರು", "ಎಪ್ಪತ್ತೇಳು", "ಎಪ್ಪತ್ತೆಂಟು", "ಎಪ್ಪತ್ತೊಂಬತ್ತು",
        "ಎಂಬತ್ತು", "ಎಂಬತ್ತೊಂದು", "ಎಂಬತ್ತೆರಡು", "ಎಂಬತ್ತಮೂರು", "ಎಂಬತ್ತನಾಲ್ಕು", "ಎಂಬತ್ತೈದು", "ಎಂಬತ್ತಾರು", "ಎಂಬತ್ತೇಳು", "ಎಂಬತ್ತೆಂಟು", "ಎಂಬತ್ತೊಂಬತ್ತು",
        "ತೊಂಬತ್ತು", "ತೊಂಬತ್ತೊಂದು", "ತೊಂಬತ್ತೆರಡು", "ತೊಂಬತ್ತಮೂರು", "ತೊಂಬತ್ತನಾಲ್ಕು", "ತೊಂಬತ್ತೈದು", "ತೊಂಬತ್ತಾರು", "ತೊಂಬತ್ತೇಳು", "ತೊಂಬತ್ತೆಂಟು", "ತೊಂಬತ್ತೊಂಬತ್ತು",
    ]
    HUNDRED = "ನೂರು"
    THOUSAND = "ಸಾವಿರ"
    LAKH = "ಲಕ್ಷ"
    CRORE = "ಕೋಟಿ"

    # Fused hundreds (standalone) and their combining (oblique) forms used
    # when a 0-99 remainder follows: 147 -> ನೂರ ನಲವತ್ತೇಳು.
    HUNDREDS = ["", "ನೂರು", "ಇನ್ನೂರು", "ಮುನ್ನೂರು", "ನಾನೂರು", "ಐನೂರು",
                "ಆರುನೂರು", "ಏಳುನೂರು", "ಎಂಟುನೂರು", "ಒಂಬೈನೂರು"]
    HUNDREDS_COMB = ["", "ನೂರ", "ಇನ್ನೂರ", "ಮುನ್ನೂರ", "ನಾನೂರ", "ಐನೂರ",
                     "ಆರುನೂರ", "ಏಳುನೂರ", "ಎಂಟುನೂರ", "ಒಂಬೈನೂರ"]
    THOUSAND_COMB = "ಸಾವಿರದ"
    LAKH_COMB = "ಲಕ್ಷದ"

    LETTERS = {
        "A": "ಎ", "B": "ಬಿ", "C": "ಸಿ", "D": "ಡಿ", "E": "ಇ", "F": "ಎಫ್",
        "G": "ಜಿ", "H": "ಎಚ್", "I": "ಐ", "J": "ಜೆ", "K": "ಕೆ", "L": "ಎಲ್",
        "M": "ಎಂ", "N": "ಎನ್", "O": "ಒ", "P": "ಪಿ", "Q": "ಕ್ಯೂ", "R": "ಆರ್",
        "S": "ಎಸ್", "T": "ಟಿ", "U": "ಯು", "V": "ವಿ", "W": "ಡಬ್ಲ್ಯೂ",
        "X": "ಎಕ್ಸ್", "Y": "ವೈ", "Z": "ಝಡ್",
    }

    MONTHS = ["ಜನವರಿ", "ಫೆಬ್ರವರಿ", "ಮಾರ್ಚ್", "ಏಪ್ರಿಲ್", "ಮೇ", "ಜೂನ್",
              "ಜುಲೈ", "ಆಗಸ್ಟ್", "ಸೆಪ್ಟೆಂಬರ್", "ಅಕ್ಟೋಬರ್", "ನವೆಂಬರ್", "ಡಿಸೆಂಬರ್"]

    SYM = {
        "rupee": "ರೂಪಾಯಿ", "dollar": "ಡಾಲರ್", "euro": "ಯೂರೋ", "pound": "ಪೌಂಡ್",
        "cent": "ಸೆಂಟ್",
        "percent": "ಪ್ರತಿಶತ",
        "point": "ದಶಮಾಂಶ",
        "minus": "ಮೈನಸ್", "plus": "ಪ್ಲಸ್", "equals": "ಸಮ",
        "times": "ಗುಣಿಸು", "divide": "ಭಾಗಿಸು",
        "squared": "ವರ್ಗ", "sqrt": "ವರ್ಗಮೂಲ",
        "lt": "ಗಿಂತ ಕಡಿಮೆ", "gt": "ಗಿಂತ ಹೆಚ್ಚು",
        "plus_minus": "ಪ್ಲಸ್ ಮೈನಸ್",
        "at": "ಅಟ್ ದಿ ರೇಟ್", "underscore": "ಅಂಡರ್‌ಸ್ಕೋರ್", "dot": "ಡಾಟ್",
        "slash": "ಸ್ಲ್ಯಾಶ್", "hash": "ಹ್ಯಾಶ್", "amp": "ಮತ್ತು",
        "half": "ಅರ್ಧ", "quarter": "ಕಾಲು", "three_quarters": "ಮುಕ್ಕಾಲು",
        "degree": "ಡಿಗ್ರಿ", "celsius": "ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್", "fahrenheit": "ಡಿಗ್ರಿ ಫ್ಯಾರನ್‌ಹೀಟ್",
        "minute": "ನಿಮಿಷ", "second": "ಸೆಕೆಂಡ್",
    }

    TIME_ON_HOUR = "{h} ಗಂಟೆ"
    TIME_HM = "{h} ಗಂಟೆ {m} ನಿಮಿಷ"
    RANGE = "{a} ರಿಂದ {b}"
    RANGE_TO = "{a} ರಿಂದ {b} ವರೆಗೆ"
    RATIO = "{a} ಅನುಪಾತ {b}"
    POWER = "{base} ಘಾತ {exp}"
    FRACTION = "{num} ಬೈ {den}"
    DECADE_SHORT = "{tens}ರ"        # decade() passes the ು-stripped stem
    DECADE_FULL = "{tens}ರ ದಶಕ"

    PERIODS = {"morning": "ಬೆಳಿಗ್ಗೆ", "afternoon": "ಮಧ್ಯಾಹ್ನ",
               "evening": "ಸಂಜೆ", "night": "ರಾತ್ರಿ"}

    TIME_MARKERS = ["ಗಂಟೆಗೆ", "ಗಂಟೆಯ", "ಗಂಟೆ"]
    MARKER_SUFFIXES = {"ಗಂಟೆಗೆ": "ಗೆ"}
    PERIOD_DETECT = {"ಬೆಳಿಗ್ಗೆ": "morning", "ಬೆಳಗ್ಗೆ": "morning",
                     "ಮುಂಜಾನೆ": "morning", "ಮಧ್ಯಾಹ್ನ": "afternoon",
                     "ಸಂಜೆ": "evening", "ರಾತ್ರಿ": "night"}
    YEAR_WORDS = ["ವರ್ಷ", "ಇಸವಿ", "ಸಾಲಿನ", "ಸಾಲು", "ಸನ್"]
    ID_KEYWORDS = ["ನಂಬರ್", "ಸಂಖ್ಯೆ", "ಖಾತೆ", "ಕೋಡ್", "ಆರ್ಡರ್", "ಫೋನ್", "ಮೊಬೈಲ್", "ಐಡಿ", "ಪಿನ್", "ಪಾಸ್‌ವರ್ಡ್", "ರೈಲು", "ಎಕ್ಸ್‌ಪ್ರೆಸ್", "ಪಿಎನ್‌ಆರ್", "ಕಾರ್ಡ್", "ರೆಫರೆನ್ಸ್", "ಟ್ರಾನ್ಸಾಕ್ಷನ್", "ಟಿಕೆಟ್", "ಓಟಿಪಿ", "ಅಕೌಂಟ್", "ಅಂಕಿ", "ಬಳಕೆದಾರ", "ಕಂತು", "ವಹಿವಾಟು"]
    ID_AFTER_WORDS = ["ಡಯಲ್", "ಕರೆ", "ಸಂಪರ್ಕ"]
    NUMBER_WORD = "ನಂಬರ್"
    ABBREV_UNITS = {"MINS":"ನಿಮಿಷ", "MIN":"ನಿಮಿಷ", "HRS":"ಗಂಟೆ", "HR":"ಗಂಟೆ", "SECS":"ಸೆಕೆಂಡ್", "SEC":"ಸೆಕೆಂಡ್", "KMS":"ಕಿಲೋಮೀಟರ್"}
    LABEL_WORDS = ["ವಿಧಿ", "ಕಲಂ", "ಕೊಠಡಿ", "ವಾರ್ಡ್", "ಅಧ್ಯಾಯ", "ಭಾಗ", "ತರಗತಿ"]
    MINUTE_WORDS = ["ನಿಮಿಷ"]
    SCALE_WORDS = {"ಕೋಟಿ": "crore", "ಲಕ್ಷ": "lakh", "ಲಕ್ಷದ": "lakh",
                   "ಸಾವಿರ": "thousand", "ಸಾವಿರದ": "thousand"}
    CURRENCY_WORDS = {"ರೂಪಾಯಿ": "rupee", "ರುಪಾಯಿ": "rupee", "ರೂ.": "rupee",
                      "ರೂ": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "ಮೊದಲನೇ"}
    ORDINAL_SUFFIX = "ನೇ"
    ORDINAL_INPUT_SUFFIXES = ["ನೆಯ", "ನೇ", "ನೆ"]

    # ==================================================================
    # Dravidian composition — fused hundreds/thousands with sandhi.
    # ==================================================================

    def hundreds(self, h, rest):
        if rest:
            return f"{self.HUNDREDS_COMB[h]} {self.two_digit(rest)}"
        return self.HUNDREDS[h]

    def under_thousand(self, n):
        if n < 100:
            return self.two_digit(n)
        return self.hundreds(n // 100, n % 100)

    def cardinal(self, n):
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n < 1000:
            return self.under_thousand(n)
        parts = []
        crore, n = divmod(n, 10**7)
        lakh, n = divmod(n, 10**5)
        thousand, n = divmod(n, 1000)
        if crore:
            # ಕೋಟಿ stays plain even when a remainder follows.
            parts.append(f"{self.cardinal(crore)} {self.CRORE}")
        if lakh:
            word = self.LAKH_COMB if (thousand or n) else self.LAKH
            parts.append(f"{self.two_digit(lakh)} {word}")
        if thousand:
            word = self.THOUSAND_COMB if n else self.THOUSAND
            if thousand == 1 and not (crore or lakh):
                # 1000 -> ಸಾವಿರ; 1947 -> ಸಾವಿರದ ಒಂಬೈನೂರ ನಲವತ್ತೇಳು (no ಒಂದು)
                parts.append(word)
            else:
                parts.append(f"{self.under_thousand(thousand)} {word}")
        if n:
            parts.append(self.under_thousand(n))
        return " ".join(parts)

    def year(self, y):
        """Kannada reads years as full cardinals (news/TTS style):
        1947 -> ಸಾವಿರದ ಒಂಬೈನೂರ ನಲವತ್ತೇಳು, 2026 -> ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು."""
        return self.cardinal(y)

    def decade(self, tens, full):
        # Genitive stem: final ು drops before -ರ (ತೊಂಬತ್ತು -> ತೊಂಬತ್ತರ).
        w = self.two_digit(tens)
        stem = w[:-1] if w.endswith("ು") else w
        tpl = self.DECADE_FULL if full else self.DECADE_SHORT
        return tpl.format(tens=stem)

    def _ordinal_compose(self, n, src_suffix):
        # ಎರಡು -> ಎರಡನೇ : final ು of the cardinal drops before ನೇ.
        base = self.cardinal(n)
        if base.endswith("ು"):
            base = base[:-1]
        suffix = src_suffix if src_suffix in self.ORDINAL_INPUT_SUFFIXES else self.ORDINAL_SUFFIX
        return f"{base}{suffix}"

    # Kannada case markers attach with oblique on ು-final words:
    # ಐದು+ಇಂದ->ಐದರಿಂದ, ಮೂರು+ಗೆ->ಮೂರಕ್ಕೆ, ಒಂಬತ್ತು+ನಲ್ಲಿ->ಒಂಬತ್ತರಲ್ಲಿ;
    # consonant/ೆ-final just attach (ನಿಮಿಷ+ಕ್ಕೆ->ನಿಮಿಷಕ್ಕೆ, ಸೊನ್ನೆ+ಗೆ->ಸೊನ್ನೆಗೆ)
    _PP = {"ಗೆ", "ಕ್ಕೆ", "ಇಂದ", "ದಿಂದ", "ರಿಂದ", "ನಲ್ಲಿ", "ಅಲ್ಲಿ",
           "ರಲ್ಲಿ", "ವರೆಗೆ", "ಗಾಗಿ", "ಕ್ಕಾಗಿ"}

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if not last or not ("\u0c80" <= last[-1] <= "\u0cff"):
            return None
        if t == last:
            return rep
        if t not in self._PP:
            return None
        base = rep[:len(rep)-len(last)]
        dative = t in ("ಗೆ", "ಕ್ಕೆ")
        ablative = t in ("ಇಂದ", "ದಿಂದ", "ರಿಂದ")
        locative = t in ("ನಲ್ಲಿ", "ಅಲ್ಲಿ", "ರಲ್ಲಿ")
        if last.endswith("\u0cc1"):              # ು-final -> drop, oblique -ರ/-ಕ್ಕೆ
            stem = last[:-1]
            suf = "ಕ್ಕೆ" if dative else "ರಿಂದ" if ablative else "ರಲ್ಲಿ" if locative else t
            return base + stem + suf
        if last.endswith(("\u0cc6", "\u0cbf", "\u0cc0")):  # ೆ/ಿ/ೀ-final
            suf = "ಗೆ" if dative else "ಯಿಂದ" if ablative else "ಯಲ್ಲಿ" if locative else t
            return base + last + suf
        # consonant/inherent-a-final -> -ಕ್ಕೆ/-ದಿಂದ/-ದಲ್ಲಿ
        suf = "ಕ್ಕೆ" if dative else "ದಿಂದ" if ablative else "ದಲ್ಲಿ" if locative else t
        return base + last + suf


class Malayalam(LangBase):
    code = "ml"
    name = "Malayalam"
    native_digits = "൦൧൨൩൪൫൬൭൮൯"
    COUNT_ONE = "ഒരു"

    ONES = [
        "പൂജ്യം", "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്", "ഒമ്പത്",
        "പത്ത്", "പതിനൊന്ന്", "പന്ത്രണ്ട്", "പതിമൂന്ന്", "പതിനാല്", "പതിനഞ്ച്", "പതിനാറ്",
        "പതിനേഴ്", "പതിനെട്ട്", "പത്തൊമ്പത്",
        "ഇരുപത്", "ഇരുപത്തിയൊന്ന്", "ഇരുപത്തിരണ്ട്", "ഇരുപത്തിമൂന്ന്", "ഇരുപത്തിനാല്",
        "ഇരുപത്തിയഞ്ച്", "ഇരുപത്തിയാറ്", "ഇരുപത്തിയേഴ്", "ഇരുപത്തിയെട്ട്", "ഇരുപത്തിയൊമ്പത്",
        "മുപ്പത്", "മുപ്പത്തിയൊന്ന്", "മുപ്പത്തിരണ്ട്", "മുപ്പത്തിമൂന്ന്", "മുപ്പത്തിനാല്",
        "മുപ്പത്തിയഞ്ച്", "മുപ്പത്തിയാറ്", "മുപ്പത്തിയേഴ്", "മുപ്പത്തിയെട്ട്", "മുപ്പത്തിയൊമ്പത്",
        "നാൽപ്പത്", "നാൽപ്പത്തിയൊന്ന്", "നാൽപ്പത്തിരണ്ട്", "നാൽപ്പത്തിമൂന്ന്", "നാൽപ്പത്തിനാല്",
        "നാൽപ്പത്തിയഞ്ച്", "നാൽപ്പത്തിയാറ്", "നാൽപ്പത്തിയേഴ്", "നാൽപ്പത്തിയെട്ട്", "നാൽപ്പത്തിയൊമ്പത്",
        "അമ്പത്", "അമ്പത്തിയൊന്ന്", "അമ്പത്തിരണ്ട്", "അമ്പത്തിമൂന്ന്", "അമ്പത്തിനാല്",
        "അമ്പത്തിയഞ്ച്", "അമ്പത്തിയാറ്", "അമ്പത്തിയേഴ്", "അമ്പത്തിയെട്ട്", "അമ്പത്തിയൊമ്പത്",
        "അറുപത്", "അറുപത്തിയൊന്ന്", "അറുപത്തിരണ്ട്", "അറുപത്തിമൂന്ന്", "അറുപത്തിനാല്",
        "അറുപത്തിയഞ്ച്", "അറുപത്തിയാറ്", "അറുപത്തിയേഴ്", "അറുപത്തിയെട്ട്", "അറുപത്തിയൊമ്പത്",
        "എഴുപത്", "എഴുപത്തിയൊന്ന്", "എഴുപത്തിരണ്ട്", "എഴുപത്തിമൂന്ന്", "എഴുപത്തിനാല്",
        "എഴുപത്തിയഞ്ച്", "എഴുപത്തിയാറ്", "എഴുപത്തിയേഴ്", "എഴുപത്തിയെട്ട്", "എഴുപത്തിയൊമ്പത്",
        "എൺപത്", "എൺപത്തിയൊന്ന്", "എൺപത്തിരണ്ട്", "എൺപത്തിമൂന്ന്", "എൺപത്തിനാല്",
        "എൺപത്തിയഞ്ച്", "എൺപത്തിയാറ്", "എൺപത്തിയേഴ്", "എൺപത്തിയെട്ട്", "എൺപത്തിയൊമ്പത്",
        "തൊണ്ണൂറ്", "തൊണ്ണൂറ്റിയൊന്ന്", "തൊണ്ണൂറ്റിരണ്ട്", "തൊണ്ണൂറ്റിമൂന്ന്", "തൊണ്ണൂറ്റിനാല്",
        "തൊണ്ണൂറ്റിയഞ്ച്", "തൊണ്ണൂറ്റിയാറ്", "തൊണ്ണൂറ്റിയേഴ്", "തൊണ്ണൂറ്റിയെട്ട്", "തൊണ്ണൂറ്റിയൊമ്പത്",
    ]
    HUNDRED = "നൂറ്"        # not used in composition (Dravidian override below)
    THOUSAND = "ആയിരം"
    LAKH = "ലക്ഷം"
    CRORE = "കോടി"

    # Fused hundreds: standalone and combining (before a remainder).
    _H_STAND = {1: "നൂറ്", 2: "ഇരുനൂറ്", 3: "മുന്നൂറ്", 4: "നാനൂറ്", 5: "അഞ്ഞൂറ്",
                6: "അറുനൂറ്", 7: "എഴുനൂറ്", 8: "എണ്ണൂറ്", 9: "തൊള്ളായിരം"}
    _H_COMB = {1: "നൂറ്റി", 2: "ഇരുനൂറ്റി", 3: "മുന്നൂറ്റി", 4: "നാനൂറ്റി", 5: "അഞ്ഞൂറ്റി",
               6: "അറുനൂറ്റി", 7: "എഴുനൂറ്റി", 8: "എണ്ണൂറ്റി", 9: "തൊള്ളായിരത്തി"}

    # Fused thousands for multipliers 1..10; others via sandhi (see _thousand).
    _TH_STAND = {1: "ആയിരം", 2: "രണ്ടായിരം", 3: "മൂവായിരം", 4: "നാലായിരം",
                 5: "അയ്യായിരം", 6: "ആറായിരം", 7: "ഏഴായിരം", 8: "എണ്ണായിരം",
                 9: "ഒമ്പതിനായിരം", 10: "പതിനായിരം"}

    _ONE_MULT = "ഒരു"       # multiplier word for 1 lakh / 1 crore

    LETTERS = {
        "A": "എ", "B": "ബി", "C": "സി", "D": "ഡി", "E": "ഇ", "F": "എഫ്",
        "G": "ജി", "H": "എച്ച്", "I": "ഐ", "J": "ജെ", "K": "കെ", "L": "എൽ",
        "M": "എം", "N": "എൻ", "O": "ഒ", "P": "പി", "Q": "ക്യു", "R": "ആർ",
        "S": "എസ്", "T": "ടി", "U": "യു", "V": "വി", "W": "ഡബ്ല്യു",
        "X": "എക്സ്", "Y": "വൈ", "Z": "ഇസഡ്",
    }

    MONTHS = ["ജനുവരി", "ഫെബ്രുവരി", "മാർച്ച്", "ഏപ്രിൽ", "മേയ്", "ജൂൺ",
              "ജൂലൈ", "ഓഗസ്റ്റ്", "സെപ്റ്റംബർ", "ഒക്ടോബർ", "നവംബർ", "ഡിസംബർ"]

    SYM = {
        "rupee": "രൂപ", "dollar": "ഡോളർ", "euro": "യൂറോ", "pound": "പൗണ്ട്",
        "cent": "സെന്റ്",
        "percent": "ശതമാനം",
        "point": "ദശാംശം",
        "minus": "മൈനസ്", "plus": "പ്ലസ്", "equals": "സമം",
        "times": "ഗുണം", "divide": "ഹരണം",
        "squared": "വർഗം", "sqrt": "വർഗമൂലം",
        "lt": "നെക്കാൾ കുറവ്", "gt": "നെക്കാൾ കൂടുതൽ",
        "plus_minus": "പ്ലസ്-മൈനസ്",
        "at": "അറ്റ് ദ റേറ്റ്", "underscore": "അണ്ടർസ്കോർ", "dot": "ഡോട്ട്",
        "slash": "സ്ലാഷ്", "hash": "ഹാഷ്", "amp": "ആൻഡ്",
        "half": "അര", "quarter": "കാൽ", "three_quarters": "മുക്കാൽ",
        "degree": "ഡിഗ്രി", "celsius": "ഡിഗ്രി സെൽഷ്യസ്", "fahrenheit": "ഡിഗ്രി ഫാരൻഹീറ്റ്",
        "minute": "മിനിറ്റ്", "second": "സെക്കൻഡ്",
    }

    TIME_ON_HOUR = "{h} മണി"
    TIME_HM = "{h} മണി {m} മിനിറ്റ്"
    RANGE = "{a} മുതൽ {b}"
    RANGE_TO = "{a} മുതൽ {b} വരെ"
    RATIO = "{a} അനുപാതം {b}"
    POWER = "{base} ഘാതം {exp}"
    FRACTION = "{num} ബൈ {den}"
    DECADE_SHORT = "{tens}കൾ"
    DECADE_FULL = "{tens}കളുടെ ദശകം"

    PERIODS = {"morning": "രാവിലെ", "afternoon": "ഉച്ചയ്ക്ക്",
               "evening": "വൈകുന്നേരം", "night": "രാത്രി"}

    TIME_MARKERS = ["മണി", "മണിക്ക്", "മണിയോടെ"]
    MARKER_SUFFIXES = {"മണിക്ക്": "ന്", "മണിയോടെ": "ഓടെ"}
    PERIOD_DETECT = {"രാവിലെ": "morning", "കാലത്ത്": "morning",
                     "ഉച്ചയ്ക്ക്": "afternoon", "ഉച്ചക്ക്": "afternoon",
                     "ഉച്ച": "afternoon", "ഉച്ചകഴിഞ്ഞ്": "afternoon",
                     "വൈകുന്നേരം": "evening", "വൈകിട്ട്": "evening",
                     "സന്ധ്യക്ക്": "evening", "രാത്രി": "night"}
    YEAR_WORDS = ["വർഷം", "വർഷത്തിൽ", "വർഷത്തെ", "കൊല്ലം", "ആണ്ട്"]
    ID_KEYWORDS = ["നമ്പർ", "നമ്പര്", "നമ്പറിൽ", "അക്കൗണ്ട്", "കോഡ്", "ഓർഡർ", "ഫോൺ", "മൊബൈൽ", "ഒടിപി", "പിൻ", "ഐഡി", "പാസ്‌വേഡ്", "ട്രെയിൻ", "എക്സ്പ്രസ്", "പിഎൻആർ", "കാർഡ്", "റഫറൻസ്", "ട്രാൻസാക്ഷൻ", "ടിക്കറ്റ്", "അക്കൗണ്ടിൽ", "നമ്പറിലേക്ക്", "യൂസർ", "തവണ", "അക്കം", "ഇടപാട്"]
    NUMBER_WORD = "നമ്പർ"
    ABBREV_UNITS = {"MINS":"മിനിറ്റ്", "MIN":"മിനിറ്റ്", "HRS":"മണിക്കൂർ", "HR":"മണിക്കൂർ", "SECS":"സെക്കൻഡ്", "SEC":"സെക്കൻഡ്", "KMS":"കിലോമീറ്റർ"}
    LABEL_WORDS = ["അനുച്ഛേദം", "വകുപ്പ്", "മുറി", "റൂം", "വാർഡ്", "ക്ലാസ്",
                   "അധ്യായം", "ഭാഗം", "ഗേറ്റ്", "കൗണ്ടർ"]
    MINUTE_WORDS = ["മിനിറ്റ്", "മിനുട്ട്"]
    SCALE_WORDS = {"കോടി": "crore", "കോടിയുടെ": "crore",
                   "ലക്ഷം": "lakh", "ലക്ഷത്തി": "lakh", "ലക്ഷത്തിന്റെ": "lakh",
                   "ആയിരം": "thousand", "ആയിരത്തി": "thousand"}
    CURRENCY_WORDS = {"രൂപ": "rupee", "രൂപാ": "rupee", "രൂപയുടെ": "rupee",
                      "രൂപയ്ക്ക്": "rupee"}

    ORDINAL_EXCEPTIONS = {}
    ORDINAL_SUFFIX = "ാം"          # ഒന്ന് -> ഒന്നാം, പത്ത് -> പത്താം
    ORDINAL_INPUT_SUFFIXES = ["ാം", "ാമത്തെ", "ാമത്"]

    _VIRAMA = "്"   # ്
    _ANUSVARA = "ം"  # ം

    # ------------------------------------------------------------------
    # Dravidian composition overrides (fused hundreds/thousands + sandhi)
    # ------------------------------------------------------------------

    def hundreds(self, h, rest):
        if rest:
            return f"{self._H_COMB[h]} {self.two_digit(rest)}"
        return self._H_STAND[h]

    def _thousand(self, m, combining):
        """Fused '<m> thousand' word for m in 1..99."""
        if m in self._TH_STAND:
            word = self._TH_STAND[m]
        else:
            base = self.ONES[m]
            if base.endswith("ത" + self._VIRAMA):
                # ends in ...ത് (round tens, -ഒമ്പത്): ഇരുപത് -> ഇരുപതിനായിരം
                word = base[:-1] + "ിനായിരം"
            elif base.endswith(self._VIRAMA):
                # virama + ആ -> ാ : അമ്പത്തിനാല് -> അമ്പത്തിനാലായിരം,
                # തൊണ്ണൂറ് -> തൊണ്ണൂറായിരം
                word = base[:-1] + "ായിരം"
            else:
                word = base + " " + self.THOUSAND
        if combining:
            # ം -> ത്തി : ആയിരം -> ആയിരത്തി
            return word[:-1] + "ത്തി"
        return word

    def cardinal(self, n):
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n < 100:
            return self.ONES[n]
        if n < 1000:
            return self.hundreds(n // 100, n % 100)
        parts = []
        crore, n = divmod(n, 10**7)
        lakh, n = divmod(n, 10**5)
        thousand, n = divmod(n, 1000)
        if crore:
            mult = self._ONE_MULT if crore == 1 else self.cardinal(crore)
            parts.append(f"{mult} {self.CRORE}")
        if lakh:
            word = self.LAKH[:-1] + "ത്തി" if (thousand or n) else self.LAKH
            mult = self._ONE_MULT if lakh == 1 else self.two_digit(lakh)
            parts.append(f"{mult} {word}")
        if thousand:
            parts.append(self._thousand(thousand, combining=bool(n)))
        if n:
            parts.append(self.under_thousand(n))
        return " ".join(parts)

    def year(self, y):
        """Malayalam reads years as full cardinals (news/TTS convention):
        1947 -> ആയിരത്തി തൊള്ളായിരത്തി നാൽപ്പത്തിയേഴ്."""
        return self.cardinal(y)

    def decade(self, tens, full):
        # തൊണ്ണൂറ് + കൾ -> തൊണ്ണൂറുകൾ (final virama -> ു before കൾ)
        stem = self.two_digit(tens)
        if stem.endswith(self._VIRAMA):
            stem = stem[:-1] + "ു"
        tpl = self.DECADE_FULL if full else self.DECADE_SHORT
        return tpl.format(tens=stem)

    def _ordinal_compose(self, n, src_suffix):
        # രണ്ട് -> രണ്ടാം : strip final virama/anusvara, then attach suffix.
        stem = self.cardinal(n)
        if stem.endswith(self._VIRAMA) or stem.endswith(self._ANUSVARA):
            stem = stem[:-1]
        suffix = src_suffix if src_suffix in self.ORDINAL_INPUT_SUFFIXES else self.ORDINAL_SUFFIX
        return f"{stem}{suffix}"

    # Malayalam case markers (often hyphen-attached in source) attach with
    # sandhi: രൂപ+ന്റെ->രൂപയുടെ, ആറ്+ൽ->ആറിൽ, രൂപ+ൽ->രൂപയിൽ
    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if not last or not ("ഀ" <= last[-1] <= "ൿ"):
            return None
        base = rep[:len(rep)-len(last)]
        virama = last.endswith("്")
        anusvara = last.endswith("ം")
        ivow = last[-1] in ("ി", "ീ")
        if t in ("ന്റെ", "ുടെ", "യുടെ"):
            if virama:   return base + last[:-1] + "ിന്റെ"
            if anusvara: return base + last[:-1] + "ത്തിന്റെ"
            return base + last + "യുടെ"
        if t in ("ൽ", "ിൽ", "ലെ", "ിലെ"):
            end = "ിലെ" if t in ("ലെ", "ിലെ") else "ിൽ"
            if virama:   return base + last[:-1] + end
            if anusvara: return base + last[:-1] + "ത്ത" + end
            return base + last + "യ" + end
        if t in ("ലേക്ക്", "ിലേക്ക്"):
            if virama:   return base + last[:-1] + "ിലേക്ക്"
            if anusvara: return base + last[:-1] + "ത്തിലേക്ക്"
            return base + last + "യിലേക്ക്"
        if t in ("ന്", "ിന്", "ക്ക്", "ിനാണ്"):
            tail = "ിനാണ്" if t == "ിനാണ്" else "ിന്"
            if virama:   return base + last[:-1] + tail
            if anusvara: return base + last[:-1] + "ത്ത" + tail
            if ivow:     return base + last + "ക്ക്"
            return base + last + "യ്ക്ക്"
        return None


class Marathi(LangBase):
    code = "mr"
    name = "Marathi"
    native_digits = "०१२३४५६७८९"
    SINGULAR_NOUNS = {"रुपये": "रुपया", "मिनिटे": "मिनिट"}

    ONES = [
        "शून्य", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ",
        "दहा", "अकरा", "बारा", "तेरा", "चौदा", "पंधरा", "सोळा", "सतरा", "अठरा", "एकोणीस",
        "वीस", "एकवीस", "बावीस", "तेवीस", "चोवीस", "पंचवीस", "सव्वीस", "सत्तावीस", "अठ्ठावीस", "एकोणतीस",
        "तीस", "एकतीस", "बत्तीस", "तेहतीस", "चौतीस", "पस्तीस", "छत्तीस", "सदतीस", "अडतीस", "एकोणचाळीस",
        "चाळीस", "एक्केचाळीस", "बेचाळीस", "त्रेचाळीस", "चव्वेचाळीस", "पंचेचाळीस", "सेहेचाळीस", "सत्तेचाळीस", "अठ्ठेचाळीस", "एकोणपन्नास",
        "पन्नास", "एक्कावन्न", "बावन्न", "त्रेपन्न", "चोपन्न", "पंचावन्न", "छप्पन्न", "सत्तावन्न", "अठ्ठावन्न", "एकोणसाठ",
        "साठ", "एकसष्ट", "बासष्ट", "त्रेसष्ट", "चौसष्ट", "पासष्ट", "सहासष्ट", "सदुसष्ट", "अडुसष्ट", "एकोणसत्तर",
        "सत्तर", "एक्काहत्तर", "बाहत्तर", "त्र्याहत्तर", "चौऱ्याहत्तर", "पंच्याहत्तर", "शहात्तर", "सत्त्याहत्तर", "अठ्ठ्याहत्तर", "एकोणऐंशी",
        "ऐंशी", "एक्क्याऐंशी", "ब्याऐंशी", "त्र्याऐंशी", "चौऱ्याऐंशी", "पंच्याऐंशी", "शहाऐंशी", "सत्त्याऐंशी", "अठ्ठ्याऐंशी", "एकोणनव्वद",
        "नव्वद", "एक्क्याण्णव", "ब्याण्णव", "त्र्याण्णव", "चौऱ्याण्णव", "पंच्याण्णव", "शहाण्णव", "सत्त्याण्णव", "अठ्ठ्याण्णव", "नव्व्याण्णव",
    ]
    HUNDRED = "शे"
    THOUSAND = "हजार"
    LAKH = "लाख"
    CRORE = "कोटी"

    LETTERS = {
        "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ",
        "G": "जी", "H": "एच", "I": "आय", "J": "जे", "K": "के", "L": "एल",
        "M": "एम", "N": "एन", "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर",
        "S": "एस", "T": "टी", "U": "यू", "V": "व्ही", "W": "डब्ल्यू",
        "X": "एक्स", "Y": "वाय", "Z": "झेड",
    }

    MONTHS = ["जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून",
              "जुलै", "ऑगस्ट", "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर"]

    SYM = {
        "rupee": "रुपये", "dollar": "डॉलर", "euro": "युरो", "pound": "पाउंड",
        "cent": "सेंट",
        "percent": "टक्के",
        "point": "पूर्णांक",
        "minus": "उणे", "plus": "अधिक", "equals": "बरोबर",
        "times": "गुणिले", "divide": "भागिले",
        "squared": "चा वर्ग", "sqrt": "वर्गमूळ",
        "lt": "पेक्षा कमी", "gt": "पेक्षा जास्त",
        "plus_minus": "अधिक-उणे",
        "at": "ॲट द रेट", "underscore": "अंडरस्कोर", "dot": "डॉट",
        "slash": "स्लॅश", "hash": "हॅश", "amp": "आणि",
        "half": "अर्धा", "quarter": "पाव", "three_quarters": "पाऊण",
        "degree": "अंश", "celsius": "अंश सेल्सिअस", "fahrenheit": "अंश फॅरनहाइट",
        "minute": "मिनिटे", "second": "सेकंद",
    }

    TIME_ON_HOUR = "{h} वाजता"
    TIME_HM = "{h} वाजून {m} मिनिटे"
    RANGE = "{a} ते {b}"
    RANGE_TO = "{a} ते {b} पर्यंत"
    RATIO = "{a} गुणोत्तर {b}"
    POWER = "{base} चा घात {exp}"
    FRACTION = "{num} बटा {den}"
    DECADE_SHORT = "{tens} च्या"
    DECADE_FULL = "{tens} चे दशक"

    PERIODS = {"morning": "सकाळी", "afternoon": "दुपारी",
               "evening": "संध्याकाळी", "night": "रात्री"}

    TIME_MARKERS = ["वाजता", "वाजून", "वाजले", "वाजल्या", "वाजेपर्यंत"]
    PERIOD_DETECT = {"सकाळी": "morning", "सकाळ": "morning", "सकाळचे": "morning",
                     "पहाटे": "morning",
                     "दुपारी": "afternoon", "दुपार": "afternoon", "दुपारचे": "afternoon",
                     "संध्याकाळी": "evening", "संध्याकाळ": "evening",
                     "सायंकाळी": "evening",
                     "रात्री": "night", "रात्र": "night", "रात्रीचे": "night"}
    YEAR_WORDS = ["साल", "साली", "सालच्या", "सन", "वर्ष", "वर्षी", "इसवी"]
    ID_KEYWORDS = ["नंबर", "क्रमांक", "खाते", "खात्याचा", "खात्यावर", "कोड", "ऑर्डर", "फोन", "मोबाईल", "मोबाइल", "अकाउंट", "अकाउंटसाठी", "अकाउंटमधून", "खात्यातून", "खात्यात", "खात्याची", "पिन", "आयडी", "पासवर्ड", "ट्रेन", "एक्सप्रेस", "पीएनआर", "कार्ड", "रेफरन्स", "ट्रांझॅक्शन", "तिकीट", "ओटीपी", "गाडी", "युजर", "यूजर", "हप्ता", "हप्त्याचा", "अंक", "व्यवहार", "ट्रान्झॅक्शन", "क्र", "क्रं", "चेक", "धनादेश"]
    ID_AFTER_WORDS = ["डायल", "कॉल", "संपर्क"]
    NUMBER_WORD = "क्रमांक"
    ABBREV_UNITS = {"MINS":"मिनिटे", "MIN":"मिनिट", "HRS":"तास", "HR":"तास", "SECS":"सेकंद", "SEC":"सेकंद", "KMS":"किलोमीटर"}
    LABEL_WORDS = ["कलम", "अनुच्छेद", "खोली", "वर्ग", "इयत्ता", "वॉर्ड",
                   "अध्याय", "भाग", "प्रकरण", "गट"]
    MINUTE_WORDS = ["मिनिटे", "मिनिटं", "मिनिट", "मिनिटांनी"]
    SCALE_WORDS = {"कोटी": "crore", "लाख": "lakh", "हजार": "thousand",
                   "अब्ज": "arab"}
    CURRENCY_WORDS = {"रुपये": "rupee", "रुपया": "rupee", "रुपयांचा": "rupee",
                      "रुपयांची": "rupee", "रु": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "पहिला", 2: "दुसरा", 3: "तिसरा", 4: "चौथा"}
    ORDINAL_SUFFIX = "वा"
    ORDINAL_INPUT_SUFFIXES = ["वा", "वी", "वे", "व्या", "वं"]

    # ------------------------------------------------------------------
    # Hundreds fuse: एक+शे -> एकशे, दोन+शे -> दोनशे (written solid).
    # Exactly 100 with no remainder is शंभर.
    # ------------------------------------------------------------------
    def hundreds(self, h, rest):
        if h == 1 and rest == 0:
            return "शंभर"
        out = f"{self.ONES[h]}{self.HUNDRED}"
        if rest:
            out += f" {self.two_digit(rest)}"
        return out

    def year(self, y):
        """1947 -> एकोणीसशे सत्तेचाळीस (fused शे); 2026 -> दोन हजार सव्वीस."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            out = f"{self.two_digit(hi)}{self.HUNDRED}"
            if lo:
                out += f" {self.two_digit(lo)}"
            return out
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # 12वा -> बारावा, 12वी -> बारावी : cardinal + preserved suffix
        if src_suffix:
            return f"{self.cardinal(n)}{src_suffix}"
        if n in self.ORDINAL_EXCEPTIONS:
            return self.ORDINAL_EXCEPTIONS[n]
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"

    # Marathi: -ए ending noun-words take oblique plural -यां before a case
    # marker (रुपये+चा->रुपयांचा, मिनिटे+ला->मिनिटांना)
    _ATTACH_NOUNS = {"रुपये", "पैसे", "डॉलर", "टक्के", "मिनिटे", "तास",
                     "किलोमीटर", "किलो", "मीटर", "लिटर", "ग्रॅम", "अंश"}
    _CASE = {"चा":"चा","ची":"ची","चे":"चे","चं":"चं","च्या":"च्या",
             "ला":"ना","ना":"ना","नी":"नी","त":"त","तून":"तून","हून":"हून",
             "पर्यंत":"पर्यंत","मध्ये":"मध्ये","साठी":"साठी"}

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-").lstrip("ांं")
        last = rep.rsplit(" ", 1)[-1]
        if last not in self._ATTACH_NOUNS or t not in self._CASE:
            return None
        base = rep[:len(rep)-len(last)]
        obl = last[:-1] + "ां" if last.endswith("े") else last
        return base + obl + self._CASE[t]


class Odia(LangBase):
    code = "or"
    name = "Odia"
    native_digits = "୦୧୨୩୪୫୬୭୮୯"

    ONES = [
        "ଶୂନ୍ୟ", "ଏକ", "ଦୁଇ", "ତିନି", "ଚାରି", "ପାଞ୍ଚ", "ଛଅ", "ସାତ", "ଆଠ", "ନଅ",
        "ଦଶ", "ଏଗାର", "ବାର", "ତେର", "ଚଉଦ", "ପନ୍ଦର", "ଷୋହଳ", "ସତର", "ଅଠର", "ଉଣେଇଶି",
        "କୋଡ଼ିଏ", "ଏକୋଇଶି", "ବାଇଶି", "ତେଇଶି", "ଚବିଶି", "ପଚିଶି", "ଛବିଶି", "ସତାଇଶି", "ଅଠେଇଶି", "ଅଣତିରିଶି",
        "ତିରିଶି", "ଏକତିରିଶି", "ବତିଶି", "ତେତିଶି", "ଚଉତିରିଶି", "ପଞ୍ଚତିରିଶି", "ଛତିଶି", "ସଇଁତିରିଶି", "ଅଠତିରିଶି", "ଅଣଚାଳିଶି",
        "ଚାଳିଶି", "ଏକଚାଳିଶି", "ବୟାଳିଶି", "ତେୟାଳିଶି", "ଚଉରାଳିଶି", "ପଞ୍ଚଚାଳିଶି", "ଛୟାଳିଶି", "ସତଚାଳିଶି", "ଅଠଚାଳିଶି", "ଅଣଚାଶ",
        "ପଚାଶ", "ଏକାବନ", "ବାଉନ", "ତେପନ", "ଚଉବନ", "ପଞ୍ଚାବନ", "ଛପନ", "ସତାବନ", "ଅଠାବନ", "ଅଣଷଠି",
        "ଷାଠିଏ", "ଏକଷଠି", "ବାଷଠି", "ତେଷଠି", "ଚଉଷଠି", "ପଞ୍ଚଷଠି", "ଛଅଷଠି", "ସତଷଠି", "ଅଠଷଠି", "ଅଣସ୍ତରି",
        "ସତୁରି", "ଏକସ୍ତରି", "ବାସ୍ତରି", "ତେସ୍ତରି", "ଚଉସ୍ତରି", "ପଞ୍ଚସ୍ତରି", "ଛଅସ୍ତରି", "ସତସ୍ତରି", "ଅଠସ୍ତରି", "ଅଣାଅଶୀ",
        "ଅଶୀ", "ଏକାଅଶୀ", "ବୟାଅଶୀ", "ତେୟାଅଶୀ", "ଚଉରାଅଶୀ", "ପଞ୍ଚାଅଶୀ", "ଛୟାଅଶୀ", "ସତାଅଶୀ", "ଅଠାଅଶୀ", "ଅଣାନବେ",
        "ନବେ", "ଏକାନବେ", "ବୟାନବେ", "ତେୟାନବେ", "ଚଉରାନବେ", "ପଞ୍ଚାନବେ", "ଛୟାନବେ", "ସତାନବେ", "ଅଠାନବେ", "ଅନେଶତ",
    ]
    HUNDRED = "ଶହ"
    THOUSAND = "ହଜାର"
    LAKH = "ଲକ୍ଷ"
    CRORE = "କୋଟି"

    LETTERS = {
        "A": "ଏ", "B": "ବି", "C": "ସି", "D": "ଡି", "E": "ଇ", "F": "ଏଫ୍",
        "G": "ଜି", "H": "ଏଚ୍", "I": "ଆଇ", "J": "ଜେ", "K": "କେ", "L": "ଏଲ୍",
        "M": "ଏମ୍", "N": "ଏନ୍", "O": "ଓ", "P": "ପି", "Q": "କ୍ୟୁ", "R": "ଆର୍",
        "S": "ଏସ୍", "T": "ଟି", "U": "ୟୁ", "V": "ଭି", "W": "ଡବ୍ଲ୍ୟୁ",
        "X": "ଏକ୍ସ", "Y": "ୱାଇ", "Z": "ଜେଡ୍",
    }

    MONTHS = ["ଜାନୁଆରୀ", "ଫେବୃଆରୀ", "ମାର୍ଚ୍ଚ", "ଅପ୍ରେଲ", "ମଇ", "ଜୁନ",
              "ଜୁଲାଇ", "ଅଗଷ୍ଟ", "ସେପ୍ଟେମ୍ବର", "ଅକ୍ଟୋବର", "ନଭେମ୍ବର", "ଡିସେମ୍ବର"]

    SYM = {
        "rupee": "ଟଙ୍କା", "dollar": "ଡଲାର", "euro": "ୟୁରୋ", "pound": "ପାଉଣ୍ଡ",
        "cent": "ସେଣ୍ଟ",
        "percent": "ପ୍ରତିଶତ",
        "point": "ଦଶମିକ",
        "minus": "ମାଇନସ୍", "plus": "ପ୍ଲସ୍", "equals": "ସମାନ",
        "times": "ଗୁଣ", "divide": "ଭାଗ",
        "squared": "ର ବର୍ଗ", "sqrt": "ବର୍ଗମୂଳ",
        "lt": "ଠାରୁ କମ୍", "gt": "ଠାରୁ ଅଧିକ",
        "plus_minus": "ପ୍ଲସ୍-ମାଇନସ୍",
        "at": "ଆଟ୍ ଦି ରେଟ୍", "underscore": "ଅଣ୍ଡରସ୍କୋର", "dot": "ଡଟ୍",
        "slash": "ସ୍ଲାସ୍", "hash": "ହ୍ୟାସ୍", "amp": "ଆଣ୍ଡ",
        "half": "ଅଧା", "quarter": "ଏକ ଚତୁର୍ଥାଂଶ", "three_quarters": "ତିନି ଚତୁର୍ଥାଂଶ",
        "degree": "ଡିଗ୍ରୀ", "celsius": "ଡିଗ୍ରୀ ସେଲସିୟସ୍", "fahrenheit": "ଡିଗ୍ରୀ ଫାରେନହାଇଟ୍",
        "minute": "ମିନିଟ୍",
        "paisa": "ପଇସା", "per": "ପ୍ରତି", "second": "ସେକେଣ୍ଡ",
    }

    TIME_ON_HOUR = "{h}ଟା"
    TIME_HM = "{h}ଟା {m} ମିନିଟ୍"
    RANGE = "{a} ରୁ {b}"
    RANGE_TO = "{a} ରୁ {b} ପର୍ଯ୍ୟନ୍ତ"
    RATIO = "{a} ଅନୁପାତ {b}"
    POWER = "{base} ର ଘାତ {exp}"
    FRACTION = "{num} ବଟା {den}"
    DECADE_SHORT = "{tens} ଦଶକ"
    DECADE_FULL = "{tens} ଦଶକ"

    PERIODS = {"morning": "ସକାଳ", "afternoon": "ଦିନ",
               "evening": "ସନ୍ଧ୍ୟା", "night": "ରାତି"}

    TIME_MARKERS = ["ଟା", "ଟାରେ", "ଟାରୁ", "ବେଳେ", "ବଜେ"]
    PERIOD_DETECT = {"ସକାଳ": "morning", "ସକାଳେ": "morning", "ସକାଳୁ": "morning",
                     "ଭୋର": "morning", "ଭୋରରୁ": "morning",
                     "ଦିନ": "afternoon", "ଅପରାହ୍ନ": "afternoon",
                     "ମଧ୍ୟାହ୍ନ": "afternoon", "ଦ୍ୱିପହର": "afternoon",
                     "ସନ୍ଧ୍ୟା": "evening", "ସନ୍ଧ୍ୟାରେ": "evening",
                     "ରାତି": "night", "ରାତିରେ": "night", "ରାତ୍ରି": "night"}
    YEAR_WORDS = ["ମସିହା", "ମସିହାରେ", "ସାଲ", "ସନ", "ବର୍ଷ", "ବର୍ଷରେ"]
    ID_KEYWORDS = ["ନମ୍ବର", "ନମ୍ବର୍", "ଖାତା", "ଆକାଉଣ୍ଟ", "କୋଡ୍", "ଅର୍ଡର", "ଫୋନ୍", "ଫୋନ", "ମୋବାଇଲ୍", "ମୋବାଇଲ", "ଓଟିପି", "ପିନ୍", "ଆଇଡି", "ପାସୱାର୍ଡ", "ଏକ୍ସପ୍ରେସ", "ପିଏନ୍‌ଆର୍", "କାର୍ଡ", "ରେଫରେନ୍ସ", "ଟିକେଟ", "ଆକାଉଣ୍ଟରୁ", "ଟ୍ରେନ୍", "ୟୁଜର", "କିସ୍ତି", "ଅଙ୍କ", "ଲାଣଦେଣ"]
    ID_AFTER_WORDS = ["ଡାଏଲ", "କଲ", "ଯୋଗାଯୋଗ"]
    NUMBER_WORD = "ନମ୍ବର"
    ABBREV_UNITS = {"MINS":"ମିନିଟ୍", "MIN":"ମିନିଟ୍", "HRS":"ଘଣ୍ଟା", "HR":"ଘଣ୍ଟା", "SECS":"ସେକେଣ୍ଡ", "SEC":"ସେକେଣ୍ଡ", "KMS":"କିଲୋମିଟର"}
    LABEL_WORDS = ["ଅନୁଚ୍ଛେଦ", "ଧାରା", "ରୁମ୍", "କୋଠରି", "ୱାର୍ଡ",
                   "ଅଧ୍ୟାୟ", "ଭାଗ", "ଗେଟ୍", "କାଉଣ୍ଟର", "ଶ୍ରେଣୀ"]
    MINUTE_WORDS = ["ମିନିଟ୍", "ମିନିଟ"]
    SCALE_WORDS = {"କୋଟି": "crore", "ଲକ୍ଷ": "lakh", "ଲାଖ": "lakh",
                   "ହଜାର": "thousand"}
    CURRENCY_WORDS = {"ଟଙ୍କା": "rupee", "ଟଙ୍କାର": "rupee", "ଟଙ୍କାରେ": "rupee",
                      "ଟଂକା": "rupee", "ରୁପି": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "ପ୍ରଥମ", 2: "ଦ୍ୱିତୀୟ", 3: "ତୃତୀୟ", 4: "ଚତୁର୍ଥ",
                          5: "ପଞ୍ଚମ", 6: "ଷଷ୍ଠ", 7: "ସପ୍ତମ", 8: "ଅଷ୍ଟମ",
                          9: "ନବମ", 10: "ଦଶମ"}
    ORDINAL_SUFFIX = "ତମ"
    ORDINAL_INPUT_SUFFIXES = ["ତମ"]

    # Odia postpositions agglutinate onto noun-words (ଟଙ୍କା + ରୁ -> ଟଙ୍କାରୁ)
    _ATTACH_NOUNS = {"ଟଙ୍କା", "ପଇସା", "ଡଲାର", "ପ୍ରତିଶତ", "କିଲୋମିଟର",
                     "କିଲୋଗ୍ରାମ", "ମିଟର", "ଲିଟର", "ଗ୍ରାମ", "ଡିଗ୍ରୀ",
                     "ମିନିଟ୍", "ଘଣ୍ଟା", "ସେକେଣ୍ଡ"}
    _PP = {"ରେ", "ରୁ", "କୁ", "ର", "ଠାରୁ", "ପର୍ଯ୍ୟନ୍ତ", "ମଧ୍ୟରେ",
           "ପାଇଁ", "ଅପେକ୍ଷା", "ଭିତରେ"}

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if t == last and last in self._ATTACH_NOUNS:
            return rep
        if t in self._PP and last in self._ATTACH_NOUNS:
            return rep + t
        return None


class Punjabi(LangBase):
    code = "pa"
    name = "Punjabi"
    native_digits = "੦੧੨੩੪੫੬੭੮੯"
    SINGULAR_NOUNS = {"ਘੰਟੇ": "ਘੰਟਾ"}

    ONES = [
        "ਸਿਫ਼ਰ", "ਇੱਕ", "ਦੋ", "ਤਿੰਨ", "ਚਾਰ", "ਪੰਜ", "ਛੇ", "ਸੱਤ", "ਅੱਠ", "ਨੌਂ",
        "ਦਸ", "ਗਿਆਰਾਂ", "ਬਾਰਾਂ", "ਤੇਰਾਂ", "ਚੌਦਾਂ", "ਪੰਦਰਾਂ", "ਸੋਲਾਂ", "ਸਤਾਰਾਂ", "ਅਠਾਰਾਂ", "ਉੱਨੀ",
        "ਵੀਹ", "ਇੱਕੀ", "ਬਾਈ", "ਤੇਈ", "ਚੌਵੀ", "ਪੱਚੀ", "ਛੱਬੀ", "ਸਤਾਈ", "ਅਠਾਈ", "ਉਣੱਤੀ",
        "ਤੀਹ", "ਇਕੱਤੀ", "ਬੱਤੀ", "ਤੇਤੀ", "ਚੌਂਤੀ", "ਪੈਂਤੀ", "ਛੱਤੀ", "ਸੈਂਤੀ", "ਅਠੱਤੀ", "ਉਣਤਾਲੀ",
        "ਚਾਲੀ", "ਇਕਤਾਲੀ", "ਬਿਆਲੀ", "ਤਰਤਾਲੀ", "ਚੁਤਾਲੀ", "ਪੰਤਾਲੀ", "ਛਿਆਲੀ", "ਸੰਤਾਲੀ", "ਅਠਤਾਲੀ", "ਉਣੰਜਾ",
        "ਪੰਜਾਹ", "ਇਕਵੰਜਾ", "ਬਵੰਜਾ", "ਤਰਵੰਜਾ", "ਚੁਰੰਜਾ", "ਪਚਵੰਜਾ", "ਛਪੰਜਾ", "ਸਤਵੰਜਾ", "ਅਠਵੰਜਾ", "ਉਣਾਹਠ",
        "ਸੱਠ", "ਇਕਾਹਠ", "ਬਾਹਠ", "ਤ੍ਰੇਹਠ", "ਚੌਂਹਠ", "ਪੈਂਹਠ", "ਛਿਆਹਠ", "ਸਤਾਹਠ", "ਅਠਾਹਠ", "ਉਣੱਤਰ",
        "ਸੱਤਰ", "ਇਕਹੱਤਰ", "ਬਹੱਤਰ", "ਤਿਹੱਤਰ", "ਚੁਹੱਤਰ", "ਪੰਝੱਤਰ", "ਛਿਹੱਤਰ", "ਸਤੱਤਰ", "ਅਠੱਤਰ", "ਉਣਾਸੀ",
        "ਅੱਸੀ", "ਇਕਾਸੀ", "ਬਿਆਸੀ", "ਤਰਾਸੀ", "ਚੁਰਾਸੀ", "ਪਚਾਸੀ", "ਛਿਆਸੀ", "ਸਤਾਸੀ", "ਅਠਾਸੀ", "ਉਣਾਨਵੇਂ",
        "ਨੱਬੇ", "ਇਕਾਨਵੇਂ", "ਬਾਨਵੇਂ", "ਤਰਾਨਵੇਂ", "ਚੁਰਾਨਵੇਂ", "ਪਚਾਨਵੇਂ", "ਛਿਆਨਵੇਂ", "ਸਤਾਨਵੇਂ", "ਅਠਾਨਵੇਂ", "ਨੜਿੰਨਵੇਂ",
    ]
    HUNDRED = "ਸੌ"
    THOUSAND = "ਹਜ਼ਾਰ"
    LAKH = "ਲੱਖ"
    CRORE = "ਕਰੋੜ"

    LETTERS = {
        "A": "ਏ", "B": "ਬੀ", "C": "ਸੀ", "D": "ਡੀ", "E": "ਈ", "F": "ਐੱਫ",
        "G": "ਜੀ", "H": "ਐੱਚ", "I": "ਆਈ", "J": "ਜੇ", "K": "ਕੇ", "L": "ਐੱਲ",
        "M": "ਐੱਮ", "N": "ਐੱਨ", "O": "ਓ", "P": "ਪੀ", "Q": "ਕਿਊ", "R": "ਆਰ",
        "S": "ਐੱਸ", "T": "ਟੀ", "U": "ਯੂ", "V": "ਵੀ", "W": "ਡਬਲਿਊ",
        "X": "ਐਕਸ", "Y": "ਵਾਈ", "Z": "ਜ਼ੈੱਡ",
    }

    MONTHS = ["ਜਨਵਰੀ", "ਫ਼ਰਵਰੀ", "ਮਾਰਚ", "ਅਪ੍ਰੈਲ", "ਮਈ", "ਜੂਨ",
              "ਜੁਲਾਈ", "ਅਗਸਤ", "ਸਤੰਬਰ", "ਅਕਤੂਬਰ", "ਨਵੰਬਰ", "ਦਸੰਬਰ"]

    SYM = {
        "rupee": "ਰੁਪਏ", "dollar": "ਡਾਲਰ", "euro": "ਯੂਰੋ", "pound": "ਪੌਂਡ",
        "cent": "ਸੈਂਟ",
        "percent": "ਫ਼ੀਸਦੀ",
        "point": "ਦਸ਼ਮਲਵ",
        "minus": "ਮਾਈਨਸ", "plus": "ਪਲੱਸ", "equals": "ਬਰਾਬਰ",
        "times": "ਗੁਣਾ", "divide": "ਭਾਗ",
        "squared": "ਦਾ ਵਰਗ", "sqrt": "ਵਰਗਮੂਲ",
        "lt": "ਤੋਂ ਘੱਟ", "gt": "ਤੋਂ ਵੱਧ",
        "plus_minus": "ਧਨ-ਰਿਣ",
        "at": "ਐਟ ਦ ਰੇਟ", "underscore": "ਅੰਡਰਸਕੋਰ", "dot": "ਡਾਟ",
        "slash": "ਸਲੈਸ਼", "hash": "ਹੈਸ਼", "amp": "ਅਤੇ",
        "half": "ਅੱਧਾ", "quarter": "ਚੌਥਾਈ", "three_quarters": "ਤਿੰਨ ਚੌਥਾਈ",
        "degree": "ਡਿਗਰੀ", "celsius": "ਡਿਗਰੀ ਸੈਲਸੀਅਸ", "fahrenheit": "ਡਿਗਰੀ ਫਾਰਨਹਾਈਟ",
        "minute": "ਮਿੰਟ", "second": "ਸਕਿੰਟ",
    }

    TIME_ON_HOUR = "{h} ਵਜੇ"
    TIME_HM = "{h} ਵੱਜ ਕੇ {m} ਮਿੰਟ"
    RANGE = "{a} ਤੋਂ {b}"
    RANGE_TO = "{a} ਤੋਂ {b} ਤੱਕ"
    RATIO = "{a} ਅਨੁਪਾਤ {b}"
    POWER = "{base} ਦੀ ਘਾਤ {exp}"
    FRACTION = "{num} ਬਟਾ {den}"
    DECADE_SHORT = "{tens} ਦੇ"
    DECADE_FULL = "{tens} ਦਾ ਦਹਾਕਾ"

    PERIODS = {"morning": "ਸਵੇਰੇ", "afternoon": "ਦੁਪਹਿਰ", "evening": "ਸ਼ਾਮ", "night": "ਰਾਤ"}

    TIME_MARKERS = ["ਵਜੇ", "ਵੱਜ ਕੇ", "ਵਜਕੇ", "ਵੱਜੇ"]
    PERIOD_DETECT = {"ਸਵੇਰੇ": "morning", "ਸਵੇਰ": "morning", "ਤੜਕੇ": "morning",
                     "ਦੁਪਹਿਰ": "afternoon", "ਸ਼ਾਮ": "evening", "ਸ਼ਾਮੀਂ": "evening",
                     "ਰਾਤ": "night", "ਰਾਤੀਂ": "night"}
    YEAR_WORDS = ["ਸਾਲ", "ਵਰ੍ਹਾ", "ਵਰ੍ਹੇ", "ਸੰਨ"]
    ID_KEYWORDS = ["ਨੰਬਰ", "ਖਾਤਾ", "ਖਾਤੇ", "ਕੋਡ", "ਆਰਡਰ", "ਫ਼ੋਨ", "ਫੋਨ", "ਮੋਬਾਈਲ", "ਮੋਬਾਇਲ", "ਅਕਾਊਂਟ", "ਅਕਾਊਂਟਵਿੱਚ", "ਪਿੰਨ", "ਆਈਡੀ", "ਪਾਸਵਰਡ", "ਟ੍ਰੇਨ", "ਐਕਸਪ੍ਰੈਸ", "ਪੀਐਨਆਰ", "ਕਾਰਡ", "ਰੈਫਰੈਂਸ", "ਟਿਕਟ", "ਓਟੀਪੀ", "ਯੂਜ਼ਰ", "ਕਿਸ਼ਤ", "ਅੰਕ", "ਲੈਣ-ਦੇਣ"]
    ID_AFTER_WORDS = ["ਡਾਇਲ", "ਕਾਲ", "ਸੰਪਰਕ"]
    NUMBER_WORD = "ਨੰਬਰ"
    ABBREV_UNITS = {"MINS":"ਮਿੰਟ", "MIN":"ਮਿੰਟ", "HRS":"ਘੰਟੇ", "HR":"ਘੰਟਾ", "SECS":"ਸਕਿੰਟ", "SEC":"ਸਕਿੰਟ", "KMS":"ਕਿਲੋਮੀਟਰ"}
    LABEL_WORDS = ["ਧਾਰਾ", "ਅਨੁਛੇਦ", "ਕਮਰਾ", "ਕਲਾਸ", "ਵਾਰਡ", "ਅਧਿਆਇ", "ਭਾਗ", "ਜਮਾਤ"]
    MINUTE_WORDS = ["ਮਿੰਟ"]
    SCALE_WORDS = {"ਕਰੋੜ": "crore", "ਲੱਖ": "lakh", "ਹਜ਼ਾਰ": "thousand", "ਹਜਾਰ": "thousand",
                   "ਅਰਬ": "arab"}
    CURRENCY_WORDS = {"ਰੁਪਏ": "rupee", "ਰੁਪਿਆ": "rupee", "ਰੁਪਈਏ": "rupee", "ਰੁਪਿਏ": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "ਪਹਿਲਾ", 2: "ਦੂਜਾ", 3: "ਤੀਜਾ", 4: "ਚੌਥਾ", 6: "ਛੇਵਾਂ"}
    ORDINAL_SUFFIX = "ਵਾਂ"
    ORDINAL_INPUT_SUFFIXES = ["ਵੀਂ", "ਵਾਂ", "ਵੇਂ", "ਵਾ"]

    def _ordinal_compose(self, n, src_suffix):
        # 12ਵੀਂ -> ਬਾਰਾਂਵੀਂ : cardinal + preserved suffix
        if src_suffix:
            return f"{self.cardinal(n)}{src_suffix}"
        if n in self.ORDINAL_EXCEPTIONS:
            return self.ORDINAL_EXCEPTIONS[n]
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"


class Tamil(LangBase):
    code = "ta"
    name = "Tamil"
    native_digits = "௦௧௨௩௪௫௬௭௮௯"
    COUNT_ONE = "ஒரு"

    ONES = [
        "பூஜ்ஜியம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு",
        "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது",
        "பத்து", "பதினொன்று", "பன்னிரண்டு", "பதின்மூன்று", "பதினான்கு",
        "பதினைந்து", "பதினாறு", "பதினேழு", "பதினெட்டு", "பத்தொன்பது",
        "இருபது", "இருபத்தொன்று", "இருபத்திரண்டு", "இருபத்துமூன்று", "இருபத்துநான்கு",
        "இருபத்தைந்து", "இருபத்தாறு", "இருபத்தேழு", "இருபத்தெட்டு", "இருபத்தொன்பது",
        "முப்பது", "முப்பத்தொன்று", "முப்பத்திரண்டு", "முப்பத்துமூன்று", "முப்பத்துநான்கு",
        "முப்பத்தைந்து", "முப்பத்தாறு", "முப்பத்தேழு", "முப்பத்தெட்டு", "முப்பத்தொன்பது",
        "நாற்பது", "நாற்பத்தொன்று", "நாற்பத்திரண்டு", "நாற்பத்துமூன்று", "நாற்பத்துநான்கு",
        "நாற்பத்தைந்து", "நாற்பத்தாறு", "நாற்பத்தேழு", "நாற்பத்தெட்டு", "நாற்பத்தொன்பது",
        "ஐம்பது", "ஐம்பத்தொன்று", "ஐம்பத்திரண்டு", "ஐம்பத்துமூன்று", "ஐம்பத்துநான்கு",
        "ஐம்பத்தைந்து", "ஐம்பத்தாறு", "ஐம்பத்தேழு", "ஐம்பத்தெட்டு", "ஐம்பத்தொன்பது",
        "அறுபது", "அறுபத்தொன்று", "அறுபத்திரண்டு", "அறுபத்துமூன்று", "அறுபத்துநான்கு",
        "அறுபத்தைந்து", "அறுபத்தாறு", "அறுபத்தேழு", "அறுபத்தெட்டு", "அறுபத்தொன்பது",
        "எழுபது", "எழுபத்தொன்று", "எழுபத்திரண்டு", "எழுபத்துமூன்று", "எழுபத்துநான்கு",
        "எழுபத்தைந்து", "எழுபத்தாறு", "எழுபத்தேழு", "எழுபத்தெட்டு", "எழுபத்தொன்பது",
        "எண்பது", "எண்பத்தொன்று", "எண்பத்திரண்டு", "எண்பத்துமூன்று", "எண்பத்துநான்கு",
        "எண்பத்தைந்து", "எண்பத்தாறு", "எண்பத்தேழு", "எண்பத்தெட்டு", "எண்பத்தொன்பது",
        "தொண்ணூறு", "தொண்ணூற்றொன்று", "தொண்ணூற்றிரண்டு", "தொண்ணூற்றுமூன்று", "தொண்ணூற்றுநான்கு",
        "தொண்ணூற்றைந்து", "தொண்ணூற்றாறு", "தொண்ணூற்றேழு", "தொண்ணூற்றெட்டு", "தொண்ணூற்றொன்பது",
    ]
    HUNDRED = "நூறு"
    THOUSAND = "ஆயிரம்"
    LAKH = "லட்சம்"
    CRORE = "கோடி"

    # Fused hundreds (index 1..9) and their combining forms before a remainder.
    HUNDREDS = ["", "நூறு", "இருநூறு", "முந்நூறு", "நானூறு", "ஐந்நூறு",
                "அறுநூறு", "எழுநூறு", "எண்ணூறு", "தொள்ளாயிரம்"]
    HUNDREDS_COMB = ["", "நூற்று", "இருநூற்று", "முந்நூற்று", "நானூற்று", "ஐந்நூற்று",
                     "அறுநூற்று", "எழுநூற்று", "எண்ணூற்று", "தொள்ளாயிரத்து"]

    LETTERS = {
        "A": "ஏ", "B": "பி", "C": "சி", "D": "டி", "E": "இ", "F": "எஃப்",
        "G": "ஜி", "H": "எச்", "I": "ஐ", "J": "ஜே", "K": "கே", "L": "எல்",
        "M": "எம்", "N": "என்", "O": "ஓ", "P": "பி", "Q": "கியூ", "R": "ஆர்",
        "S": "எஸ்", "T": "டி", "U": "யூ", "V": "வி", "W": "டபிள்யூ",
        "X": "எக்ஸ்", "Y": "ஒய்", "Z": "இசட்",
    }

    MONTHS = ["ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்",
              "ஜூலை", "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்"]

    SYM = {
        "rupee": "ரூபாய்", "dollar": "டாலர்", "euro": "யூரோ", "pound": "பவுண்ட்",
        "cent": "சென்ட்",
        "percent": "சதவீதம்",
        "point": "புள்ளி",
        "minus": "மைனஸ்", "plus": "பிளஸ்", "equals": "சமம்",
        "times": "பெருக்கல்", "divide": "வகுத்தல்",
        "squared": "இன் வர்க்கம்", "sqrt": "வர்க்கமூலம்",
        "lt": "விடக் குறைவு", "gt": "விட அதிகம்",
        "plus_minus": "பிளஸ் மைனஸ்",
        "at": "அட் தி ரேட்", "underscore": "அண்டர்ஸ்கோர்", "dot": "டாட்",
        "slash": "ஸ்லாஷ்", "hash": "ஹேஷ்", "amp": "மற்றும்",
        "half": "அரை", "quarter": "கால்", "three_quarters": "முக்கால்",
        "degree": "டிகிரி", "celsius": "டிகிரி செல்சியஸ்",
        "fahrenheit": "டிகிரி ஃபாரன்ஹீட்",
        "minute": "நிமிடம்", "second": "வினாடி",
    }

    TIME_ON_HOUR = "{h} மணி"
    TIME_HM = "{h} மணி {m} நிமிடம்"
    RANGE = "{a} முதல் {b}"
    RANGE_TO = "{a} முதல் {b} வரை"
    RATIO = "{a} விகிதம் {b}"
    POWER = "{base} இன் அடுக்கு {exp}"
    FRACTION = "{num} வகுத்தல் {den}"
    DECADE_SHORT = "{tens}கள்"
    DECADE_FULL = "{tens}களின் தசாப்தம்"

    PERIODS = {"morning": "காலை", "afternoon": "மதியம்",
               "evening": "மாலை", "night": "இரவு"}

    TIME_MARKERS = ["மணிக்கு", "மணி"]
    MARKER_SUFFIXES = {"மணிக்கு": "க்கு"}
    PERIOD_DETECT = {"காலை": "morning", "அதிகாலை": "morning",
                     "மதியம்": "afternoon", "நண்பகல்": "afternoon",
                     "பிற்பகல்": "afternoon",
                     "மாலை": "evening", "இரவு": "night"}
    YEAR_WORDS = ["ஆண்டு", "ஆண்டில்", "வருடம்", "வருடத்தில்", "வருஷம்"]
    ID_KEYWORDS = ["எண்", "நம்பர்", "கணக்கு", "தொலைபேசி", "போன்", "மொபைல்", "ஆர்டர்", "ஓடிபி", "பின்", "ஐடி", "கடவுச்சொல்", "ரயில்", "எக்ஸ்பிரஸ்", "பிஎன்ஆர்", "கார்டு", "குறியீடு", "டிக்கெட்", "கணக்கில்", "யூசர்", "தவணை", "இலக்கம்", "பரிவர்த்தனை"]
    ID_AFTER_WORDS = ["டயல்", "அழை", "தொடர்பு"]
    NUMBER_WORD = "எண்"
    ABBREV_UNITS = {"MINS":"நிமிடம்", "MIN":"நிமிடம்", "HRS":"மணி", "HR":"மணி", "SECS":"விநாடி", "SEC":"விநாடி", "KMS":"கிலோமீட்டர்"}
    LABEL_WORDS = ["பிரிவு", "சரத்து", "அறை", "வார்டு", "அத்தியாயம்",
                   "வகுப்பு", "பாகம்", "கவுண்டர்", "கேட்"]
    MINUTE_WORDS = ["நிமிடம்", "நிமிடங்கள்", "நிமிஷம்"]
    SCALE_WORDS = {"கோடி": "crore", "கோடியே": "crore",
                   "லட்சம்": "lakh", "லட்சத்து": "lakh", "இலட்சம்": "lakh",
                   "ஆயிரம்": "thousand", "ஆயிரத்து": "thousand"}
    CURRENCY_WORDS = {"ரூபாய்": "rupee", "ரூ": "rupee", "ருபாய்": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "முதலாவது"}
    ORDINAL_SUFFIX = "ஆவது"
    ORDINAL_INPUT_SUFFIXES = ["ஆவது", "வது", "ஆம்", "ம்"]

    # Consonants that double onto a preceding combining form (வல்லினம்).
    _DOUBLE = {"க": "க்", "ச": "ச்", "த": "த்", "ப": "ப்"}

    # ------------------------------------------------------------------
    # Dravidian composition: fused hundreds/thousands + sandhi joins.
    # ------------------------------------------------------------------
    def _sandhi_join(self, parts):
        """Join number parts; after a combining form ending -த்து/-ற்று a
        following க/ச/த/ப doubles onto it (ஆயிரத்துத் தொள்ளாயிரம்)."""
        out = [p for p in parts if p]
        for i in range(len(out) - 1):
            if out[i].endswith(("த்து", "ற்று")) and out[i + 1][0] in self._DOUBLE:
                out[i] += self._DOUBLE[out[i + 1][0]]
        return " ".join(out)

    def hundreds(self, h, rest):
        if rest:
            return self._sandhi_join([self.HUNDREDS_COMB[h], self.two_digit(rest)])
        return self.HUNDREDS[h]

    def under_thousand(self, n):
        if n < 100:
            return self.two_digit(n)
        return self.hundreds(n // 100, n % 100)

    def _thousands_word(self, t, combining):
        """1 -> ஆயிரம்/ஆயிரத்து; else fuse multiplier: final -ு drops before
        ஆ- (நான்கு+ஆயிரம் -> நான்காயிரம், ஐம்பத்துநான்கு -> ஐம்பத்துநான்காயிரம்)."""
        base = "ஆயிரத்து" if combining else "ஆயிரம்"
        if t == 1:
            return base
        return self.ONES[t][:-1] + "ா" + base[1:]

    def cardinal(self, n):
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n == 0:
            return self.ONES[0]
        crore, rest = divmod(n, 10 ** 7)
        lakh, rest = divmod(rest, 10 ** 5)
        thousand, low = divmod(rest, 1000)
        parts = []
        if crore:
            mult = "ஒரு" if crore == 1 else self.cardinal(crore)
            parts.append(f"{mult} " + ("கோடியே" if (lakh or thousand or low) else "கோடி"))
        if lakh:
            mult = "ஒரு" if lakh == 1 else self.two_digit(lakh)
            parts.append(f"{mult} " + ("லட்சத்து" if (thousand or low) else "லட்சம்"))
        if thousand:
            parts.append(self._thousands_word(thousand, bool(low)))
        if low:
            parts.append(self.under_thousand(low))
        return self._sandhi_join(parts)

    def year(self, y):
        """Tamil reads years as full cardinals:
        1947 -> ஆயிரத்துத் தொள்ளாயிரத்து நாற்பத்தேழு,
        2026 -> இரண்டாயிரத்து இருபத்தாறு."""
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # 12ஆவது/12வது -> பன்னிரண்டாவது; 12ஆம்/12ம் -> பன்னிரண்டாம்.
        suffix = "ஆம்" if src_suffix in ("ஆம்", "ம்") else "ஆவது"
        if n == 1:
            return "முதலாம்" if suffix == "ஆம்" else "முதலாவது"
        base = self.cardinal(n)
        if base[-1] in ("ு", "்"):
            # drop final -u / virama and merge the suffix vowel: -ாவது / -ாம்
            return base[:-1] + "ா" + suffix[1:]
        if base[-1] in ("ி", "ீ", "ை"):
            # glide insertion: கோடி -> கோடியாவது
            return base + "யா" + suffix[1:]
        return base + suffix

    # Tamil case markers attach with sandhi to noun-words
    _ATTACH_NOUNS = {"ரூபாய்", "டாலர்", "சதவீதம்", "மணி", "நிமிடம்",
                     "விநாடி", "கிலோமீட்டர்", "கிலோ", "மீட்டர்", "லிட்டர்",
                     "கிராம்", "டிகிரி", "மைல்"}

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        base = rep[:len(rep)-len(last)]
        num_obl = {"ஒன்று":"ஒன்ற","இரண்டு":"இரண்ட","மூன்று":"மூன்ற",
                   "நான்கு":"நான்க","ஐந்து":"ஐந்த","ஆறு":"ஆற","ஏழு":"ஏழ",
                   "எட்டு":"எட்ட","ஒன்பது":"ஒன்பத","பத்து":"பத்த","பூஜ்ஜியம்":"பூஜ்ஜியத்த"}
        if last in num_obl:                       # number + case (hyphen markers)
            if t in ("இல்","ல்"): return base+num_obl[last]+"ில்"
            if t in ("க்கு","கு"): return base+num_obl[last]+"ிற்கு"
            if t in ("இலிருந்து","ல்இருந்து"): return base+num_obl[last]+"ிலிருந்து"
            return None
        if last not in self._ATTACH_NOUNS:
            return None
        if t in ("க்கு", "கு"):
            if last == "நிமிடம்":
                return base + "நிமிடத்திற்கு"
            if last == "சதவீதம்":
                return base + "சதவீதத்திற்கு"
            return base + last + "க்கு"
        if t in ("இல்", "ல்", "இலிருந்து", "ல்இருந்து"):
            if last.endswith("ம்"):
                return base + last[:-2] + "த்தில்"
            return base + last + "இல்"
        return None


class Telugu(LangBase):
    code = "te"
    name = "Telugu"
    native_digits = "౦౧౨౩౪౫౬౭౮౯"

    ONES = [
        "సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
        "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పందొమ్మిది",
        "ఇరవై", "ఇరవై ఒకటి", "ఇరవై రెండు", "ఇరవై మూడు", "ఇరవై నాలుగు", "ఇరవై ఐదు", "ఇరవై ఆరు", "ఇరవై ఏడు", "ఇరవై ఎనిమిది", "ఇరవై తొమ్మిది",
        "ముప్పై", "ముప్పై ఒకటి", "ముప్పై రెండు", "ముప్పై మూడు", "ముప్పై నాలుగు", "ముప్పై ఐదు", "ముప్పై ఆరు", "ముప్పై ఏడు", "ముప్పై ఎనిమిది", "ముప్పై తొమ్మిది",
        "నలభై", "నలభై ఒకటి", "నలభై రెండు", "నలభై మూడు", "నలభై నాలుగు", "నలభై ఐదు", "నలభై ఆరు", "నలభై ఏడు", "నలభై ఎనిమిది", "నలభై తొమ్మిది",
        "యాభై", "యాభై ఒకటి", "యాభై రెండు", "యాభై మూడు", "యాభై నాలుగు", "యాభై ఐదు", "యాభై ఆరు", "యాభై ఏడు", "యాభై ఎనిమిది", "యాభై తొమ్మిది",
        "అరవై", "అరవై ఒకటి", "అరవై రెండు", "అరవై మూడు", "అరవై నాలుగు", "అరవై ఐదు", "అరవై ఆరు", "అరవై ఏడు", "అరవై ఎనిమిది", "అరవై తొమ్మిది",
        "డెబ్బై", "డెబ్బై ఒకటి", "డెబ్బై రెండు", "డెబ్బై మూడు", "డెబ్బై నాలుగు", "డెబ్బై ఐదు", "డెబ్బై ఆరు", "డెబ్బై ఏడు", "డెబ్బై ఎనిమిది", "డెబ్బై తొమ్మిది",
        "ఎనభై", "ఎనభై ఒకటి", "ఎనభై రెండు", "ఎనభై మూడు", "ఎనభై నాలుగు", "ఎనభై ఐదు", "ఎనభై ఆరు", "ఎనభై ఏడు", "ఎనభై ఎనిమిది", "ఎనభై తొమ్మిది",
        "తొంభై", "తొంభై ఒకటి", "తొంభై రెండు", "తొంభై మూడు", "తొంభై నాలుగు", "తొంభై ఐదు", "తొంభై ఆరు", "తొంభై ఏడు", "తొంభై ఎనిమిది", "తొంభై తొమ్మిది",
    ]
    # Indo-Aryan style fields kept for interface completeness; the fused
    # forms below (వంద/నూట/వందలు/వందల etc.) are what cardinal() actually uses.
    HUNDRED = "వంద"
    THOUSAND = "వెయ్యి"
    LAKH = "లక్ష"
    CRORE = "కోటి"

    LETTERS = {
        "A": "ఏ", "B": "బీ", "C": "సీ", "D": "డీ", "E": "ఈ", "F": "ఎఫ్",
        "G": "జీ", "H": "హెచ్", "I": "ఐ", "J": "జే", "K": "కే", "L": "ఎల్",
        "M": "ఎం", "N": "ఎన్", "O": "ఓ", "P": "పీ", "Q": "క్యూ", "R": "ఆర్",
        "S": "ఎస్", "T": "టీ", "U": "యూ", "V": "వీ", "W": "డబ్ల్యూ",
        "X": "ఎక్స్", "Y": "వై", "Z": "జెడ్",
    }

    MONTHS = ["జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్",
              "జూలై", "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్"]
    MONTH_ALIASES = {"సెప్టెంబరు": 9, "అక్టోబరు": 10, "నవంబరు": 11, "డిసెంబరు": 12}

    SYM = {
        "rupee": "రూపాయలు", "dollar": "డాలర్లు", "euro": "యూరోలు",
        "pound": "పౌండ్లు", "cent": "సెంట్లు",
        "percent": "శాతం", "paisa": "పైసలు",
        "point": "పాయింట్",
        "minus": "మైనస్", "plus": "ప్లస్", "equals": "సమానం",
        "times": "ఇంటూ", "divide": "భాగించగా",
        "squared": "వర్గం", "sqrt": "వర్గమూలం",
        "lt": "కంటే తక్కువ", "gt": "కంటే ఎక్కువ",
        "plus_minus": "ప్లస్ మైనస్",
        "at": "ఎట్ ద రేట్", "underscore": "అండర్ స్కోర్", "dot": "డాట్",
        "slash": "స్లాష్", "hash": "హ్యాష్", "amp": "మరియు",
        "half": "అర", "quarter": "పావు", "three_quarters": "ముప్పావు",
        "degree": "డిగ్రీలు", "celsius": "డిగ్రీల సెల్సియస్", "fahrenheit": "డిగ్రీల ఫారెన్‌హీట్",
        "minute": "నిమిషాలు", "per": "పర్", "second": "సెకన్లు",
    }

    TIME_ON_HOUR = "{h} గంటలు"
    TIME_HM = "{h} గంటల {m} నిమిషాలు"
    RANGE = "{a} నుంచి {b}"
    RANGE_TO = "{a} నుంచి {b} వరకు"
    RATIO = "{a} నిష్పత్తి {b}"
    POWER = "{base} పవర్ {exp}"
    FRACTION = "{num} బై {den}"
    DECADE_SHORT = "{tens}ల"
    DECADE_FULL = "{tens}ల దశకం"

    PERIODS = {"morning": "ఉదయం", "afternoon": "మధ్యాహ్నం",
               "evening": "సాయంత్రం", "night": "రాత్రి"}

    TIME_MARKERS = ["గంటలకు", "గంటలకి", "గంటకు", "గంటకి", "గంటల", "గంటలు", "గం"]
    MARKER_SUFFIXES = {"గంటలకు": "కు", "గంటలకి": "కి", "గంటకు": "కు",
                       "గంటకి": "కి", "నిమిషాలకు": "కు", "నిమిషానికి": "కి"}
    PERIOD_DETECT = {"ఉదయం": "morning", "పొద్దున": "morning", "పొద్దున్న": "morning",
                     "తెల్లవారుజామున": "morning", "మధ్యాహ్నం": "afternoon",
                     "సాయంత్రం": "evening", "రాత్రి": "night"}
    YEAR_WORDS = ["సంవత్సరం", "సంవత్సరంలో", "ఏడాది", "సం"]
    ID_KEYWORDS = ["నంబర్", "నంబరు", "నెంబర్", "నెంబరు", "ఖాతా", "ఫోన్", "మొబైల్", "కోడ్", "ఆర్డర్", "ఎక్స్ప్రెస్", "express", "అకౌంట్", "టికెట్", "రిజిస్ట్రేషన్", "యూజర్", "వాయిదా", "అంకె", "లావాదేవీ", "పిన్"]
    ID_AFTER_WORDS = ["ఎండ్", "ముగిసే", "ముగుస్తుంది", "చివర", "చివరి", "అంకెలతో", "డయల్", "పిన్", "పిన్‌కోడ్", "pin", "pincode", "కాల్", "సంప్రదించండి"]
    LABEL_WORDS = ["గది", "వార్డు", "అధికరణ", "అధ్యాయం", "తరగతి", "భాగం", "సెక్షన్",
                   "ప్లాట్ఫారమ్", "ప్లాట్‌ఫారమ్", "సీటు", "సీట్", "కోచ్", "గేట్",
                   "రూమ్", "హాల్", "టెర్మినల్", "కౌంటర్"]
    MINUTE_WORDS = ["నిమిషాలకు", "నిమిషానికి", "నిమిషాలు", "నిమిషాల", "నిమిషం"]
    SCALE_WORDS = {"కోటి": "crore", "కోటీ": "crore", "కోట్లు": "crore", "కోట్ల": "crore",
                   "లక్ష": "lakh", "లక్షా": "lakh", "లక్షలు": "lakh", "లక్షల": "lakh",
                   "వెయ్యి": "thousand", "వేలు": "thousand", "వేల": "thousand"}
    CURRENCY_WORDS = {"రూపాయలు": "rupee", "రూపాయల": "rupee", "రూపాయి": "rupee",
                      "రూపాయ": "rupee", "రూ": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "మొదటి"}
    ORDINAL_SUFFIX = "వ"
    ORDINAL_INPUT_SUFFIXES = ["వ", "వది", "ో"]

    # ==================================================================
    # Dravidian composition — fused/sandhi forms (overrides base).
    # ==================================================================

    def hundreds(self, h, rest):
        """100 వంద / నూట; 200+ వందలు / combining వందల."""
        if h == 1:
            if rest:
                return f"నూట {self.ONES[rest]}"
            return "వంద"
        if rest:
            return f"{self.ONES[h]} వందల {self.ONES[rest]}"
        return f"{self.ONES[h]} వందలు"

    def under_thousand(self, n):
        if n < 100:
            return self.ONES[n]
        return self.hundreds(n // 100, n % 100)

    def _thousands(self, t, has_rest):
        """t in 1..99. 1000 వెయ్యి (same before a remainder); 2+ వేలు/వేల."""
        if t == 1:
            return "వెయ్యి"
        return f"{self.ONES[t]} {'వేల' if has_rest else 'వేలు'}"

    def _lakhs(self, l, has_rest):
        """l in 1..99. లక్ష / sandhi లక్షా; 2+ లక్షలు/లక్షల."""
        if l == 1:
            return "లక్షా" if has_rest else "లక్ష"
        return f"{self.ONES[l]} {'లక్షల' if has_rest else 'లక్షలు'}"

    def _crores(self, c, has_rest):
        """కోటి / sandhi కోటీ; 2+ కోట్లు/కోట్ల."""
        if c == 1:
            return "కోటీ" if has_rest else "కోటి"
        return f"{self.cardinal(c)} {'కోట్ల' if has_rest else 'కోట్లు'}"

    def cardinal(self, n):
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n < 100:
            return self.ONES[n]
        if n < 1000:
            return self.under_thousand(n)
        crore, rest = divmod(n, 10**7)
        lakh, rest = divmod(rest, 10**5)
        thousand, sub = divmod(rest, 1000)
        parts = []
        if crore:
            parts.append(self._crores(crore, bool(lakh or thousand or sub)))
        if lakh:
            parts.append(self._lakhs(lakh, bool(thousand or sub)))
        if thousand:
            parts.append(self._thousands(thousand, bool(sub)))
        if sub:
            parts.append(self.under_thousand(sub))
        return " ".join(parts)

    def year(self, y):
        """Spoken Telugu reads 1100-1999 hundred-style — 1947 పందొమ్మిది
        వందల నలభై ఏడు (the "వెయ్యి తొమ్మిది వందల…" full cardinal sounds
        stiff; flagged by native review). 2000+ stays full cardinal:
        2026 రెండు వేల ఇరవై ఆరు."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            if lo:
                return f"{self.ONES[hi]} వందల {self.ONES[lo]}"
            return f"{self.ONES[hi]} వందలు"
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # Formal -వ ordinal with the regular Telugu stem change:
        # రెండు -> రెండవ, ఎనిమిది -> ఎనిమిదవ, ఇరవై -> ఇరవయ్యవ, వంద -> వందవ.
        if n in self.ORDINAL_EXCEPTIONS and not src_suffix:
            return self.ORDINAL_EXCEPTIONS[n]
        word = self.cardinal(n)
        if word.endswith("ు") or word.endswith("ి"):   # ు / ి matra
            stem = word[:-1]
        elif word.endswith("ై"):                            # ై matra
            stem = word[:-1] + "య్య"
        else:
            stem = word
        return f"{stem}{self.ORDINAL_SUFFIX}"

    # ------------------------------------------------------------------
    # Inherited Telugu grammar hooks. Upstream cited an external review, not
    # independently verified human/native-speaker approval.
    # ------------------------------------------------------------------

    # metric units read as full words, not letter names
    UNIT_WORDS = {
        "KM": "కిలోమీటర్లు", "KG": "కిలోలు", "GM": "గ్రాములు",
        "MG": "మిల్లీగ్రాములు", "MCG": "మైక్రోగ్రాములు",
        "ML": "మిల్లీలీటర్లు", "CM": "సెంటీమీటర్లు", "MM": "మిల్లీమీటర్లు",
        "H": "అవర్", "HR": "అవర్",
    }

    NUMBER_WORD = "నంబర్"
    ABBREV_UNITS = {"MINS": "నిమిషాలు", "MIN": "నిమిషాలు",
                    "HRS": "గంటలు", "HR": "గంట",
                    "SECS": "సెకన్లు", "SEC": "సెకన్లు", "KMS": "కిలోమీటర్లు"}

    # singular noun forms used after "1" (ఒక రూపాయి, ఒక గంట)
    _NOUN_SINGULAR = {"రూపాయలు": "రూపాయి", "డాలర్లు": "డాలర్",
                      "గంటలు": "గంట", "నిమిషాలు": "నిమిషం", "పైసలు": "పైసా", "సెకన్లు": "సెకను"}

    def _oblique_last(self, words):
        """Oblique stem for the final word before an appended noun:
        వందలు/వేలు/లక్షలు/కోట్లు -> వందల/వేల/లక్షల/కోట్ల ; ఒకటి -> ఒక."""
        last = words.rsplit(" ", 1)[-1]
        if last.endswith("లు"):
            return words[:-1]          # strip ు : వందలు -> వందల
        return words

    def decimal(self, int_part, frac_digits):
        frac = " ".join(self.digit(int(d)) for d in frac_digits)
        return f"{self._oblique_last(self.cardinal(int_part))} " \
               f"{self.SYM['point']} {frac}"

    def join_amount(self, words, noun):
        if words == "ఒకటి":
            return f"ఒక {self._NOUN_SINGULAR.get(noun, noun)}"
        return f"{self._oblique_last(words)} {noun}"

    def time_phrase(self, h, m, period_key=None):
        h12 = h % 12 or 12
        hw = "ఒక" if h12 == 1 else self.two_digit(h12)
        hnoun = "గంట" if h12 == 1 else ("గంటలు" if m == 0 else "గంటల")
        if h12 == 1 and m != 0:
            hnoun = "గంట"
        if m == 0:
            core = f"{hw} {hnoun}"
        elif m == 1:
            core = f"{hw} {hnoun} ఒక నిమిషం"
        else:
            core = f"{hw} {hnoun} {self.two_digit(m)} నిమిషాలు"
        if period_key:
            return f"{self.PERIODS[period_key]} {core}"
        return core

    # postpositions appearing as a separate token right after a rendered
    # span get attached with correct sandhi ("గంటలు కి" -> "గంటలకు")
    _PP_SPACED = {"వరకు", "దాకా", "లోపు", "నుండి", "నుంచి",
                  "పాటు", "క్రితం", "తర్వాత", "ముందు", "చొప్పున", "కంటే", "దగ్గర", "లోపల"}
    _PP_GLUE = {"కి", "కు", "ని", "న", "నే", "కే", "లో", "తో",
                "లోని", "లోకి", "లోనే", "కల్లా"}
    # గా keeps the nominative: రూపాయలు + గా -> రూపాయలుగా
    _PP_NOMINATIVE = {"గా", "గానే"}
    # common counted nouns: a plural-final cardinal right before these takes
    # the oblique stem (మూడు లక్షలు రూపాయలు -> మూడు లక్షల రూపాయలు)
    _COUNT_NOUNS = {"రూపాయలు", "మంది", "కిలోలు", "కిలోమీటర్లు", "గంటలు",
                    "నిమిషాలు", "రోజులు", "సంవత్సరాలు", "ఏళ్లు", "సార్లు",
                    "మార్కులు", "ఓట్లు", "సీట్లు", "పరుగులు", "వికెట్లు",
                    "లీటర్లు", "గ్రాములు", "టన్నులు", "ఎకరాలు", "శాతం",
                    "కోట్లు", "లక్షలు", "వడ్డీ", "జీతం", "ఖర్చు",
                    "ధర", "ఆదాయం", "లాభం", "నష్టం", "పెట్టుబడి", "రుణం",
                    "జరిమానా", "బడ్జెట్", "మోతాదు", "మాత్ర", "మాత్రలు", "టాబ్లెట్", "కోట్ల", "లక్షల",
                    "దూరం", "బరువు", "వేగం", "మైళ్లు", "మైళ్ళు", "అడుగులు", "సెకన్లు",
                    "పేజీలు", "సమయం", "సమయానికి"}

    def scale_noun(self, canon, numstr):
        one = numstr.replace(",", "") == "1"
        return {"crore": "కోటి" if one else "కోట్లు",
                "lakh": "లక్ష" if one else "లక్షలు",
                "thousand": "వెయ్యి" if one else "వేలు"}.get(canon, canon)

    _PP_ROMAN = {"ki": "కి", "ku": "కు", "ni": "ని", "lo": "లో",
                 "ga": "గా", "ke": "కే", "varaku": "వరకు", "nunchi": "నుంచి",
                 "nundi": "నుండి", "daggara": "దగ్గర"}

    def attach_postposition(self, rep, tok):
        tok = self._PP_ROMAN.get(tok.lower(), tok)
        last = rep.rsplit(" ", 1)[-1]
        if rep == "ఒకటి" and tok[:1].isascii() and tok[:1].isalpha():
            return "ఒక " + tok               # 1 litre -> ఒక litre
        if not last or not ("ఀ" <= last[-1] <= "౿"):
            return None            # last word not Telugu (Latin unit etc.)
        if last in self.LETTERS.values():
            return None            # bare letter names take no glued case
        if tok == last:
            return rep             # duplicate word (వరకు వరకు) — consume
        plural = last.endswith("లు")
        if tok in self._PP_NOMINATIVE:
            return rep + tok                 # రూపాయలు గా -> రూపాయలుగా
        base_tok = tok.replace("\u200c", "")
        for end in ("ను", "ని", "కు", "కి", "లో", "గా", "తో", "పై"):
            if base_tok.endswith(end) and base_tok[:-len(end)] in self._COUNT_NOUNS:
                base_tok = base_tok[:-len(end)]
                break
        if plural and base_tok in self._COUNT_NOUNS:
            return rep[:-1] + " " + tok
        if rep == "ఒకటి" and tok in self._COUNT_NOUNS:
            return "ఒక " + self._NOUN_SINGULAR.get(tok, tok)
        if tok in self._PP_SPACED:
            return (rep[:-1] if plural else rep) + " " + tok
        if tok not in self._PP_GLUE:
            return None
        if last == "శాతం" and tok in ("కి", "కు"):
            return rep[:-1] + "ానికి"        # శాతం కి -> శాతానికి
        if plural:
            if tok in ("కి", "కు"):
                return rep[:-1] + tok        # keep the source marker: గంటలు కి -> గంటలకి
            if tok == "ని":
                return rep[:-1] + "ను"       # రూపాయలు ని -> రూపాయలను
            return rep[:-1] + tok            # గంటలు లో -> గంటలలో
        if tok in ("కి", "కు"):
            return rep + tok                 # keep the source marker: ఐదు కి -> ఐదుకి
        return rep + tok


class Assamese(Bengali):
    """Assamese lexicon; Bengali grammar/entries remain the documented fallback.

    Number forms: AI4Bharat data; Gregorian months: Unicode CLDR as.xml.
    No new inference dependency and no translation of customer prose.
    """
    code = "as"
    name = "Assamese"
    if __package__:
        from .assamese_numbers import ONES, HUNDREDS
    else:
        from assamese_numbers import ONES, HUNDREDS
    HUNDRED = "শ"
    THOUSAND = "হাজাৰ"
    MONTHS = ['জানুৱাৰী', 'ফেব্ৰুৱাৰী', 'মাৰ্চ', 'এপ্ৰিল', 'মে’', 'জুন', 'জুলাই', 'আগষ্ট', 'ছেপ্তেম্বৰ', 'অক্টোবৰ', 'নৱেম্বৰ', 'ডিচেম্বৰ']
    SYM = {**Bengali.SYM, "rupee": "টকা", "paisa": "পইচা", "dollar": "ডলাৰ",
           "euro": "ইউৰো", "pound": "পাউণ্ড"}
    CURRENCY_WORDS = {**Bengali.CURRENCY_WORDS, "টকা": "rupee", "টকাৰ": "rupee",
                      "টকাত": "rupee", "ৰুপী": "rupee"}
    SCALE_WORDS = {**Bengali.SCALE_WORDS, "হাজাৰ": "thousand"}
    ID_KEYWORDS = Bengali.ID_KEYWORDS + ["নম্বৰ", "একাউণ্ট", "কাৰ্ড", "নম্বৰত", "অংক"]
    NUMBER_WORD = "নম্বৰ"
    YEAR_WORDS = Bengali.YEAR_WORDS + ["চন", "চনত", "বছৰ", "বছৰত"]
    TIME_HM = "{h} বাজি {m} মিনিট"
    RANGE = "{a}ৰ পৰা {b}"
    RANGE_TO = "{a}ৰ পৰা {b}লৈ"
    FRACTION = "{num} বাই {den}"
    SINGULAR_NOUNS = {}

    def hundreds(self, h, rest):
        return self.HUNDREDS[h] + (" " + self.two_digit(rest) if rest else "")

    def year(self, y):
        return self.cardinal(y)

LANGUAGES = {cls.code: cls() for cls in (
    Hindi, Bengali, Assamese, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu
)}
'''),
    'mithi_tts.text_norm.tn.normalize': (False, r'''"""Dependency-free, reviewable TTS normalization.

Indian English and ten Indic language tables share one corrected parser.
The selected locale controls numbers, currency, dates, units and identifiers;
it does not translate prose or choose/change the model conditioning tag.
Hindi speech output folds nukta and ends standalone है with a danda outside
explicit question sentences (ending in ?).
These are intentional pronunciation/pause workarounds, not spelling corrections.
Invalid/ambiguous tokens fall back to digit and separator words.
FR3_TN_STRICT=1 retains the strict audit contract.
See docs/NORMALIZATION_REVIEW.md for the contract and remaining listening gates.
"""
# -*- coding: utf-8 -*-
# The former generator/indic_norm package is absent from this repository.
# This tracked file is the source of truth; protect edits with tests/test_normalize.py.
import re
import os
import json
import sys
import argparse
from datetime import date
import unicodedata
from functools import lru_cache

# Keep CLI execution and the imported table module on the same LangBase class.
if __name__ == "__main__":
    sys.modules.setdefault("normalize", sys.modules[__name__])


LANGUAGE_NAMES = {"en-IN": "Indian English", "hi": "Hindi", "te": "Telugu",
    "ta": "Tamil", "bn": "Bengali", "as": "Assamese", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "or": "Odia", "pa": "Punjabi"}
_LANGUAGE_ALIASES = {alias.lower(): code for code, name in LANGUAGE_NAMES.items()
    for alias in (code, name, code if code == "en-IN" else code + "-IN")}
_LANGUAGE_ALIASES.update({"en": "en-IN", "english": "en-IN", "en-us": "en-IN",
    "en-gb": "en-IN", "indian-english": "en-IN", "oriya": "or", "od": "or"})

# Input aliases, not translations of arbitrary prose. Keep identifiers digitwise
# even when callers write OTP in native script (including Indian English mode).
_OTP_INPUT_WORDS = {"ओटीपी", "ఓటీపీ", "ఓటిపి", "ஓடிபி", "ஓடீபி", "ওটিপি",
                    "ઓટીપી", "ಒಟಿಪಿ", "ಓಟಿಪಿ", "ഒടിപി", "ഓടിപി", "ଓଟିପି", "ਓਟੀਪੀ"}
_PAISA_INPUT_WORDS = {"paise", "paisa", "पैसा", "पैसे", "పైసా", "పైసలు", "பைசா",
                      "পয়সা", "પૈસા", "ಪೈಸೆ", "പൈസ", "ପଇସା", "ਪੈਸਾ", "ਪੈਸੇ"}


def resolve_language(language):
    if not isinstance(language, str) or language.lower() not in _LANGUAGE_ALIASES:
        raise NormalizationError("unsupported_normalization_language", "Choose one of the documented normalization languages.")
    return _LANGUAGE_ALIASES[language.lower()]


@lru_cache(maxsize=12)
def language_table(language):
    code = resolve_language(language)
    if code == "en-IN":
        return LANG
    if __package__:
        from .indic_languages import LANGUAGES
    else:
        from indic_languages import LANGUAGES
    return LANGUAGES[code]


class NormalizationError(ValueError):
    """A safe-to-return error code. Never includes the caller's financial text."""
    def __init__(self, code, message, span=None, source=None):
        self.span, self.source = span, source
        self.code = code
        super().__init__(message)


_TAMIL_MONTHS = ["ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்",
                 "ஜூலை", "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்"]
# Calendar days only. Native-speaker/audio review is required before rollout.
_TAMIL_ORDINALS = ["", "முதலாம்", "இரண்டாம்", "மூன்றாம்", "நான்காம்", "ஐந்தாம்",
    "ஆறாம்", "ஏழாம்", "எட்டாம்", "ஒன்பதாம்", "பத்தாம்", "பதினொன்றாம்",
    "பன்னிரண்டாம்", "பதின்மூன்றாம்", "பதினான்காம்", "பதினைந்தாம்", "பதினாறாம்",
    "பதினேழாம்", "பதினெட்டாம்", "பத்தொன்பதாம்", "இருபதாம்", "இருபத்தொன்றாம்",
    "இருபத்திரண்டாம்", "இருபத்துமூன்றாம்", "இருபத்துநான்காம்", "இருபத்தைந்தாம்",
    "இருபத்தாறாம்", "இருபத்தேழாம்", "இருபத்தெட்டாம்", "இருபத்தொன்பதாம்",
    "முப்பதாம்", "முப்பத்தொன்றாம்"]


def _ascii_digits(text):
    # str.isdigit also accepts superscripts. Only Unicode DECIMAL digits belong
    # here; keep math superscripts for the math matcher.
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)


# Hindi speech compatibility, not a general Unicode diacritic remover. Apply to
# FINAL output because number/date/letter/URL expansions can introduce nukta.
# Keep every other script, accent, vowel sign and nasal mark untouched.
_HINDI_NUKTA_FOLD = str.maketrans({
    "\u093c": None,
    "\u0929": "\u0928", "\u0931": "\u0930", "\u0934": "\u0933",
    "\u0958": "\u0915", "\u0959": "\u0916", "\u095a": "\u0917",
    "\u095b": "\u091c", "\u095c": "\u0921", "\u095d": "\u0922",
    "\u095e": "\u092b", "\u095f": "\u092f",
})


def _calendar_check(day, month, year=2000, span=None):
    # A missing year can include Feb 29; never invent a year in the output.
    try:
        date(year, month, day)
    except ValueError:
        raise NormalizationError("invalid_date", "Invalid calendar date; correct the input.", span=span) from None


def _sentence_before(text, start):
    return re.split(r"[.!?;।॥\n]", text[max(0, start - 100):start])[-1]


def _sentence_after(text, end):
    return re.split(r"[.!?;।॥\n]", text[end:end + 100])[0]


def _valid_grouping(number):
    integer = number.lstrip("+-−").split(".")[0]
    return "," not in integer or bool(re.fullmatch(
        r"[0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,2}(?:,[0-9]{2})*,[0-9]{3}", integer))


def _month_lookup(locale):
    return {**_ENGLISH_MONTHS, **{name.lower(): i + 1 for i, name in enumerate(locale.MONTHS)},
            **{name.lower(): i for name, i in locale.MONTH_ALIASES.items()}}


def validate_text(text, *, date_order="DMY", two_digit_year_base=None, language="en-IN"):
    """Validate before destructive cleanup. Dates default to explicit DMY policy.

    Two-digit years require a caller-supplied century (1900 or 2000). Short dates
    in date context must be spelled out; ordinary fractions remain fractions.
    Plain text only: SSML/templates are rejected, not stripped and spoken.
    """
    if not isinstance(text, str):
        raise NormalizationError("invalid_text", "text must be a string.")
    if len(text) > 20000:
        raise NormalizationError("text_too_long", "text exceeds the 20000 character limit.")
    if date_order not in ("DMY", "MDY"):
        raise NormalizationError("invalid_date_order", "date_order must be DMY or MDY.")
    if two_digit_year_base is not None and (type(two_digit_year_base) is not int or two_digit_year_base not in (1900, 2000)):
        raise NormalizationError("invalid_century", "two_digit_year_base must be 1900 or 2000.")
    locale = language_table(language)
    t = _ascii_digits(text)
    if (bad := re.search(r"<\s*(?:/?[A-Za-z_]|[!?])|[{}]|\$\{", t)):
        raise NormalizationError("markup_or_placeholder", "Resolve templates and send plain text, not markup/SSML.", span=bad.span())
    for i, c in enumerate(t):
        if (unicodedata.category(c) == "Cc" and c not in "\n\r\t") or unicodedata.category(c) == "Co":
            raise NormalizationError("control_character", "Unsupported control character in text.", span=(i, i + 1))
    # URI numeric components are identifiers, not dates, amounts or clocks.
    t = re.sub(r"https?://\S+|www\.\S+|\b[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}", lambda m: " " * len(m[0]), t)
    if (bad := re.search(r"[0-9]{31,}", t)):
        raise NormalizationError("number_too_long", "Numeric spans are limited to 30 digits.", span=bad.span())
    if (bad := re.search(r"\d{4}-\d{2}-\d{2}T\d", t, re.I)):
        raise NormalizationError("unsupported_timestamp", "Spell out the date, time and timezone separately.", span=bad.span())
    for m in re.finditer(r"(?<![\w.])[+-]?[0-9][0-9,]*(?:\.[0-9]+)?", t):
        if len(re.sub(r"\D", "", m[0])) > 30:
            raise NormalizationError("number_too_long", "Numeric spans are limited to 30 digits.", span=m.span())
        if not _valid_grouping(m[0].rstrip(",")):
            raise NormalizationError("invalid_grouping", "Invalid number grouping; send ungrouped digits or standard grouping.", span=m.span())
    for m in re.finditer(r"(?<![\w\d])([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})(?!\d)", t):
        _calendar_check(int(m[3]), int(m[2]), int(m[1]), span=m.span())
    for m in re.finditer(r"(?<![\w\d-])([0-9]{1,2})([/.-])([0-9]{1,2})\2([0-9]{2,4})(?!\d)", t):
        y = int(m[4])
        if len(m[4]) != 4:
            if len(m[4]) != 2 or two_digit_year_base is None:
                raise NormalizationError("ambiguous_year", "Use a four-digit year or explicitly configure the century.", span=m.span())
            y += two_digit_year_base
        a, b = int(m[1]), int(m[3])
        _calendar_check(a if date_order == "DMY" else b, b if date_order == "DMY" else a, y, span=m.span())
    months = {**{name: i + 1 for i, name in enumerate(_TAMIL_MONTHS)}, **_month_lookup(locale)}
    month_re = "|".join(sorted(map(re.escape, months), key=len, reverse=True))
    ordinal_re = "|".join(sorted(map(re.escape, set(locale.ORDINAL_INPUT_SUFFIXES) | {"st", "nd", "rd", "th", "ஆம்", "ம்"}), key=len, reverse=True))
    for pattern, reverse in [
        (rf"(?<!\w)([0-9]{{1,2}})(?:-?(?:{ordinal_re}))?[ -]+({month_re})(?!\w)(?:[ ,\-]+([0-9]{{4}}))?", False),
        (rf"(?<!\w)({month_re})[ -]+([0-9]{{1,2}})(?:-?(?:{ordinal_re}))?(?!\d)(?:,?\s+([0-9]{{4}}))?", True),
    ]:
        for m in re.finditer(pattern, t, re.I):
            day, month = (m[2], m[1]) if reverse else (m[1], m[2])
            _calendar_check(int(day), months[month.lower()], int(m[3]) if m[3] else 2000, span=m.span())
    tamil_month_re = "|".join(_TAMIL_MONTHS)
    for m in re.finditer(rf"({tamil_month_re})\s+(\d{{1,2}})(?:-?(?:ஆம்|ம்))?(?:\s*தேதி)?(?:\s+(\d{{4}}))?", t):
        _calendar_check(int(m[2]), months[m[1]], int(m[3]) if m[3] else 2000, span=m.span())
    if (bad := re.search(r"\b(?:on|by|before|after|date|due)\s+[0-9]{1,2}/[0-9]{1,2}(?![/\d])", t, re.I)):
        raise NormalizationError("ambiguous_short_date", "Spell out the month and year for a short date.", span=bad.span())
    if (bad := re.search(r"தேதி\s+[0-9]{1,2}\.\s*[0-9]{4}", t)):
        raise NormalizationError("conflicting_date", "Conflicting date components; confirm the intended date.", span=bad.span())
    rupee_words = [word for word, key in locale.CURRENCY_WORDS.items() if key == "rupee"]
    rupee_aliases = "|".join(sorted(map(re.escape, rupee_words + ["INR", "Rs"]), key=len, reverse=True))
    rupee_prefix = rf"(?:₹|(?<!\w)(?:{rupee_aliases})\.?)"
    minor_units = "|".join(sorted(map(re.escape, _PAISA_INPUT_WORDS), key=len, reverse=True))
    if (bad := re.search(rf"{rupee_prefix}\s*[+-]?[0-9][0-9,.]*\s*(?:{minor_units})(?=$|[\s.,!?;।॥-])", t, re.I)):
        raise NormalizationError("conflicting_currency_unit", "Rupee prefix conflicts with a paise suffix; confirm the unit.", span=bad.span())
    if (bad := re.search(rf"(?:{rupee_prefix}|[$€£]|\b(?:USD|EUR|GBP))\s*[+-]?[0-9][0-9,.]*\s*%", t, re.I)):
        raise NormalizationError("conflicting_currency_unit", "A value cannot have both a currency prefix and percent suffix.", span=bad.span())
    for m in re.finditer(r"(?<![\w:])([0-9]{1,3}):([0-9]{2}):([0-9]{2})(?![\d:])", t):
        if int(m[2]) > 59 or int(m[3]) > 59:
            raise NormalizationError("invalid_duration", "Duration minutes and seconds must be below 60.", span=m.span())
    no_durations = re.sub(r"[0-9]+:[0-9]{2}:[0-9]{2}", lambda m: " " * len(m[0]), t)
    for m in re.finditer(r"(?<![\w:])([0-9]{1,2})(?:[:.]([0-9]{2}))?\s*(a\.?m\.?|p\.?m\.?)(?!\w)|(?<![\w:])([0-9]{1,2}):([0-9]{2})(?![\d:])", no_durations, re.I):
        if m[1]:
            valid = 1 <= int(m[1]) <= 12 and int(m[2] or 0) < 60
        else:
            # Explicit ratio context is not a clock.
            if re.search(r"\b(?:ratio|odds|proportion)\b", _sentence_before(t, m.start()), re.I):
                continue
            valid = int(m[4]) < 24 and int(m[5]) < 60
        if not valid:
            raise NormalizationError("invalid_time", "Invalid clock time; use 00:00-23:59 or 1-12 with am/pm.", span=m.span())

class LangBase:
    # ---- identity -------------------------------------------------------
    code = ""            # ISO 639-1: hi bn ta te mr gu kn ml pa or
    name = ""
    native_digits = ""   # the 10 native digit glyphs "०१२३४५६७८९" etc.

    # ---- numbers --------------------------------------------------------
    ONES = []            # words for 0..99 (100 entries, native script)
    HUNDRED = ""         # word used in "X hundred" composition (Indo-Aryan)
    THOUSAND = ""
    LAKH = ""
    CRORE = ""

    # ---- letters (English letter names written in native script) --------
    LETTERS = {}         # {"A": "...", ..., "Z": "..."}

    # ---- calendar -------------------------------------------------------
    MONTHS = []          # 12 native month names (Jan..Dec, Gregorian)
    MONTH_ALIASES = {}   # accepted alternate input spelling -> month 1..12

    # ---- symbols / unit words -------------------------------------------
    # Values are native-script spoken words. Keys are fixed; the engine
    # relies on every key existing.
    SYM = {
        "rupee": "", "dollar": "", "euro": "", "pound": "", "cent": "",
        "percent": "",          # % symbol
        "point": "",            # decimal point word (Hindi: दशमलव)
        "minus": "", "plus": "", "equals": "", "times": "", "divide": "",
        "squared": "",          # ² (Hindi: का वर्ग — appended after operand)
        "sqrt": "",             # √ (spoken before operand)
        "lt": "", "gt": "",     # < >
        "plus_minus": "",       # ±
        "at": "",               # @  (Hindi: एट द रेट)
        "underscore": "",
        "dot": "",              # . in emails/URLs
        "slash": "",            # / in URLs
        "hash": "",             # #
        "amp": "",              # &
        "half": "", "quarter": "", "three_quarters": "",
        "degree": "", "celsius": "", "fahrenheit": "",
        "minute": "",           # time minutes word
    }

    # ---- templates (use {placeholders}) ----------------------------------
    TIME_ON_HOUR = ""    # "{h} बजे"
    TIME_HM = ""         # "{h} बजकर {m} मिनट"
    RANGE = ""           # "{a} से {b}"          (10–20)
    RANGE_TO = ""        # "{a} से {b} तक"       (time ranges)
    RATIO = ""           # "{a} अनुपात {b}"
    POWER = ""           # "{base} की घात {exp}"
    FRACTION = ""        # "{num} बटा {den}"     (2/7)
    DECADE_SHORT = ""    # "{tens} के"           (90s)
    DECADE_FULL = ""     # "{tens} का दशक"       (1980s)

    # Period words emitted before the time (order: period first, then time).
    PERIODS = {"morning": "", "afternoon": "", "evening": "", "night": ""}

    # ---- source-text detection vocab (native words the engine must
    #      recognise in the INPUT). Include all common variants/spellings.
    TIME_MARKERS = []      # e.g. ["बजे", "बजकर"]
    # markers that carry a case suffix which must be re-attached to the
    # rendered time ("10 గంటలకు" -> "పది గంటలకు", not "పది గంటలు")
    MARKER_SUFFIXES = {}   # e.g. {"గంటలకు": "కు"}
    PERIOD_DETECT = {}     # native word -> "morning"/"afternoon"/"evening"/"night"
    YEAR_WORDS = []        # e.g. ["साल", "वर्ष", "सन्"]
    ID_KEYWORDS = []       # native keywords marking identifiers: नंबर, खाता ...
    ID_AFTER_WORDS = []    # words AFTER a digit-run marking it as an
                           # identifier ("7104 తో ఎండ్ అయ్యే number")
    LABEL_WORDS = []       # native label words read as cardinal: कमरा, अनुच्छेद ...
    MINUTE_WORDS = []      # native "minute" words in input, e.g. ["मिनट"]
    SCALE_WORDS = {}       # native scale word in input -> "crore"/"lakh"/"thousand"
    CURRENCY_WORDS = {}    # native currency words in input -> canonical key
                           # e.g. {"रुपये": "rupee", "रु": "rupee"}

    # ---- ordinals ---------------------------------------------------------
    ORDINAL_EXCEPTIONS = {}   # n -> word (Hindi: {1: "पहला", 2: "दूसरा", ...})
    ORDINAL_SUFFIX = ""       # default suffix appended to cardinal (Hindi: वाँ)
    # Suffixes seen in input that should be preserved on output when possible
    ORDINAL_INPUT_SUFFIXES = []   # e.g. ["वीं", "वाँ", "वें", "वां"]

    # ======================================================================
    # Composition methods — Indo-Aryan defaults; Dravidian modules override.
    # ======================================================================

    def digit(self, d: int) -> str:
        return self.ONES[d]

    def two_digit(self, n: int) -> str:
        return self.ONES[n]

    def hundreds(self, h: int, rest: int) -> str:
        """h in 1..9, rest 0..99 -> 'X sau [rest]'."""
        out = f"{self.ONES[h]} {self.HUNDRED}"
        if rest:
            out += f" {self.two_digit(rest)}"
        return out

    def under_thousand(self, n: int) -> str:
        if n < 100:
            return self.two_digit(n)
        return self.hundreds(n // 100, n % 100)

    def cardinal(self, n: int) -> str:
        """Indian-system cardinal. Override in Dravidian modules."""
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n < 1000:
            return self.under_thousand(n)
        parts = []
        crore, n = divmod(n, 10**7)
        lakh, n = divmod(n, 10**5)
        thousand, n = divmod(n, 1000)
        if crore:
            parts.append(f"{self.cardinal(crore)} {self.CRORE}")
        if lakh:
            parts.append(f"{self.two_digit(lakh)} {self.LAKH}")
        if thousand:
            parts.append(f"{self.two_digit(thousand)} {self.THOUSAND}")
        if n:
            parts.append(self.under_thousand(n))
        return " ".join(parts)

    def year(self, y: int) -> str:
        """Year-style reading. Indo-Aryan: 1947 -> 'unnis sau saintalis'.
        Dravidian modules usually override to full cardinal."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            out = f"{self.two_digit(hi)} {self.HUNDRED}"
            if lo:
                out += f" {self.two_digit(lo)}"
            return out
        return self.cardinal(y)

    def letter(self, c: str) -> str:
        return self.LETTERS.get(c.upper(), c)

    def month(self, i: int) -> str:
        """i is 1..12"""
        return self.MONTHS[i - 1]

    def time_phrase(self, h: int, m: int, period_key=None) -> str:
        # _resolve_time may deliberately hand back a 24-hour hour (languages
        # that transcribe am/pm keep "14:30" as "fourteen thirty"); only fold
        # into 12-hour when it has not already been decided upstream.
        if self.AMPM_WORDS and h > 12:
            h12 = h
        else:
            h12 = h % 12
            if h12 == 0:
                h12 = 12
        if m == 0:
            core = self.TIME_ON_HOUR.format(h=self.two_digit(h12))
        else:
            core = self.TIME_HM.format(h=self.two_digit(h12), m=self.two_digit(m))
        if period_key:
            return f"{self.PERIODS[period_key]} {core}"
        return core

    def ordinal(self, n: int, src_suffix: str = "") -> str:
        if n in self.ORDINAL_EXCEPTIONS and not src_suffix:
            return self.ORDINAL_EXCEPTIONS[n]
        return self._ordinal_compose(n, src_suffix)

    def _ordinal_compose(self, n: int, src_suffix: str) -> str:
        """Default: cardinal + suffix. Languages with stem changes override."""
        base = self.ORDINAL_EXCEPTIONS.get(n)
        if base and not src_suffix:
            return base
        suffix = src_suffix if src_suffix in self.ORDINAL_INPUT_SUFFIXES else self.ORDINAL_SUFFIX
        return f"{self.cardinal(n)}{suffix}"

    def decimal(self, int_part: int, frac_digits: str) -> str:
        frac = " ".join(self.digit(int(d)) for d in frac_digits)
        return f"{self.cardinal(int_part)} {self.SYM['point']} {frac}"

    def decade(self, tens: int, full: bool) -> str:
        """tens: e.g. 90 for '90s', 80 for '1980s'."""
        tpl = self.DECADE_FULL if full else self.DECADE_SHORT
        return tpl.format(tens=self.two_digit(tens))

    # ------------------------------------------------------------------
    # Grammar hooks — no-ops by default; languages with case/sandhi
    # morphology (Telugu, ...) override these.
    # ------------------------------------------------------------------

    # unit initialisms that read as full native words instead of letters
    # (e.g. Telugu KM -> కిలోమీటర్లు)
    UNIT_WORDS = {}

    # written abbreviations that are not speakable words — expanded to native
    # words wherever they appear (mins -> నిమిషాలు). Keys uppercased.
    ABBREV_UNITS = {}

    # spoken word for the "No." / "no." abbreviation before digits
    NUMBER_WORD = ""

    # When non-empty the language transcribes the am/pm token literally and the
    # engine stops INFERRING a period from the hour. Indo-Aryan languages say
    # the period naturally ("shaam paanch baje"), so they leave this empty;
    # English must not have "in the evening" inserted into text that only said
    # "pm". Keys are lowercase, dots stripped.
    AMPM_WORDS = {}
    COUNT_ONE = None
    SINGULAR_NOUNS = {}

    def join_amount(self, words: str, noun: str) -> str:
        """Join number words with a following noun we insert (currency,
        scale, unit). Languages may apply oblique/singular morphology."""
        if words == self.ONES[1]:
            return f"{self.COUNT_ONE or words} {self.SINGULAR_NOUNS.get(noun, noun)}"
        return f"{words} {noun}"

    def scale_noun(self, canon: str, numstr: str) -> str:
        """Scale word used in '₹3 crore'-type phrases; languages may
        pluralize when the number is not 1."""
        return {"crore": self.CRORE, "lakh": self.LAKH,
                "thousand": self.THOUSAND}.get(canon, canon)

    def attach_postposition(self, rep: str, tok: str):
        """Called when a native token `tok` immediately follows a rendered
        span `rep`. Return the merged string (consuming the token), or None
        to leave both untouched."""
        return None



# ----------------------------------------------------------------------------
# Static tables
# ----------------------------------------------------------------------------

# All Indic decimal digit blocks -> ASCII
_INDIC_DIGIT_BLOCKS = [
    0x0966,  # Devanagari
    0x09E6,  # Bengali
    0x0A66,  # Gurmukhi
    0x0AE6,  # Gujarati
    0x0B66,  # Odia
    0x0BE6,  # Tamil
    0x0C66,  # Telugu
    0x0CE6,  # Kannada
    0x0D66,  # Malayalam
]
_DIGIT_TRANS = {}
for _base in _INDIC_DIGIT_BLOCKS:
    for _i in range(10):
        _DIGIT_TRANS[_base + _i] = ord("0") + _i

# Emoji / decorative ranges to strip entirely
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # symbols & pictographs, emoticons, transport...
    "\U00002700-\U000027BF"   # dingbats
    "\U00002600-\U000026FF"   # misc symbols
    "\U0001F1E6-\U0001F1FF"   # flags
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0001F000-\U0001F0FF"
    "\U00002B00-\U00002BFF"   # arrows/stars (⭐)
    "\U00002300-\U000023FF"   # misc technical (⌛ ⏳ ⏬)
    "\U00002190-\U000021FF"   # arrows (→)
    "\U000025A0-\U000025FF"   # geometric (● ▶)
    "\U00002022"              # bullet •
    "\U00002605\U00002606"    # ★ ☆
    "\U00002122\U000000A9\U000000AE"  # ™ © ®
    # NOTE: ZWNJ/ZWJ (200C/200D) are deliberately NOT stripped — they are
    # meaningful orthography in Telugu/Malayalam (ఆన్‌లైన్). Only the truly
    # invisible ZWSP/BOM are removed, without inserting a space.
    "`~|\\\\{}\\[\\]«»‹›„‚•·¤§†‡"
    "]+"
)
_ZW_STRIP = {0x200B: None, 0x200E: None, 0x200F: None, 0xFEFF: None}

_ENGLISH_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Established acronyms that stay in Latin script (never spelled out unless
# they are part of an alphanumeric identifier).
_ACRONYM_WHITELIST = {
    "UPI", "EMI", "API", "MRI", "GST", "SEBI", "MCX", "IGI", "HDFC", "ISRO",
    "IMD", "SBI", "RBI", "ICICI", "IRCTC", "ATM", "SIM", "OTP", "PIN", "CEO",
    "CFO", "CTO", "IPL", "BCCI", "DNA", "RNA", "WHO", "NASA", "PDF", "GPS",
    "LED", "LCD", "USB", "WIFI", "AI", "IT", "TV", "FM", "AC", "EV", "CBI",
    "NIA", "ED", "GDP", "CPI", "FII", "DII", "IPO", "NSE", "BSE", "KYC",
    "NEFT", "RTGS", "IMPS", "IFSC", "CVV", "PAN", "UID", "NRI", "MLA", "MP",
    "PM", "CM", "DM", "SP", "IAS", "IPS", "NEET", "JEE", "UPSC", "CBSE",
    "ICSE", "BA", "MA", "MBA", "MBBS", "AIIMS", "ISKCON", "DLF", "TCS",
    "LTIMINDTREE", "ONGC", "NTPC", "BHEL", "SAIL", "GAIL", "LIC",
    "RAM", "ROM", "CPU", "GPU", "SSD", "HDD", "GPA", "SUV", "SLA", "GDP",
    "UIDAI", "FASTAG", "GSTIN", "OK", "US", "UK", "UAE", "USA", "EU", "UN",
    "PSU", "FIR", "RTI", "PIL", "GPRS", "HD", "UHD", "CCTV", "AQI", "UV",
}

# Units read letter-by-letter (initialisms). Keys uppercased for lookup.
_UNIT_INITIALISMS = {
    "GB", "MB", "KB", "TB", "PB", "MAH", "MPH", "KMPH", "MHZ", "GHZ", "KHZ",
    "HZ", "KWH", "RPM", "HP", "BHP", "CC", "DPI", "PPI", "FPS", "MBPS",
    "GBPS", "KBPS", "PSI", "BPM", "V", "KV", "W", "KW", "MW", "MA", "AH",
    "WH", "DB", "PPM", "ML", "MG", "KG", "KM", "CM", "MM", "GM", "MCG", "IU",
}

# English "big number" scale words that stay in Latin (5 million -> पाँच million)
_EN_SCALES = {"million", "billion", "trillion", "mn", "bn", "millions",
              "billions"}

_ID_KEYWORDS_EN = {
    "phone", "mobile", "card", "account", "otp", "pin", "code", "order",
    "ref", "reference", "train", "flight", "ticket", "tracking", "license",
    "licence", "booking", "pnr", "upi", "imei", "registration", "number",
    "no", "ending", "khasra", "plot", "docket", "complaint", "case", "loan",
    "policy", "invoice", "receipt", "transaction", "txn", "utr", "id",
    "extension", "helpline", "hotline", "whatsapp", "aadhaar", "aadhar",
    "passport", "cvv", "gstin",
}

_LABEL_WORDS_EN = {
    "hall", "window", "counter", "room", "ward", "terminal", "zone", "gate",
    "section", "tower", "platform", "floor", "seat", "berth", "coach",
    "season", "episode", "chapter", "class", "grade", "level", "article",
    "position", "page", "question", "step", "stage", "round", "phase",
    "wing", "block", "lane", "screen", "stand", "row", "table", "lesson",
    "unit", "part", "volume", "rank", "standard", "semester", "waitlist",
}

_STREET_WORDS = {
    "मार्ग", "रोड", "नगर", "सड़क", "गली", "चौक",
    "road", "street", "lane", "avenue", "marg", "nagar", "chowk", "colony",
}

_CURRENCY_SYMBOLS = {"₹": "rupee", "$": "dollar", "€": "euro", "£": "pound",
                     "Rs.": "rupee", "Rs": "rupee"}

_MATH_WORDS_EN = {"squared", "cubed", "plus", "minus", "times", "equals",
                  "power", "root", "over", "by", "divided", "into"}


def _fmt(template, **kw):
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


class Normalizer:
    def __init__(self, lang, *, date_order="DMY", two_digit_year_base=None):
        self.lang = lang
        self.date_order = date_order
        self.two_digit_year_base = two_digit_year_base
        self._build_regexes()

    # ------------------------------------------------------------------
    # Regex construction (depends on language vocab)
    # ------------------------------------------------------------------
    def _build_regexes(self):
        L = self.lang

        def alt(words):
            return "|".join(re.escape(w) for w in
                            sorted(set(words), key=len, reverse=True)) or r"(?!x)x"

        self._period_re = alt(list(L.PERIOD_DETECT.keys()))
        self._marker_re = alt(L.TIME_MARKERS)
        self._month_lookup = _month_lookup(L)
        self._month_re = alt(self._month_lookup)
        self._scale_re = alt(list(L.SCALE_WORDS.keys()) +
                             ["crore", "crores", "Crore", "Crores", "Cr", "cr",
                              "lakh", "lakhs", "Lakh", "Lakhs", "thousand"] +
                             sorted(_EN_SCALES))
        self._ord_suffix_re = alt(list(L.ORDINAL_INPUT_SUFFIXES) +
                                  ["st", "nd", "rd", "th"])
        native_currency = [word for word, key in L.CURRENCY_WORDS.items()
                           if not word.isascii() and key in {"rupee", "dollar", "euro", "pound"}]
        self._currency_re = r"₹|\$|€|£|Rs\.?|INR|USD|EUR|GBP"
        if native_currency:
            self._currency_re += "|(?:" + alt(native_currency) + r")\.?"
        self._id_keywords = _ID_KEYWORDS_EN | _OTP_INPUT_WORDS | {w.lower() for w in L.ID_KEYWORDS}
        self._label_words = _LABEL_WORDS_EN | {w.lower() for w in L.LABEL_WORDS}
        self._minute_re = alt(L.MINUTE_WORDS + ["minute", "minutes", "min"])
        # ratio detection keywords: english words + the language's own ratio
        # word extracted from its RATIO template ("{a} अनुपात {b}" -> अनुपात),
        # so "விகிதம் 9:15" reads as a ratio, not a time
        tpl_words = re.sub(r"\{[ab]\}", " ", L.RATIO or "").split()
        self._ratio_kws = {"ratio", "odds", "proportion", "अनुपात"} | \
            {w.lower() for w in tpl_words}

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _month_index(self, token):
        return self._month_lookup.get(token.strip().lower())

    def _digits_spelled(self, digits, comma=True):
        if os.environ.get("FR3_TN_LONG_DIGITS", "0") == "1" and len(digits) >= 8:
            if __package__:
                from .identifier_rules import group_digits
            else:
                from identifier_rules import group_digits
            return group_digits(digits, self.lang)
        words = [self.lang.digit(int(d)) for d in digits]
        return ", ".join(words) if comma else " ".join(words)

    def _letters_spelled(self, letters, comma=False):
        words = [self.lang.letter(c) for c in letters]
        return ", ".join(words) if comma else " ".join(words)

    def _spell_alnum(self, token, comma):
        """Spell a code segment: letters by sound, digits one by one."""
        parts = []
        for ch in token:
            if ch.isdigit():
                parts.append(self.lang.digit(int(ch)))
            elif ch.isalpha() and ch.isascii():
                parts.append(self.lang.letter(ch))
            # other chars (hyphens) become separators, dropped
        return (", " if comma else " ").join(parts)

    def _spell_model_segment(self, seg):
        """WH-1000XM5 style segment: letter runs by sound, digit runs as
        cardinal when a clean round block, else digit-by-digit; spaces."""
        out = []
        for run in re.findall(r"[A-Za-z]+|\d+", seg):
            if run.isdigit():
                n = int(run)
                if len(run) == 1 or (n % 100 == 0 and n >= 100):
                    out.append(self.lang.cardinal(n))
                else:
                    out.append(self._digits_spelled(run, comma=False))
            else:
                out.append(self._letters_spelled(run))
        return " ".join(out)

    def _number_words(self, numstr):
        """Plain number string (commas allowed) -> cardinal / decimal words."""
        if numstr.startswith(("-", "+")):
            sign = self.lang.SYM["minus" if numstr[0] == "-" else "plus"]
            return f"{sign} {self._number_words(numstr[1:])}"
        numstr = numstr.replace(",", "")
        if "." in numstr:
            ip, fp = numstr.split(".", 1)
            return self.lang.decimal(int(ip or 0), fp)
        return self.lang.cardinal(int(numstr))

    def _keyword_before(self, text, start, keywords, window=3):
        before = _sentence_before(text, start).replace("\u200c", "")
        toks = re.findall(r"[\w#ऀ-෿']+", before.lower())
        for tok in toks[-window:]:
            tok = tok.strip("#'")
            if tok in keywords:
                return True
        return False

    def _period_for_hour24(self, h):
        if 4 <= h < 12:
            return "morning"
        if 12 <= h < 16:
            return "afternoon"
        if 16 <= h < 20:
            return "evening"
        return "night"

    def _time_words(self, h, m, period_key):
        if self.lang.code != "en" and (h == 0 or h > 12):
            # Keep the written 24-hour hour. The inherited language time methods
            # otherwise fold zero into twelve or fourteen into two.
            template = self.lang.TIME_HM if m else self.lang.TIME_ON_HOUR
            return template.format(h=self.lang.cardinal(h), m=self.lang.cardinal(m))
        return self.lang.time_phrase(h, m, period_key)

    # ------------------------------------------------------------------
    # Pre-clean
    # ------------------------------------------------------------------
    def _preclean(self, text):
        text = _ascii_digits(text)
        text = text.replace("−", "-")
        text = text.translate(_ZW_STRIP)
        # double quotation marks (incl. CSV-style "" and """ runs) carry no
        # sound — a voice model cannot speak them.
        text = re.sub(r'["“”„]+', "", text)
        # single quotes wrapping phrases ('Premium Home Decor') go too.
        # Kept: apostrophes inside words (D'Souza, John's) and quoted single
        # letters ('Y' -> letter name, handled by its own matcher).
        text = re.sub(r"(?<=[\s(।,—–-])['‘’](?![A-Za-z]['‘’])(?=\S)", "", text)
        text = re.sub(r"^['‘’](?![A-Za-z]['‘’])(?=\S)", "", text)
        text = re.sub(r"(?<=\S)(?<!['‘’][A-Za-z])['‘’](?=[\s,।.!?;:)]|$)", "", text)
        text = re.sub(r"#{2,}", " ", text)   # markdown header/### junk
        # ellipses (… / ... / ..) are a spoken pause -> read as a comma
        text = re.sub(r"\s*(?:…|‥|\.{2,})\s*", ", ", text)
        # parentheses are silent; the aside becomes a comma pause
        text = re.sub(r"\s*[()]\s*", ", ", text)
        # //comment// markers are markup junk ((?<!:) keeps https:// intact)
        text = re.sub(r"(?<!:)/{2,}", " ", text)
        # "No." / "no." before digits is the number abbreviation, not "no"
        if self.lang.NUMBER_WORD:
            text = re.sub(r"\b[Nn][Oo]s?\.\s*(?=\d)",
                          self.lang.NUMBER_WORD + " ", text)
        # written-only unit abbreviations -> native words (mins -> నిమిషాలు);
        # guards keep code segments like MIN-22 intact
        for ab, word in self.lang.ABBREV_UNITS.items():
            # consume the abbreviation's own full stop when the sentence
            # continues ("Dr. Rao" -> "doctor Rao"), but keep it when the
            # abbreviation ends the sentence ("... approx." -> "approximately.")
            text = re.sub(rf"(?<![\w-]){ab}(?:\.(?=\s+\S))?(?![\w-])", word, text,
                          flags=re.IGNORECASE)
        text = re.sub(r"^[,\s]+", "", text)
        # Comparison operators are meaningful even without a left numeric operand.
        text = _EMOJI_RE.sub(" ", text)
        # en/em dash between numbers -> canonical range dash; others deleted
        text = re.sub(r"(?<=[\d])\s*[–—]\s*(?=[\d])", "–", text)
        # time-range dash: "11 AM – 3 PM" (dash not directly digit-adjacent)
        text = re.sub(r"\s+[–—]\s+(?=\d)", " – ", text)
        text = re.sub(r"[—]", " ", text)
        text = re.sub(r"(?<![\d\s])[–](?![\d\s])", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    # ------------------------------------------------------------------
    # Matchers — each returns list of (start, end, replacement)
    # Priority = order in self._matchers (earlier wins).
    # ------------------------------------------------------------------
    def _gather(self, text):
        matches = []  # (priority, start, end, replacement)
        matchers = [
            self._m_email,
            self._m_url,
            self._m_handle,
            self._m_comparison,
            self._m_time_range,
            self._m_duration,
            self._m_time_verbalized,
            self._m_time,
            self._m_tamil_date,
            self._m_date,
            self._m_currency_range,
            self._m_currency,
            self._m_bare_rupee_suffix,
            self._m_long_digit_runs,
            self._m_amount_context,
            self._m_percent_range,
            self._m_percent,
            self._m_negative,
            self._m_designation,
            self._m_vehicle_plate,
            self._m_ordinal,
            self._m_code,
            self._m_id_digits,
            self._m_chem,
            self._m_math,
            self._m_num_range,
            self._m_dimensions,
            self._m_roman,
            self._m_unit_initialism,
            self._m_kmb_suffix,
            self._m_fraction,
            self._m_ratio,
            self._m_label,
            self._m_acronym_model,
            self._m_num_hyphen_word,
            self._m_decade,
            self._m_year,
            self._m_quoted_letter,
            self._m_short_mixed_token,
            self._m_decimal,
            self._m_int,
        ]
        for prio, fn in enumerate(matchers):
            for (s, e, rep) in fn(text):
                matches.append((prio, s, e, rep))
        return matches

    # --- email / url / handle -----------------------------------------
    def _spell_uri_part(self, part):
        """Word chunks stay; digits spelled; symbols to words."""
        if os.environ.get("FR3_TN_URL_SPELL", "0") == "1":
            if __package__:
                from .identifier_rules import spell_uri
            else:
                from identifier_rules import spell_uri
            return spell_uri(part, self.lang, self._digits_spelled)
        S = self.lang.SYM
        out = []
        for run in re.findall(r"[^\W\d_]+|\d+|[^\w\s]|_", part):
            if run.isdigit():
                out.append(self._digits_spelled(run, comma=False))
            elif run == ".":
                out.append(S["dot"])
            elif run == "_":
                out.append(S["underscore"])
            elif run == "@":
                out.append(S["at"])
            elif run == "/":
                out.append(S["slash"])
            elif run == "-":
                out.append("dash")
            elif run == "+":
                out.append(S["plus"])
            elif run in {"?", "=", "&", "#", "%", ":", "~"}:
                out.append({"?": "question mark", "=": "equals", "&": "ampersand",
                            "#": "hash", "%": "percent", ":": "colon", "~": "tilde"}[run])
            elif run.lower() == "www":
                out.append(self._letters_spelled("WWW"))
            elif len(run) == 1 or run.upper() in self.lang.ABBREV_UNITS:
                # Do not expand /Dr/ to /doctor/ on a second normalization pass.
                out.append(self._letters_spelled(run.upper()))
            else:
                out.append(run)
        return " ".join(w for w in out if w)

    def _m_email(self, text):
        for m in re.finditer(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", text):
            yield m.start(), m.end(), self._spell_uri_part(m.group(0))

    def _m_url(self, text):
        pat = (r"https?://[^\s<>]+|www\.[^\s<>]+"
               # bare domains: maya.ai, irctc.co.in/booking
               r"|\b[A-Za-z][\w\-]*\.(?:com|org|net|ai|io|co|dev|app|edu|gov|in)"
               r"(?:\.(?:in|uk|us))?(?:[/?#][^\s<>]*)?(?![\w.])")
        for m in re.finditer(pat, text):
            token = m.group(0).rstrip(".,!?;)")
            body = re.sub(r"^https?://", "", token)
            yield m.start(), m.start() + len(token), self._spell_uri_part(body)

    def _m_handle(self, text):
        # trailing \w so a sentence-final period after the handle is not
        # swallowed and spoken as "dot" ("@test_user." -> handle + full stop)
        pat = r"(?<![\w.])@([A-Za-z](?:[\w.\-]*\w)?)" if os.environ.get("FR3_TN_URL_SPELL", "0") == "1" else r"(?<![\w.])@([A-Za-z](?:[\w.]*\w)?)"
        for m in re.finditer(pat, text):
            yield m.start(), m.end(), \
                f"{self.lang.SYM['at']} {self._spell_uri_part(m.group(1))}"

    # --- time -----------------------------------------------------------
    def _parse_time_match(self, m, text):
        """Shared logic. Returns (h12, mm, period_key)."""
        pass

    def _time_pattern(self):
        P = self._period_re
        MK = self._marker_re
        # (?<!\w) / (?!\w) around native-word alternations: "ગુજરાત" ends
        # with રાત (night) — without boundaries the period word matches
        # inside other words
        return re.compile(
            rf"(?:(?<!\w)({P})\s+)?"                 # leading period word
            rf"(\d{{1,2}})"                          # hour
            # a real time needs :mm, AM/PM or a time marker right after the
            # hour — otherwise bare numbers spuriously swallow period words
            rf"(?=:\d{{2}}"
            rf"|\.\d{{2}}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)(?![A-Za-z])"   # 6.45 pm — dot form
            rf"|\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)(?![A-Za-z])|\s*(?:{MK})(?!\w))"
            rf"(?:[.:](\d{{2}}))?"                   # :mm  or  .mm (gated by the lookahead)
            rf"(?:\s*(AM|PM|am|pm|a\.m\.|p\.m\.))?"  # am/pm
            rf"(?:\s*({MK})(?!\w))?"                 # बजे / मणि ...
            rf"(?:\s+({P})(?!\w))?", re.I           # trailing period word
        )

    def _resolve_time(self, lead_p, h, mm, ampm, marker, trail_p):
        L = self.lang
        h = int(h)
        m = int(mm) if mm is not None else 0
        period = None
        src_period = lead_p or trail_p
        if src_period:
            period = L.PERIOD_DETECT.get(src_period)
        if L.code != "en" and not ampm and not src_period:
            return h, m, None
        infer = not L.AMPM_WORDS      # languages that speak am/pm keep the token instead
        if ampm:
            a = ampm.lower().replace(".", "")
            h24 = h % 12 + (12 if a == "pm" else 0)
            if infer:
                period = period or self._period_for_hour24(h24)
        elif h > 12 or h == 0:
            # 24-hour clock
            if infer:
                period = period or self._period_for_hour24(h)
        if L.AMPM_WORDS and not ampm and (h > 12 or h == 0):
            return h, m, period      # "14:30" -> "fourteen thirty", not "two thirty"
        h12 = h % 12
        if h12 == 0:
            h12 = 12
        return h12, m, period

    def _m_duration(self, text):
        # H:MM:SS is a duration, not a 12-hour clock. Zero hours must not become 12.
        L = self.lang
        sec_word = L.SYM.get("second", "seconds")
        for m in re.finditer(r"\b(\d{1,3}):(\d{2}):(\d{2})\b", text):
            h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if mm > 59 or ss > 59:
                continue
            hours = L.ABBREV_UNITS.get("HRS", "hours")
            minutes = L.SYM["minute"]
            parts = [L.join_amount(L.cardinal(n), unit if n != 1 or L.code != 'en' else unit.rstrip('s'))
                     for n, unit in ((h, hours), (mm, minutes), (ss, sec_word)) if n]
            yield m.start(), m.end(), " ".join(parts) or L.join_amount(L.cardinal(0), sec_word)

    def _m_time_verbalized(self, text):
        # "9 बजकर 15 मिनट" — already half-verbalized with digits: rebuild the
        # canonical phrase instead of reading the pieces separately
        L = self.lang
        MIN = self._minute_re
        pat = re.compile(
            rf"\b(\d{{1,2}})\s*(?:{self._marker_re})\s+(\d{{1,2}})\s*({MIN})(?!\w)")
        for m in pat.finditer(text):
            h, mm = int(m.group(1)), int(m.group(2))
            if h > 23 or mm > 59:
                continue
            rep = self._time_words(h, mm, None)
            sfx = L.MARKER_SUFFIXES.get(m.group(3) or "")
            if sfx:
                rep = L.attach_postposition(rep, sfx) or rep
            yield m.start(), m.end(), rep

    def _m_time(self, text):
        pat = self._time_pattern()
        for m in pat.finditer(text):
            lead_p, h, mm, ampm, marker, trail_p = m.groups()
            if h is None:
                continue
            # must look like a time: needs marker, am/pm, period word, or a
            # valid HH:MM
            hh = int(h)
            is_hhmm = mm is not None and int(mm) <= 59 and hh < 24
            if not (ampm or marker or ((lead_p or trail_p) and (is_hhmm or mm is None))):
                if not (is_hhmm and mm is not None and hh > 12):
                    # bare h:mm with hh<=12: still a time if mm is 2-digit <=59
                    if not (is_hhmm and not self._keyword_before(
                            text, m.start(), self._ratio_kws, window=4)):
                        continue
            if mm is None and not (ampm or marker):
                continue  # bare number is not a time
            if mm is not None and int(mm) > 59:
                continue
            if hh > 23:
                continue
            h12, mnt, period = self._resolve_time(lead_p, h, mm, ampm, marker, trail_p)
            src_period = lead_p or trail_p
            if src_period:
                rep = f"{src_period} {self.lang.time_phrase(h12, mnt, None)}"
            else:
                rep = self._time_words(h12, mnt, period)
            if ampm and self.lang.AMPM_WORDS:
                w = self.lang.AMPM_WORDS.get(ampm.lower().replace(".", ""))
                if w:
                    # a bare hour with am/pm reads "eight p m", never
                    # "eight o'clock p m" -- o'clock was not in the input
                    if mnt == 0:
                        rep = self.lang.two_digit(h12)
                    rep = f"{rep} {w}"
            sfx = self.lang.MARKER_SUFFIXES.get(marker or "")
            if sfx:
                rep = self.lang.attach_postposition(rep, sfx) or rep
            yield m.start(), m.end(), rep

    def _m_time_range(self, text):
        # 11 AM – 3 PM  /  9:00 – 17:00  / 11 AM to 3 PM stays "to"
        pat = self._time_pattern()
        rng = re.compile(
            rf"({pat.pattern})\s*[–\-]\s*({pat.pattern})", pat.flags)
        for m in rng.finditer(text):
            g = m.groups()
            left, right = g[0], g[7]
            gl, gr = g[1:7], g[8:14]
            if gl[1] is None or gr[1] is None:
                continue
            # each side must independently look like a time
            if not ((gl[3] or gl[4] or gl[2]) and (gr[3] or gr[4] or gr[2])):
                continue
            h1, m1, p1 = self._resolve_time(*gl)
            h2, m2, p2 = self._resolve_time(*gr)
            a = self._time_words(h1, m1, p1)
            b = self._time_words(h2, m2, p2)
            if self.lang.AMPM_WORDS:
                if gl[3]:
                    a = (self.lang.two_digit(h1) if m1 == 0 else a) + " " + self.lang.AMPM_WORDS[gl[3].lower().replace('.', '')]
                if gr[3]:
                    b = (self.lang.two_digit(h2) if m2 == 0 else b) + " " + self.lang.AMPM_WORDS[gr[3].lower().replace('.', '')]
            rep = _fmt(self.lang.RANGE_TO, a=a, b=b)
            tail = self.lang.RANGE_TO.split("{b}")[-1].strip()
            if tail and re.match(rf"\s*{re.escape(tail)}(?!\w)",
                                 text[m.end():]):
                rep = rep[: -len(tail)].rstrip()  # source already says it
            yield m.start(), m.end(), rep

    # --- dates -----------------------------------------------------------
    def _m_tamil_date(self, text):
        if self.lang.code not in ("en", "ta"):
            return
        months = "|".join(_TAMIL_MONTHS)
        # Only calendar-day suffixes. Do not pretend all Tamil number morphology
        # can be repaired by concatenating an English cardinal with a suffix.
        pat = rf"(?<!\w)({months})\s+(\d{{1,2}})(?:-?(?:ஆம்|ம்))?(?![\dA-Za-z])(?:\s*தேதி)?"
        for m in re.finditer(pat, text):
            d = int(m[2])
            if 1 <= d <= 31:
                yield m.start(), m.end(), f"{m[1]} {_TAMIL_ORDINALS[d]} தேதி"
        for m in re.finditer(r"(?<!\w)(\d+)-?(?:ஆம்|ம்)(?![\w\u0B80-\u0BFF])", text):
            d = int(m[1])
            if not 1 <= d <= 31:
                if self.lang.code == "en":
                    raise NormalizationError("unsupported_tamil_ordinal", "Select Tamil normalization for ordinals outside calendar days 1-31.", span=m.span(), source=text)
                yield m.start(), m.end(), self.lang.ordinal(d, "ஆம்")
            else:
                yield m.start(), m.end(), _TAMIL_ORDINALS[d]

    def _date_words(self, day, month, year=None, ordinal=False, month_first=False):
        L = self.lang
        if L.code == "ta":
            out = f"{L.month(month)} {_TAMIL_ORDINALS[day]} தேதி"
        else:
            suffix = ordinal if isinstance(ordinal, str) and ordinal in L.ORDINAL_INPUT_SUFFIXES else ""
            day_words = L.ordinal(day, suffix) if ordinal else L.cardinal(day)
            out = f"{L.month(month)} {day_words}" if month_first and L.code != "en" else f"{day_words} {L.month(month)}"
        return f"{out} {L.year(year)}" if year is not None else out

    def _m_date(self, text):
        L = self.lang
        for m in re.finditer(rf"\b(\d{{1,2}})[-–](\d{{1,2}})\s+({self._month_re})(?!\w)(?:\s+(\d{{4}}))?", text, re.I):
            mo = self._month_index(m[3])
            for day in (m[1], m[2]):
                _calendar_check(int(day), mo, int(m[4]) if m[4] else 2000)
            days = _fmt(L.RANGE, a=L.cardinal(int(m[1])), b=L.cardinal(int(m[2])))
            out = f"{days} {L.month(mo)}"
            if m[4]:
                out += f" {L.year(int(m[4]))}"
            yield m.start(), m.end(), out
        for m in re.finditer(r"(?<![\w-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])", text):
            yield m.start(), m.end(), self._date_words(int(m[3]), int(m[2]), int(m[1]))
        # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
        for m in re.finditer(r"\b(\d{1,2})([/.\-])(\d{1,2})\2(\d{2,4})\b(?![/A-Za-z-])", text):
            d, mo, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
            if self.date_order == "MDY":
                d, mo = mo, d
            if not (1 <= d <= 31 and 1 <= mo <= 12):
                continue
            if len(m.group(4)) == 2:
                y += self.two_digit_year_base
            yield m.start(), m.end(), self._date_words(d, mo, y)
        # DD-Mon-YYYY / DD Mon YYYY / DD Mon, with native or English months
        # (?!\w) after the month: without it "25 ಮೇಲೆ" would parse as
        # "25 May" because ಮೇಲೆ starts with ಮೇ (May)
        pat = re.compile(
            rf"\b(\d{{1,2}})(?:-?({self._ord_suffix_re}))?[\-\s]({self._month_re})(?!\w)"
            rf"(?:[\-\s,]+(\d{{4}}))?", re.IGNORECASE)
        for m in pat.finditer(text):
            d = int(m.group(1))
            mo = self._month_index(m.group(3))
            if mo is None or not (1 <= d <= 31):
                continue
            out = self._date_words(d, mo, int(m[4]) if m[4] else None, m[2])
            yield m.start(), m.end(), out
        # Month DD, YYYY (English order)
        pat2 = re.compile(
            rf"\b({self._month_re})\s+(\d{{1,2}})(?:-?({self._ord_suffix_re}))?(?:,?\s+(\d{{4}}))?(?!\d)",
            re.IGNORECASE)
        for m in pat2.finditer(text):
            mo = self._month_index(m.group(1))
            d = int(m.group(2))
            if mo is None or not (1 <= d <= 31):
                continue
            out = self._date_words(d, mo, int(m[4]) if m[4] else None, m[3], month_first=True)
            yield m.start(), m.end(), out

    # --- currency ---------------------------------------------------------
    _CURRENCY_CODE_KEYS = {"₹": "rupee", "$": "dollar", "€": "euro",
                           "£": "pound", "RS": "rupee", "INR": "rupee",
                           "USD": "dollar", "EUR": "euro", "GBP": "pound"}

    def _cur_key(self, symtok):
        t = symtok.strip().rstrip(".").strip().upper()
        return self._CURRENCY_CODE_KEYS.get(t) or self.lang.CURRENCY_WORDS.get(t) or self.lang.CURRENCY_WORDS.get(t + ".") or "rupee"

    def _m_currency_range(self, text):
        # ₹500-700 -> "five hundred to seven hundred rupees"
        L = self.lang
        pat = re.compile(
            rf"(?<!\w)([+-]?)({self._currency_re})"
            rf"\s*([+-]?\d+(?:,\d{{2,3}})*(?:\.\d+)?)\s*[–\-]\s*([+-]?\d+(?:,\d{{2,3}})*(?:\.\d+)?)(?!\d|[.,]\d)", re.I)
        for m in pat.finditer(text):
            if m[1] and m[3].startswith(("-", "+")):
                raise NormalizationError("conflicting_sign", "Currency amount has two signs; confirm the value.", span=m.span(), source=text)
            a, b = self._number_words(m[1] + m[3]), self._number_words(m[4])
            cw = L.SYM[self._cur_key(m[2])]
            rng = _fmt(L.RANGE, a=a, b=L.join_amount(b, cw))
            yield m.start(), m.end(), rng

    def _m_bare_rupee_suffix(self, text):
        L = self.lang
        for m in re.finditer(r"(?<![\w₹.])(\d+(?:,\d{2,3})*(?:\.\d+)?)\s?/-", text):
            num = m.group(1).replace(",", "")
            words = self._number_words(num)
            yield m.start(), m.end(), L.join_amount(words, L.SYM["rupee"])

    def _m_currency(self, text):
        L = self.lang
        S = L.SYM
        pat = re.compile(
            rf"(?<!\d)([+-]?)\s*(₹|\$|€|£|(?<!\w)(?:{self._currency_re}))"
            rf"\s*([+-]?)(\d+(?:,\d{{2,3}})*(?:\.\d+)?)"
            rf"(?:\s*({self._scale_re})(?:((?:ां|ं)?(?:चा|ची|चे|च्या|त|ना|हून|पर्यंत|मध्ये))|(?![ऀ-ःऺ-ॏঀ-ঃা-্ਁ-ਃਾ-੍ઁ-ઃા-્ଁ-ଃା-୍ஂா-்ఀ-ఄా-ౌಁ-ಃಾ-ೌഀ-ഃാ-ൌ]))(?![A-Za-z0-9]))?(?:\s?/-)?", re.I)
        for m in pat.finditer(text):
            if m[1] and m[3]:
                raise NormalizationError("conflicting_sign", "Currency amount has two signs; confirm the value.", span=m.span(), source=text)
            key = self._cur_key(m[2])
            num = m[4].replace(",", "")
            scale = m[5]
            cur_word = S[key]
            # money with exactly 2 decimals and no scale word -> main + sub
            # unit ($84.20 -> 84 dollar 20 cent; ₹95.41 -> 95 rupees 41 paise)
            if "." in num and not scale and len(num.split(".", 1)[1]) == 2:
                ip, fp = num.split(".", 1)
                main = L.cardinal(int(ip or 0))
                if key == "rupee" and L.SYM.get("paisa"):
                    sub_word = L.SYM["paisa"]
                else:
                    sub_word = S["cent"] if key != "rupee" else None
                if fp.strip("0") == "":
                    out = L.join_amount(main, cur_word)
                elif sub_word:
                    out = (f"{L.join_amount(main, cur_word)} "
                           f"{L.join_amount(L.cardinal(int(fp)), sub_word)}")
                else:
                    out = L.join_amount(L.decimal(int(ip or 0), fp), cur_word)
            else:
                words = self._number_words(num)
                if scale:
                    sl = scale.lower()
                    canon = L.SCALE_WORDS.get(scale)
                    if canon is None and sl in _EN_SCALES:
                        scale_word = scale  # 1.5 million -> ... million dollar
                    else:
                        if canon is None:
                            canon = ("crore" if sl in ("cr",) or "crore" in sl
                                     else "lakh" if "lakh" in sl
                                     else "thousand")
                        scale_word = L.scale_noun(canon, num)
                    out = L.join_amount(L.join_amount(words, scale_word),
                                        cur_word)
                    case = m[6]
                    if case:
                        out = L.attach_postposition(out, case) or out
                else:
                    out = L.join_amount(words, cur_word)
            sign = m[1] or m[3]
            if sign:
                out = f"{S['minus'] if sign == '-' else S['plus']} {out}"
            # Keep whitespace outside the matched amount.
            space = m[0][:1].isspace() or (m.start() > 0 and text[m.start() - 1].isalnum())
            yield m.start(), m.end(), (" " if space else "") + out

    def _m_amount_context(self, text):
        # Explicit currency suffix/amount label wins over broad identifier hints.
        units = (r"rupees?|rs\.?|inr|dollars?|usd|euros?|pounds?|paise|paisa|cents?|"
                 r"ரூபாய்|பைசா|रुपये|रुपया|रुपए|రూ\.?|రూపాయలు|രൂപ")
        units += "|" + "|".join(map(re.escape, self.lang.CURRENCY_WORDS))
        for m in re.finditer(rf"(?<![\w.])([+-]?\d[\d,]*(?:\.\d+)?)(?=\s*(?:{units})(?![A-Za-z]))", text, re.I):
            yield m.start(), m.end(), self._number_words(m[1])
        labels = {"balance", "amount", "payable", "outstanding", "payment", "price", "fee", "pay", "emi", "தவணை"}
        for m in re.finditer(r"(?<![\w.,+/-])\d+(?:,\d{2,3})*(?!\w|[.,/-]\d)", text):
            before = _sentence_before(text, m.start()).lower()
            toks = re.findall(r"\w+", before)[-3:]
            if any(t in labels for t in toks) and not any(t in {"number", "id", "code", "otp", "pin"} for t in toks):
                yield m.start(), m.end(), self._number_words(m[0])

    # --- percent ------------------------------------------------------------
    def _m_percent_range(self, text):
        # 40-70% / 40–70% -> "40 to 70 percent"
        L = self.lang
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*[–\-]\s*(\d+(?:\.\d+)?)\s*%",
                             text):
            a = self._number_words(m.group(1))
            b = self._number_words(m.group(2))
            yield m.start(), m.end(), \
                f"{_fmt(L.RANGE, a=a, b=b)} {L.SYM['percent']}"

    def _m_percent(self, text):
        L = self.lang
        for m in re.finditer(r"(-?)(\d+(?:,\d{2,3})*(?:\.\d+)?)\s*%", text):
            words = self._number_words(m.group(2))
            if m.group(1):
                words = f"{L.SYM['minus']} {words}"
            yield m.start(), m.end(), f"{words} {L.SYM['percent']}"

    # --- designations (Sony WH-1000XM5, Kepler-186f, Soyuz T-11) -----------
    def _m_designation(self, text):
        # TitleCase brand fused: Kepler-186f, Salyut-7
        for m in re.finditer(r"\b([A-Z][a-z]{2,})-([A-Za-z0-9]+)\b", text):
            seg = m.group(2)
            if not re.search(r"\d", seg):
                continue
            if self._month_index(seg) or self._month_index(m.group(1)):
                continue
            yield m.start(), m.end(), \
                f"{m.group(1)} {self._spell_model_segment(seg)}"
        # Brand word + short designation: Sony WH-1000XM5, Soyuz T-11
        pat = re.compile(
            r"\b([A-Z][a-z]{2,})\s+([A-Z]{1,4}-[A-Za-z0-9]+)\b")
        for m in pat.finditer(text):
            brand = m.group(1)
            if brand.lower() in self._id_keywords or brand.lower() in self._label_words:
                continue
            seg = m.group(2)
            if not re.search(r"\d", seg):
                continue
            yield (m.start(2), m.end(2), self._spell_model_segment(seg))

    # --- alphanumeric codes -------------------------------------------------
    def _m_code(self, text):
        # hyphenated tokens containing at least one letter and one digit
        pat = re.compile(r"#?\b(?=[A-Za-z0-9/-]*\d)(?=[A-Za-z0-9/-]*[A-Za-z])"
                         r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+(?![A-Za-z0-9])")
        for m in pat.finditer(text):
            tok = m.group(0).lstrip("#")
            # date like 12-Aug-2025 -> handled by date matcher (higher prio)
            parts = re.split(r"[-/]", tok)
            if (any(self._month_index(p) for p in parts)
                    and all(p.isdigit() for p in parts
                            if not self._month_index(p))):
                continue  # date-like 12-Aug-2025; MAY-17Y7 is still a code
            # a lowercase word segment means this is prose, not a code
            # (900-point, 80-watt) -> handled by num-hyphen-word matcher
            if any(re.fullmatch(r"[a-z][a-z]+|[A-Z][a-z]{2,}", p) for p in parts):
                continue  # prose word segment (900-point, 7-East)
            if (len(parts) == 2 and parts[0].isdigit() and parts[1].isalpha()
                    and len(parts[1]) >= 3):
                continue  # 30-DAY, 24-HOUR -> number + word
            # 15-GB -> unit matcher
            if (len(parts) == 2 and parts[0].isdigit()
                    and parts[1].upper() in _UNIT_INITIALISMS):
                continue
            # Hall 3-A / Window 5-B -> label matcher (but CR-2A stays a code)
            if (re.fullmatch(r"\d{1,3}-?[A-Za-z]", tok)
                    and self._keyword_before(text, m.start(),
                                             self._label_words, window=2)):
                continue
            if sum(c.isalnum() for c in tok) <= 2:
                continue
            yield m.start(), m.end(), self._spell_alnum(tok, comma=True)
        # solid mixed tokens (length>=5, >=2 digits): MH12AB8821, 77K2L9
        pat2 = re.compile(r"\b(?=(?:[A-Za-z]*\d){2})(?=[A-Za-z0-9]*[A-Za-z])"
                          r"[A-Za-z0-9]{5,}\b")
        for m in pat2.finditer(text):
            tok = m.group(0)
            if re.fullmatch(r"\d+s", tok):
                continue  # 1980s -> decade matcher
            if re.fullmatch(r"\d+[x×]\d+", tok):
                continue  # 1920x1080 -> dimensions matcher
            m2 = re.fullmatch(r"\d+([A-Za-z]{1,4})", tok)
            if m2 and m2.group(1).upper() in _UNIT_INITIALISMS:
                continue  # 250ml -> unit matcher
            if re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", tok) and self._chem_ok(tok):
                continue  # chemistry wins
            yield m.start(), m.end(), self._spell_alnum(tok, comma=True)

    def _m_vehicle_plate(self, text):
        # spaced Indian plates: AP 09 CL 4421 / TS 09 EA 1234
        L = self.lang
        pat = re.compile(r"\b([A-Z]{2})[ -](\d{1,2})[ -]([A-Z]{1,3})[ -](\d{3,4})\b")
        for m in pat.finditer(text):
            out = []
            for grp in m.groups():
                for ch in grp:
                    out.append(L.digit(int(ch)) if ch.isdigit() else L.letter(ch))
            yield m.start(), m.end(), ", ".join(out)

    # --- identifier digit runs ----------------------------------------------
    def _m_long_digit_runs(self, text):
        if os.environ.get("FR3_TN_LONG_DIGITS", "0") != "1":
            return
        if __package__:
            from .identifier_rules import long_runs
        else:
            from identifier_rules import long_runs
        reserved = [(s, e) for matcher in (self._m_percent_range, self._m_percent, self._m_negative)
                    for s, e, _ in matcher(text)]
        for s, e, spoken in long_runs(text, self.lang):
            if not any(s < end and e > start for start, end in reserved):
                yield s, e, spoken

    def _m_id_digits(self, text):
        L = self.lang
        # hyphen/space separated digit groups, or long runs, or # prefixed
        pat = re.compile(r"(\+?)(?:#?\b\d+(?:[-\s]\d+)+\b|#\s?\d{3,}\b"
                         r"|(?<![\d.])(?<!\d,)\d{7,}(?![.,]?\d))")
        for m in pat.finditer(text):
            has_plus = m.group(1) == "+"
            tok = m.group(0).lstrip("+# ")
            groups = re.split(r"[-\s]", tok)
            digits = "".join(groups)
            is_id = False
            if has_plus and len(digits) >= 8:
                is_id = True  # +91 ... country-code phone
            elif m.group(0).startswith("#"):
                is_id = True
            elif len(digits) >= 7 and " " not in tok:
                is_id = True
            elif any(len(g) >= 4 for g in groups) and len(groups) >= 2 and "-" in tok:
                is_id = True
            elif len(groups) >= 3 and "-" in tok:
                is_id = True
            elif self._keyword_before(text, m.start(), self._id_keywords):
                is_id = True
            else:
                a_toks = re.findall(r"[\w#ऀ-෿']+",
                                    _sentence_after(text, m.end()).lower())[:4]
                if any(t in self._id_keywords for t in a_toks):
                    is_id = True
            if not is_id:
                continue
            # ranges like 10-20 without keyword: ascending small pairs -> skip
            a_toks2 = re.findall(r"[\w#ऀ-෿']+",
                                 _sentence_after(text, m.end()).lower())[:4]
            if ("-" in tok and len(groups) == 2
                    and all(len(g) <= 3 for g in groups)
                    and int(groups[0]) < int(groups[1])
                    and not self._keyword_before(text, m.start(), self._id_keywords)
                    and not any(t in self._id_keywords for t in a_toks2)):
                continue
            rep = self._digits_spelled(digits, comma=True)
            if has_plus:
                rep = f"{L.SYM['plus']} {rep}"
            yield m.start(), m.end(), rep
        # keyword-driven short runs: OTP 4471, code 22377, train 207,
        # "ending in 7104"
        for m in re.finditer(r"(?<![\d.])(?<!\d,)\d{3,6}(?![.,]?\d)(?!s\b)", text):
            # "139 dial करें" — dial/डायल shortly after marks a hotline number
            after_toks = re.findall(r"[\w#ऀ-෿']+",
                                    _sentence_after(text, m.end()).lower())[:6]
            id_after = {"dial", "डायल"} | set(self.lang.ID_AFTER_WORDS)
            if any(t in id_after for t in after_toks):
                yield m.start(), m.end(), \
                    self._digits_spelled(m.group(0), comma=True)
                continue
            # an identifier keyword within ~4 tokens AFTER also marks it
            # ("4321 हा OTP", "998877 वापरा password") — but not label/year
            if (any(t in self._id_keywords for t in after_toks[:4])
                    and not self._keyword_before(text, m.start(), self._label_words, window=2)
                    and not self._keyword_before(text, m.start(),
                             {w.lower() for w in L.YEAR_WORDS} | {"year"}, window=2)):
                yield m.start(), m.end(), \
                    self._digits_spelled(m.group(0), comma=True)
                continue
            if self._keyword_before(text, m.start(), self._id_keywords):
                # label words override (Room 314 stays cardinal)
                if self._keyword_before(text, m.start(), self._label_words, window=2):
                    continue
                # year words override (साल 2026)
                if self._keyword_before(text, m.start(),
                                        {w.lower() for w in L.YEAR_WORDS} | {"year"},
                                        window=2):
                    continue
                yield m.start(), m.end(), \
                    self._digits_spelled(m.group(0), comma=True)

    # --- chemistry ------------------------------------------------------------
    _ELEMENTS = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
        "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Fe", "Cu", "Zn", "Ag",
        "Au", "Hg", "Pb", "I", "Mn", "Cr", "Ni", "Sn", "Br", "Mg",
    }
    _CHEM_WHITELIST = {"HCl", "NaCl", "KCl", "NaOH", "KOH", "HBr", "HF",
                       "CO", "NO", "NH3", "CH4", "H2O", "CO2", "SO2", "O2",
                       "N2", "H2", "H2SO4", "HNO3", "CaCO3", "C6H12O6",
                       "NaHCO3", "MgSO4", "ZnO", "FeO"}

    def _chem_ok(self, tok):
        if tok in self._CHEM_WHITELIST:
            return True
        pieces = re.findall(r"[A-Z][a-z]?|\d+", tok)
        letters = [p for p in pieces if not p.isdigit()]
        digits = [p for p in pieces if p.isdigit()]
        if not letters or any(p not in self._ELEMENTS for p in letters):
            return False
        return bool(digits) and len(letters) >= 2 or (bool(digits) and tok in self._CHEM_WHITELIST)

    def _m_chem(self, text):
        L = self.lang
        for m in re.finditer(r"\b(?:[A-Z][a-z]?\d*){1,8}\b", text):
            tok = m.group(0)
            if len(tok) < 2 or tok.upper() in _ACRONYM_WHITELIST:
                continue
            if not self._chem_ok(tok):
                continue
            out = []
            for piece in re.findall(r"[A-Z][a-z]?|\d+", tok):
                if piece.isdigit():
                    out.append(L.cardinal(int(piece)))
                else:
                    out.append(self._letters_spelled(piece.upper()))
            yield m.start(), m.end(), " ".join(out)

    # --- math -------------------------------------------------------------
    def _m_comparison(self, text):
        # Indic comparative phrases are postpositions: '<500' -> '500 से कम',
        # not 'से कम 500'. Treat the numeric bound as one semantic span.
        L = self.lang
        number = r"[+-]?\d+(?:,\d{2,3})*(?:\.\d+)?"
        operand = rf"(?:₹\s*)?{number}(?:\s*%)?"
        pattern = rf"(?<![\w.])(?:(?P<left>{operand})\s*)?(?P<op><=|>=|≤|≥|<|>)\s*(?P<right>{operand})"
        def words(token):
            s = token.strip()
            if s.startswith("₹"):
                return L.join_amount(self._number_words(s[1:].strip()), L.SYM["rupee"])
            if s.endswith("%"):
                return f"{self._number_words(s[:-1].strip())} {L.SYM['percent']}"
            return self._number_words(s)
        ors = {"hi": "या", "te": "లేదా", "ta": "அல்லது", "bn": "বা", "gu": "અથવા",
               "kn": "ಅಥವಾ", "ml": "അല്ലെങ്കിൽ", "mr": "किंवा", "or": "କିମ୍ବା", "pa": "ਜਾਂ"}
        for m in re.finditer(pattern, text):
            op = m['op']
            comparison = L.SYM["lt" if op.startswith("<") or op == "≤" else "gt"]
            if len(op) > 1 or op in ("≤", "≥"):
                comparison += " or equal to" if L.code == "en" else f" {ors[L.code]} {L.SYM['equals']}"
            left = words(m['left']) + " " if m['left'] else ""
            right = words(m['right'])
            yield m.start(), m.end(), left + (f"{comparison} {right}" if L.code == "en" else f"{right} {comparison}")

    def _m_math(self, text):
        L = self.lang
        S = L.SYM
        # power: 10^23  /  x^2
        for m in re.finditer(r"(\d+(?:\.\d+)?|[A-Za-z])\s*\^\s*(\d+|[A-Za-z])", text):
            base, exp = m.group(1), m.group(2)
            bw = self._number_words(base) if base[0].isdigit() else L.letter(base)
            ew = self._number_words(exp) if exp[0].isdigit() else L.letter(exp)
            yield m.start(), m.end(), _fmt(L.POWER, base=bw, exp=ew)
        # squared ²
        for m in re.finditer(r"(\d+(?:\.\d+)?|[A-Za-z])²", text):
            v = m.group(1)
            vw = self._number_words(v) if v[0].isdigit() else L.letter(v)
            yield m.start(), m.end(), f"{vw} {S['squared']}"
        # √
        for m in re.finditer(r"√\s*(\d+(?:\.\d+)?)", text):
            yield m.start(), m.end(), f"{S['sqrt']} {self._number_words(m.group(1))}"
        # operator chains: x + 5 = 12, E = mc^2, 17 × 3 = 51 — rendered only
        # when a strong math operator (= × ÷ ^ √) is present, so hyphen
        # ranges/ids and prose like "e-mail" never match
        eq = re.compile(
            r"(?<![\w.])((?:[A-Za-z]{1,2}|\d+(?:\.\d+)?)"
            r"(?:\s*[=+\-−×*÷/^]\s*(?:[A-Za-z]{1,2}|\d+(?:\.\d+)?))+)")
        for m in eq.finditer(text):
            span = m.group(1)
            if not re.search(r"[=×÷^√]", span):
                continue
            yield m.start(1), m.end(1), self._render_math_expr(span)
        # single letters adjacent to verbal math words: "a squared plus b"
        for m in re.finditer(r"\b([A-Za-z])\b(?=\s+(?:squared|cubed|plus|minus|"
                             r"times|equals|to the power)\b)", text):
            yield m.start(), m.end(), L.letter(m.group(1))
        # Operators are expanded in-place in _final_sweep. This also handles
        # thresholds without a left operand and preserves x < y word order.

    def _render_math_expr(self, span):
        L = self.lang
        S = L.SYM
        opmap = {"=": S["equals"], "+": S["plus"], "-": S["minus"],
                 "−": S["minus"], "×": S["times"], "*": S["times"],
                 "÷": S["divide"], "/": S["divide"]}
        toks = re.findall(r"\d+(?:\.\d+)?|[A-Za-z]+|[=+\-−×*÷/^]", span)
        out = []
        i = 0
        while i < len(toks):
            tok = toks[i]
            if tok == "^":
                # base already emitted; render exponent via the power template
                exp = toks[i + 1] if i + 1 < len(toks) else ""
                ew = (self._number_words(exp) if exp[:1].isdigit()
                      else self._letters_spelled(exp.upper()))
                base = out.pop() if out else ""
                out.append(_fmt(L.POWER, base=base, exp=ew))
                i += 2
                continue
            if tok in opmap:
                out.append(opmap[tok])
            elif tok[0].isdigit():
                out.append(self._number_words(tok))
            else:
                out.append(self._letters_spelled(tok.upper()))
            i += 1
        return " ".join(out)

    # --- ranges -----------------------------------------------------------
    def _m_num_range(self, text):
        L = self.lang
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*[–]\s*(\d+(?:\.\d+)?)", text):
            a = self._number_words(m.group(1))
            b = self._number_words(m.group(2))
            yield m.start(), m.end(), _fmt(L.RANGE, a=a, b=b)
        # hyphen range: only ascending small pairs, no id keyword
        for m in re.finditer(r"\b(\d{1,3})-(\d{1,3})\b", text):
            if self._keyword_before(text, m.start(), self._id_keywords):
                continue
            a, b = int(m.group(1)), int(m.group(2))
            if a == b:
                # 50-50 -> "fifty fifty"
                yield m.start(), m.end(), f"{L.cardinal(a)} {L.cardinal(b)}"
                continue
            if a > b:
                if re.match(r"\s+(?:days?|months?|years?|hours?|minutes?)\b", text[m.end():], re.I):
                    yield m.start(), m.end(), _fmt(L.RANGE, a=L.cardinal(a), b=L.cardinal(b))
                    continue
                # score-style pair: "3-2 से जीते" -> "तीन दो"
                yield m.start(), m.end(), f"{L.cardinal(a)} {L.cardinal(b)}"
                continue
            yield m.start(), m.end(), \
                _fmt(L.RANGE, a=L.cardinal(a), b=L.cardinal(b))

    # --- dimensions / "x" as times: 24x7, 1920x1080, 6 x 4 -------------------
    def _m_dimensions(self, text):
        L = self.lang
        for m in re.finditer(
                r"\b(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\b", text):
            if m.group(1) == "0":
                continue  # hex literals like 0x1F
            yield m.start(), m.end(), \
                f"{self._number_words(m.group(1))} {L.SYM['times']} " \
                f"{self._number_words(m.group(2))}"
        # lone x between something and a number (higher-priority matcher may
        # have consumed the left operand: "₹4,583.33 x 24")
        for m in re.finditer(r"(?<=\s)[x×](?=\s\d)", text):
            yield m.start(), m.end(), L.SYM["times"]

    # --- Roman numerals after label words: Class XII, World War II ----------
    _ROMAN_VALS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    def _roman_value(self, s):
        total = 0
        for i, ch in enumerate(s):
            v = self._ROMAN_VALS[ch]
            if i + 1 < len(s) and self._ROMAN_VALS[s[i + 1]] > v:
                total -= v
            else:
                total += v
        return total

    def _m_roman(self, text):
        L = self.lang
        # {2,7}: single letters stay letters (Counter 2-C reads "C", not 100)
        pat = re.compile(
            r"\b(?=[IVXLCDM]{2,7}\b)M{0,3}(?:CM|CD|D?C{0,3})"
            r"(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b")
        ctx = self._label_words | {"war", "vol", "part", "act", "book",
                                   "youdh", "युद्ध", "अध्याय", "कक्षा",
                                   "भाग", "खंड"}
        for m in pat.finditer(text):
            tok = m.group(0)
            if not tok:
                continue
            # only with a label-ish word right before — avoids XL sizes,
            # initials, and words like "MIX"
            if not self._keyword_before(text, m.start(), ctx, window=2):
                continue
            yield m.start(), m.end(), L.cardinal(self._roman_value(tok))

    # --- K/M/B suffixes glued to numbers: 3.4K, 1.2M -------------------------
    def _m_kmb_suffix(self, text):
        L = self.lang
        for m in re.finditer(r"(?<![\d.A-Za-z])(\d+(?:\.\d+)?)([KMB])\b", text):
            yield m.start(), m.end(), \
                f"{self._number_words(m.group(1))} {L.letter(m.group(2))}"

    # --- ordinals -----------------------------------------------------------
    def _m_ordinal(self, text):
        L = self.lang
        pat = re.compile(rf"\b(\d+)-?({self._ord_suffix_re})(?![A-Za-z0-9])", re.I)
        for m in pat.finditer(text):
            n = int(m.group(1))
            suf = m.group(2).lower()
            if suf in ("st", "nd", "rd", "th"):
                yield m.start(), m.end(), L.ordinal(n)
            else:
                yield m.start(), m.end(), L.ordinal(n, suf)

    # --- fractions -----------------------------------------------------------
    def _m_fraction(self, text):
        L = self.lang
        S = L.SYM
        pat = re.compile(r"\b(\d{1,4}(?:\.\d+)?)/(\d{1,4}(?:\.\d+)?)\b")
        for m in pat.finditer(text):
            a, b = m.group(1), m.group(2)
            if (a, b) == ("1", "2"):
                rep = S["half"]
            elif (a, b) == ("1", "4"):
                rep = S["quarter"]
            elif (a, b) == ("3", "4"):
                rep = S["three_quarters"]
            else:
                rep = _fmt(L.FRACTION, num=self._number_words(a),
                           den=self._number_words(b))
            yield m.start(), m.end(), rep

    # --- ratios -----------------------------------------------------------
    def _m_ratio(self, text):
        L = self.lang
        for m in re.finditer(r"\b(\d{1,3}):(\d{1,2})\b", text):
            mm = m.group(2)
            # 2-digit right side <=59 looks like a time -> time matcher takes it
            if len(mm) == 2 and int(mm) <= 59 and not self._keyword_before(
                    text, m.start(), self._ratio_kws, window=4):
                continue
            after = text[m.end():m.end() + 20].strip()
            if any(after.startswith(w) for w in self._ratio_kws if not
                   w.isascii()):
                # source already says the ratio word ("1:3 నిష్పత్తిలో")
                yield m.start(), m.end(), \
                    f"{L.cardinal(int(m.group(1)))} {L.cardinal(int(mm))}"
                continue
            yield m.start(), m.end(), \
                _fmt(L.RATIO, a=L.cardinal(int(m.group(1))),
                     b=L.cardinal(int(mm)))

    # --- labels (Room 314, Hall 3-A, Seat 23A, coach B4) --------------------
    def _m_label(self, text):
        L = self.lang
        lbl = "|".join(re.escape(w) for w in sorted(self._label_words, key=len,
                                                    reverse=True))
        pat = re.compile(
            rf"\b({lbl})\s+(?:(\d+)(?![\d.]|,\d)(?!-[A-Za-z]{{2,}})(?:[-\s]?([A-Za-z])\b)?|([A-Za-z])\b(?:[-\s]?(\d+))?)",
            re.IGNORECASE)
        for m in pat.finditer(text):
            word = m.group(1)
            out = [word]
            if m.group(2):
                out.append(L.cardinal(int(m.group(2))))
                if m.group(3):
                    out.append(L.letter(m.group(3)))
            elif m.group(4):
                if len(m.group(4)) == 1:
                    out.append(L.letter(m.group(4)))
                else:
                    out.append(m.group(4))
                if m.group(5):
                    out.append(L.cardinal(int(m.group(5))))
            yield m.start(), m.end(), " ".join(out)

    # --- ALLCAPS model prefix + number: "XPS 15" ----------------------------
    def _m_acronym_model(self, text):
        L = self.lang
        for m in re.finditer(r"\b([A-Z]{2,4})\s+(\d{1,3})(?![\d.])", text):
            tok = m.group(1)
            if tok in _ACRONYM_WHITELIST or tok.upper() in _UNIT_INITIALISMS:
                continue
            if tok.lower() in self._id_keywords or tok.lower() in self._label_words:
                continue
            yield m.start(), m.end(), \
                f"{self._letters_spelled(tok)} {L.cardinal(int(m.group(2)))}"

    # --- unit initialisms after numbers: 1.5 GB, 5,000 mAh ------------------
    def _m_unit_initialism(self, text):
        L = self.lang
        units = "|".join(sorted(_UNIT_INITIALISMS, key=len, reverse=True))
        pat = re.compile(
            rf"(?<![\w.])(-?)\b(\d+(?:,\d{{2,3}})*(?:\.\d+)?)[\s\-]?({units})"
            rf"(?:\s?/\s?([A-Za-z]{{1,4}}))?\b",
            re.IGNORECASE)
        for m in pat.finditer(text):
            unit = m.group(3)
            if unit.upper() not in _UNIT_INITIALISMS:
                continue
            # single-letter units (V, W) only when written in caps —
            # a lowercase "5 v" is too ambiguous
            if len(unit) == 1 and not unit.isupper():
                continue
            num = self._number_words(m.group(2))
            if m.group(1):
                num = f"{L.SYM['minus']} {num}"
            uk = re.sub(r"[^A-Za-z]", "", unit).upper()
            word = L.UNIT_WORDS.get(uk) or self._letters_spelled(uk)
            rep = L.join_amount(num, word)
            if m.group(4):
                u2 = m.group(4).upper()
                w2 = L.UNIT_WORDS.get(u2) or self._letters_spelled(u2)
                rep += f" {L.SYM.get('per', 'per')} {w2}"
            yield m.start(), m.end(), rep

    # --- number-hyphen-word: 80-watt, 900-point ------------------------------
    def _m_num_hyphen_word(self, text):
        for m in re.finditer(r"\b(\d+(?:,\d{2,3})*(?:\.\d+)?)-([A-Za-zऀ-ൿ]{2,})\b",
                             text):
            word = m.group(2)
            if self._month_index(word):
                continue  # 12-Aug
            if word.upper() in _UNIT_INITIALISMS:
                continue  # unit matcher handles
            num = self._number_words(m.group(1))
            yield m.start(), m.end(), f"{num} {word}"

    # --- decades ------------------------------------------------------------
    def _m_decade(self, text):
        L = self.lang
        for m in re.finditer(r"\b(19|20)(\d)0s\b", text):
            if m.group(1) == "19":
                # 1980s -> "eighty's decade" (spec)
                yield m.start(), m.end(), L.decade(int(m.group(2)) * 10, full=True)
            else:
                # 2020s -> full year words + decade template
                y = int(m.group(0)[:4])
                yield m.start(), m.end(), _fmt(L.DECADE_FULL, tens=L.year(y))
        for m in re.finditer(r"(?<![\w])'?(\d)0s\b", text):
            yield m.start(), m.end(), L.decade(int(m.group(1)) * 10, full=False)

    # --- years ---------------------------------------------------------------
    def _m_year(self, text):
        L = self.lang
        year_words = {w.lower() for w in L.YEAR_WORDS} | \
            {"year", "in", "since", "by", "from", "साल", "वर्ष"}
        for m in re.finditer(r"\b(1[5-9]\d{2}|20\d{2})\b", text):
            y = int(m.group(0))
            ctx = self._keyword_before(text, m.start(), year_words, window=2)
            # attached native locative suffix right after: 1947में / 1947-ல்
            after = text[m.end():m.end() + 20]
            # danda ।/॥ are inside the Devanagari block — they are sentence
            # punctuation, not a locative suffix
            suffixed = bool(re.match(r"^-?(?![।॥])[ऀ-෿]", after))
            # following token is/starts with a year word: "1947 সালে",
            # "1983 साली", "१९४७ ମସିହାରେ", or Hindi locatives
            next_tok = after.strip().split()[0] if after.strip() else ""
            follower = next_tok in ("में", "से", "तक", "માં", "मध्ये",
                                    "ਵਿੱਚ", "ਤੋਂ") or any(
                next_tok.startswith(w) for w in L.YEAR_WORDS if w)
            if ctx or suffixed or follower:
                yield m.start(), m.end(), L.year(y)

    # --- quoted single letters: reply 'Y' -----------------------------------
    def _m_quoted_letter(self, text):
        for m in re.finditer(r"['\"‘“]([A-Za-z])['\"’”]", text):
            yield m.start(), m.end(), self.lang.letter(m.group(1))

    # --- short mixed tokens: 5G, A4, B12, Y2K, S24, A380 ---------------------
    def _m_short_mixed_token(self, text):
        L = self.lang
        pat = re.compile(r"(?<![\d.])\b(?=[A-Za-z0-9]{2,5}\b)"
                         r"(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])"
                         r"[A-Za-z0-9]+\b")
        for m in pat.finditer(text):
            tok = m.group(0)
            if re.fullmatch(r"\d+(?:st|nd|rd|th|s)", tok):
                continue  # ordinals / decades
            out = []
            for run in re.findall(r"[A-Za-z]+|\d+", tok):
                if run.isdigit():
                    out.append(L.cardinal(int(run)))
                else:
                    out.append(self._letters_spelled(run.upper()))
            yield m.start(), m.end(), " ".join(out)

    # --- decimals -------------------------------------------------------------
    def _m_decimal(self, text):
        # trailing (?!\.?\d) instead of (?![\d.]) so a sentence-final full
        # stop after the decimal ("...830.93.") doesn't block the match
        for m in re.finditer(
                r"(?<![\d.])(\d{1,3}(?:,\d{2,3})+|\d+)\.(\d+)(?!\.?\d)", text):
            yield m.start(), m.end(), \
                self.lang.decimal(int(m.group(1).replace(",", "")), m.group(2))

    # --- negative numbers -------------------------------------------------------
    def _m_negative(self, text):
        L = self.lang
        for m in re.finditer(r"(?<![\w.,-])-(\d+(?:\.\d+)?)", text):
            yield m.start(), m.end(), \
                f"{L.SYM['minus']} {self._number_words(m.group(1))}"

    # --- plain integers ----------------------------------------------------------
    def _m_int(self, text):
        L = self.lang
        for m in re.finditer(r"\b\d{1,3}(?:,\d{2,3})+\b|\b\d+\b", text):
            digits = m.group(0).replace(",", "")
            # street numbers read year-style (1422 गांधी मार्ग -> चौदह सौ बाईस)
            after = text[m.end():m.end() + 30].strip().lower()
            first_words = after.split()[:2]
            if (len(digits) == 4 and digits[0] != "0"
                    and any(w.strip(".,") in _STREET_WORDS for w in first_words)):
                yield m.start(), m.end(), L.year(int(digits))
                continue
            if digits.startswith("0") and len(digits) > 1:
                yield m.start(), m.end(), self._digits_spelled(digits, comma=True)
                continue
            yield m.start(), m.end(), L.cardinal(int(digits))

    # ------------------------------------------------------------------
    # Final sweep: leftover meaningful symbols -> words; decorative junk out
    # ------------------------------------------------------------------
    def _final_sweep(self, text):
        S = self.lang.SYM
        text = re.sub(r"<=|≤", " less than or equal to ", text)
        text = re.sub(r">=|≥", " greater than or equal to ", text)
        text = re.sub(r"°\s*C\b", f" {S['celsius']} ", text)
        text = re.sub(r"°\s*F\b", f" {S['fahrenheit']} ", text)
        text = text.replace("°", f" {S['degree']} ")
        rep = {"%": S["percent"], "@": S["at"], "&": S["amp"],
               "=": S["equals"], "+": S["plus"], "×": S["times"],
               "÷": S["divide"], "±": S["plus_minus"], "√": S["sqrt"],
               "_": S["underscore"], "#": S["hash"]}
        for sym, word in rep.items():
            text = text.replace(sym, f" {word} ")
        text = text.replace("<", f" {S['lt']} ").replace(">", f" {S['gt']} ")
        text = re.sub(r"[\^*]", " ", text)
        # slashes not consumed by URL/fraction/date/code matchers are junk
        text = re.sub(r"\s*/\s*", " ", text)
        # clause colons read as a pause ("అడ్రస్: Door" -> "అడ్రస్, Door")
        text = re.sub(r"\s*:\s*", ", ", text)
        text = re.sub(r"[₹$€£](?!\s?\d)", " ", text)  # bare currency signs
        text = re.sub(r"[–—]", " ", text)
        return text

    # ------------------------------------------------------------------
    # Fallback: absolutely no digits may remain
    # ------------------------------------------------------------------
    def _digit_fallback(self, text):
        def repl(m):
            d = m.group(0)
            if len(d) >= 7 or d.startswith("0"):
                return self._digits_spelled(d, comma=True)
            return self.lang.cardinal(int(d))
        return re.sub(r"\d+", repl, text)

    # ------------------------------------------------------------------
    def normalize(self, text):
        validate_text(text, date_order=self.date_order, two_digit_year_base=self.two_digit_year_base, language=self.lang.code)
        # URLs/emails are opaque identifiers. Protect them before prose cleanup
        # can expand Dr/No inside a path or turn // and parentheses into pauses.
        raw = _ascii_digits(text)
        protected, originals, spans = {}, {}, []
        identifiers = (self._m_email, self._m_url)
        if os.environ.get("FR3_TN_URL_SPELL", "0") == "1":
            identifiers += (self._m_handle,)
        for matcher in identifiers:
            for s, e, spoken in matcher(raw):
                if not any(s < b and e > a for a, b, _ in spans):
                    key = chr(0xF0000 + len(protected))
                    protected[key] = spoken
                    originals[key] = raw[s:e]
                    spans.append((s, e, key))
        for s, e, key in sorted(spans, reverse=True):
            raw = raw[:s] + key + raw[e:]
        text = raw
        text = self._preclean(text)
        try:
            matches = self._gather(text)
        except NormalizationError as exc:
            if exc.span is not None:
                start, end = exc.span
                pieces = [text[:start], text[start:end], text[end:]]
                # Parser coordinates refer to precleaned text containing opaque
                # URI placeholders. Restore their source text before retrying.
                for key, original in originals.items():
                    pieces = [piece.replace(key, original) for piece in pieces]
                exc.source = "".join(pieces)
                exc.span = (len(pieces[0]), len(pieces[0]) + len(pieces[1]))
            raise
        # priority asc, then leftmost, then longest
        matches.sort(key=lambda t: (t[0], t[1], -(t[2] - t[1])))
        taken = []
        accepted = []
        for prio, s, e, rep in matches:
            if any(not (e <= ts or s >= te) for ts, te in taken):
                continue
            taken.append((s, e))
            accepted.append((s, e, rep))
        accepted.sort(key=lambda t: t[0])
        out = []
        pos = 0
        for s, e, rep in accepted:
            out.append(text[pos:s])
            pos = e
            # native postposition right after the span (spaced OR attached:
            # "గంటలు కి" / "6:30కి") -> language attaches it with sandhi
            while True:
                nxt = re.match(r"[ \t]*-?([^\s\d,।.!?;:()\-—–]{1,14})",
                               text[pos:])
                if not nxt:
                    break
                if text[pos + nxt.end(): pos + nxt.end() + 1] == ".":
                    break  # abbreviation like కి.మీ — not a postposition
                merged = self.lang.attach_postposition(rep, nxt.group(1))
                if merged is None:
                    break
                rep = merged
                pos += nxt.end()
            out.append(rep)
        out.append(text[pos:])
        result = "".join(out)
        result = self._final_sweep(result)
        result = self._digit_fallback(result)
        for key, spoken in protected.items():
            result = result.replace(key, spoken)
        # A foreign cardinal must never be glued into an Indic suffix (fiveన).
        result = re.sub(r"(?<=[A-Za-z])(?=[\u0900-\u0DFF])", " ", result)
        # tidy whitespace / punctuation spacing
        result = re.sub(r"\s+([,।.!?;])", r"\1", result)
        result = re.sub(r",\s*([।.!?;,])", r"\1", result)
        result = re.sub(r",(?=\S)", ", ", result)  # space after every comma
        result = re.sub(r"\s{2,}", " ", result)
        result = re.sub(r"\s+’", "’", result)
        result = re.sub(r"[,\s]+$", ".", result) if result.rstrip().endswith(",") else result
        if self.lang.code == "hi":
            result = result.translate(_HINDI_NUKTA_FOLD)
        return result.strip()




# ---------------------------------------------------------------------------
# English table (the only language table)
# ---------------------------------------------------------------------------

_ONES_0_19 = ["zero", "one", "two", "three", "four", "five", "six", "seven",
              "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
              "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _two_digit(n):
    """0..99 -> 'zero' .. 'ninety nine'. No hyphen: the model reads words."""
    if n < 20:
        return _ONES_0_19[n]
    t, o = divmod(n, 10)
    return _TENS[t] if not o else f"{_TENS[t]} {_ONES_0_19[o]}"


_ORDINAL_IRREGULAR = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth",
                      9: "ninth", 12: "twelfth"}


def _ordinal(n):
    if n in _ORDINAL_IRREGULAR:
        return _ORDINAL_IRREGULAR[n]
    w = _two_digit(n) if n < 100 else None
    if w is None:
        return None
    # twenty-first, thirty-second ... take the irregular form on the last digit
    tail = n % 10
    if n >= 20 and tail in _ORDINAL_IRREGULAR:
        return f"{_TENS[n // 10]} {_ORDINAL_IRREGULAR[tail]}"
    if w.endswith("y"):                 # twenty -> twentieth
        return w[:-1] + "ieth"
    return w + "th"


class English(LangBase):
    code = "en"
    name = "English"
    native_digits = "0123456789"        # English uses ASCII digits

    ONES = [_two_digit(i) for i in range(100)]
    HUNDRED = "hundred"
    THOUSAND = "thousand"
    LAKH = "lakh"
    CRORE = "crore"

    # Letter names are already English, so each maps to itself.
    LETTERS = {chr(65 + i): chr(65 + i) for i in range(26)}

    MONTHS = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]

    SYM = {
        "rupee": "rupees", "dollar": "dollars", "euro": "euros",
        "pound": "pounds", "cent": "cents",
        "percent": "percent",
        "point": "point",
        "minus": "minus", "plus": "plus", "equals": "equals",
        "times": "times", "divide": "divided by",
        "squared": "squared", "sqrt": "square root of",
        "lt": "less than", "gt": "greater than",
        "plus_minus": "plus or minus",
        "at": "at", "underscore": "underscore", "dot": "dot",
        "slash": "slash", "hash": "hash", "amp": "and",
        "half": "half", "quarter": "quarter", "three_quarters": "three quarters",
        "degree": "degrees", "celsius": "celsius", "fahrenheit": "fahrenheit",
        "minute": "minutes",
    }

    TIME_ON_HOUR = "{h} o'clock"
    TIME_HM = "{h} {m}"                 # "ten thirty" — how a clock is read aloud
    RANGE = "{a} to {b}"
    RANGE_TO = "{a} to {b}"
    RATIO = "{a} to {b}"
    POWER = "{base} to the power {exp}"
    FRACTION = "{num} over {den}"
    DECADE_SHORT = "{tens}s"
    DECADE_FULL = "{tens}s"

    PERIODS = {"morning": "in the morning", "afternoon": "in the afternoon",
               "evening": "in the evening", "night": "at night"}

    # ---- vocabulary the engine must recognise in the INPUT ----------------
    TIME_MARKERS = ["o'clock", "oclock"]
    MARKER_SUFFIXES = {}
    PERIOD_DETECT = {"morning": "morning", "afternoon": "afternoon",
                     "evening": "evening", "night": "night",
                     "am": "morning", "a.m.": "morning",
                     "pm": "afternoon", "p.m.": "afternoon"}
    YEAR_WORDS = ["year", "years"]
    # Deliberately NOT "order"/"invoice"/"ticket" on their own: "your order is 250
    # rupees" is a price, and listing them made it read "two, five, zero". They only
    # mark an identifier in the explicit "order number ..." form, which "number" catches.
    # One normalizer serves every language, so an identifier must be recognised
    # whatever script the sentence around it is written in. Without the native
    # words, "ऑर्डर नंबर 47291" reads as a cardinal instead of digit-by-digit.
    ID_KEYWORDS = ["number", "pin", "otp", "pnr", "imei", "ifsc", "account", "नंबर", "नम्बर",
        "खाता", "खाते", "कोड", "ऑर्डर", "फ़ोन", "फोन", "मोबाइल", "నంబర్", "నంబరు", "నెంబర్",
        "నెంబరు", "ఖాతా", "ఫోన్", "మొబైల్", "కోడ్", "ఆర్డర్", "ఎక్స్ప్రెస్", "express",
        "అకౌంట్", "టికెట్", "రిజిస్ట్రేషన్", "యూజర్", "వాయిదా", "అంకె", "లావాదేవీ", "పిన్",
        "எண்", "நம்பர்", "கணக்கு", "தொலைபேசி", "போன்", "மொபைல்", "ஆர்டர்", "ஓடிபி", "பின்",
        "ஐடி", "கடவுச்சொல்", "ரயில்", "எக்ஸ்பிரஸ்", "பிஎன்ஆர்", "கார்டு", "குறியீடு",
        "டிக்கெட்", "கணக்கில்", "யூசர்", "தவணை", "இலக்கம்", "பரிவர்த்தனை", "ನಂಬರ್", "ಸಂಖ್ಯೆ",
        "ಖಾತೆ", "ಕೋಡ್", "ಆರ್ಡರ್", "ಫೋನ್", "ಮೊಬೈಲ್", "ಐಡಿ", "ಪಿನ್", "ಪಾಸ್‌ವರ್ಡ್", "ರೈಲು",
        "ಎಕ್ಸ್‌ಪ್ರೆಸ್", "ಪಿಎನ್‌ಆರ್", "ಕಾರ್ಡ್", "ರೆಫರೆನ್ಸ್", "ಟ್ರಾನ್ಸಾಕ್ಷನ್", "ಟಿಕೆಟ್", "ಓಟಿಪಿ",
        "ಅಕೌಂಟ್", "ಅಂಕಿ", "ಬಳಕೆದಾರ", "ಕಂತು", "ವಹಿವಾಟು", "നമ്പർ", "നമ്പര്", "നമ്പറിൽ",
        "അക്കൗണ്ട്", "കോഡ്", "ഓർഡർ", "ഫോൺ", "മൊബൈൽ", "ഒടിപി", "പിൻ", "ഐഡി", "പാസ്‌വേഡ്",
        "ട്രെയിൻ", "എക്സ്പ്രസ്", "പിഎൻആർ", "കാർഡ്", "റഫറൻസ്", "ട്രാൻസാക്ഷൻ", "ടിക്കറ്റ്",
        "അക്കൗണ്ടിൽ", "നമ്പറിലേക്ക്", "യൂസർ", "തവണ", "അക്കം", "ഇടപാട്", "क्रमांक", "खात्याचा",
        "खात्यावर", "मोबाईल", "अकाउंट", "अकाउंटसाठी", "अकाउंटमधून", "खात्यातून", "खात्यात",
        "खात्याची", "पिन", "आयडी", "पासवर्ड", "ट्रेन", "एक्सप्रेस", "पीएनआर", "कार्ड",
        "रेफरन्स", "ट्रांझॅक्शन", "तिकीट", "ओटीपी", "गाडी", "युजर", "यूजर", "हप्ता",
        "हप्त्याचा", "अंक", "व्यवहार", "ट्रान्झॅक्शन", "क्र", "क्रं", "चेक", "धनादेश", "নম্বর",
        "নাম্বার", "অ্যাকাউন্ট", "একাউন্ট", "কোড", "অর্ডার", "ফোন", "মোবাইল", "ওটিপি", "পিন",
        "আইডি", "পাসওয়ার্ড", "ট্রেন", "এক্সপ্রেস", "পিএনআর", "কার্ড", "রেফারেন্স",
        "ট্রানজেকশন", "টিকিট", "অ্যাকাউন্টের", "নম্বরে", "আইডিতে", "ইউজার", "ইনস্টলমেন্ট",
        "কিস্তি", "ডিজিট", "অঙ্ক", "লেনদেন", "નંબર", "નમ્બર", "ખાતા", "ખાતું", "ખાતાં", "કોડ",
        "ઓર્ડર", "ફોન", "મોબાઇલ", "મોબાઈલ", "એકાઉન્ટ", "એકાઉન્ટમાં", "એકાઉન્ટનો", "પિન",
        "આઈડી", "પાસવર્ડ", "ટ્રેન", "એક્સપ્રેસ", "પીએનઆર", "કાર્ડ", "રેફરન્સ", "ટ્રાન્ઝેક્શન",
        "ટિકિટ", "ઓટીપી", "નંબરમાં", "યુઝર", "હપ્તો", "અંક", "ડિજિટ", "ਨੰਬਰ", "ਖਾਤਾ", "ਖਾਤੇ",
        "ਕੋਡ", "ਆਰਡਰ", "ਫ਼ੋਨ", "ਫੋਨ", "ਮੋਬਾਈਲ", "ਮੋਬਾਇਲ", "ਅਕਾਊਂਟ", "ਅਕਾਊਂਟਵਿੱਚ", "ਪਿੰਨ",
        "ਆਈਡੀ", "ਪਾਸਵਰਡ", "ਟ੍ਰੇਨ", "ਐਕਸਪ੍ਰੈਸ", "ਪੀਐਨਆਰ", "ਕਾਰਡ", "ਰੈਫਰੈਂਸ", "ਟਿਕਟ", "ਓਟੀਪੀ",
        "ਯੂਜ਼ਰ", "ਕਿਸ਼ਤ", "ਅੰਕ", "ਲੈਣ-ਦੇਣ", "ନମ୍ବର", "ନମ୍ବର୍", "ଖାତା", "ଆକାଉଣ୍ଟ", "କୋଡ୍",
        "ଅର୍ଡର", "ଫୋନ୍", "ଫୋନ", "ମୋବାଇଲ୍", "ମୋବାଇଲ", "ଓଟିପି", "ପିନ୍", "ଆଇଡି", "ପାସୱାର୍ଡ",
        "ଏକ୍ସପ୍ରେସ", "ପିଏନ୍‌ଆର୍", "କାର୍ଡ", "ରେଫରେନ୍ସ", "ଟିକେଟ", "ଆକାଉଣ୍ଟରୁ", "ଟ୍ରେନ୍", "ୟୁଜର",
        "କିସ୍ତି", "ଅଙ୍କ", "ଲାଣଦେଣ"]
    ID_AFTER_WORDS = ["number", "code", "खत्म", "डायल", "कॉल", "संपर्क", "ఎండ్", "ముగిసే",
        "ముగుస్తుంది", "చివర", "చివరి", "అంకెలతో", "డయల్", "పిన్", "పిన్‌కోడ్", "pin",
        "pincode", "కాల్", "సంప్రదించండి", "டயல்", "அழை", "தொடர்பு", "ಡಯಲ್", "ಕರೆ", "ಸಂಪರ್ಕ",
        "ডায়াল", "কল", "যোগাযোগ", "ડાયલ", "કૉલ", "સંપર્ક", "ਡਾਇਲ", "ਕਾਲ", "ਸੰਪਰਕ", "ଡାଏଲ",
        "କଲ", "ଯୋଗାଯୋଗ"]
    LABEL_WORDS = ["room", "flat", "floor", "gate", "platform", "seat",
                   "section", "chapter", "page", "block", "phase"]
    MINUTE_WORDS = ["minute", "minutes", "min", "mins"]
    SCALE_WORDS = {"crore": "crore", "crores": "crore", "cr": "crore",
                   "lakh": "lakh", "lakhs": "lakh", "lac": "lakh",
                   "thousand": "thousand", "thousands": "thousand", "k": "thousand"}
    CURRENCY_WORDS = {"rupee": "rupee", "rupees": "rupee", "rs": "rupee",
                      "inr": "rupee", "dollar": "dollar", "dollars": "dollar",
                      "usd": "dollar", "euro": "euro", "euros": "euro",
                      "pound": "pound", "pounds": "pound",
                      "paisa": "cent", "paise": "cent",
                      "cent": "cent", "cents": "cent"}

    # Unit initialisms spoken as full words instead of letters. The engine
    # falls back to letter-spelling ("K G") when a key is absent, which is
    # right for GB/MB but wrong for the metric units customers actually say.
    UNIT_WORDS = {
        "KG": "kilograms", "KM": "kilometers", "CM": "centimeters",
        "MM": "millimeters", "GM": "grams", "MG": "milligrams",
        "ML": "millilitres", "MCG": "micrograms",
        "KMPH": "kilometers per hour", "MPH": "miles per hour",
        "KWH": "kilowatt hours", "RPM": "R P M",
    }

    # Written-only abbreviations expanded wherever they appear. Case-insensitive,
    # guarded by word boundaries, so "Drew" and "MIN-22" are untouched.
    ABBREV_UNITS = {
        "MINS": "minutes", "MIN": "minutes", "HRS": "hours", "SECS": "seconds",
        "DR": "doctor", "MR": "mister", "MRS": "missus", "MS": "miss",
        "PROF": "professor", "LTD": "limited", "PVT": "private",
        "A/C": "account", "APPROX": "approximately", "ETA": "E T A",
    }

    # "No. 12" -> "number twelve"
    NUMBER_WORD = "number"

    # English transcribes the am/pm that was written; it never invents a period.
    # "6.45 pm" -> "six forty five p m", NOT "six forty five in the evening".
    AMPM_WORDS = {"am": "a m", "pm": "p m"}

    ORDINAL_EXCEPTIONS = {n: _ordinal(n) for n in range(1, 100) if _ordinal(n)}
    ORDINAL_SUFFIX = "th"
    ORDINAL_INPUT_SUFFIXES = ["st", "nd", "rd", "th"]


    def time_phrase(self, h, m, period_key=None):
        """English puts the period AFTER the clock time — "five in the evening", not
        the base class's Indo-Aryan "in the evening five" (शाम पाँच बजे)."""
        if h == 0:
            core = "zero " + ("hundred" if m == 0 else self.two_digit(m))
        else:
            core = super().time_phrase(h, m, None)
        return f"{core} {self.PERIODS[period_key]}" if period_key else core

    def _ordinal_compose(self, n, src_suffix=""):
        if n < 100:
            return "zeroth" if n == 0 else _ordinal(n)
        for scale, word in ((10**7, self.CRORE), (10**5, self.LAKH),
                            (1000, self.THOUSAND), (100, self.HUNDRED)):
            if n >= scale:
                count, rest = divmod(n, scale)
                head = f"{self.cardinal(count)} {word}"
                return f"{head} {self.ordinal(rest)}" if rest else head + "th"


LANG = English()
_NORMALIZER = Normalizer(LANG)


@lru_cache(maxsize=66)
def _engine(language, date_order, two_digit_year_base):
    return Normalizer(language_table(language), date_order=date_order, two_digit_year_base=two_digit_year_base)


def normalize(text, language="en-IN", *, date_order="DMY", two_digit_year_base=None):
    """Locale-specific numbers and symbols; prose is preserved, never translated.

    Omitted language retains Indian English for compatibility. Use an explicit
    language/locale for Indic speech. language="auto" selects majority-script
    inference with a guard against introducing foreign-script expansions.
    """
    auto = isinstance(language, str) and language == "auto"
    original = text
    if auto:
        if __package__:
            from .script_guard import detect_language, guard_output
        else:
            from script_guard import detect_language, guard_output
        language = detect_language(text) if isinstance(text, str) else "en-IN"
    code = resolve_language(language)
    # Validate before accessing the cache, including unhashable option values.
    warning = None
    while True:
        try:
            validate_text(text, date_order=date_order, two_digit_year_base=two_digit_year_base, language=code)
            result = _engine(code, date_order, two_digit_year_base).normalize(text)
            break
        except NormalizationError as exc:
            # Request type/options are not text tokens and cannot be rendered.
            if os.environ.get("FR3_TN_STRICT", "0") == "1" or exc.span is None:
                raise
            warning = warning or exc.code
            source = exc.source if exc.source is not None else text
            start, end = exc.span
            token = source[start:end]
            locale = language_table(code)
            symbols = {"/": "slash", ".": "dot", "-": "minus", "+": "plus",
                       "@": "at", "_": "underscore", "₹": "rupee", "$": "dollar",
                       "%": "percent"}
            words = []
            for char in token:
                if char.isdecimal():
                    words.append(" " + locale.ONES[unicodedata.decimal(char)] + " ")
                elif char in symbols:
                    words.append(" " + locale.SYM[symbols[char]] + " ")
                elif char in ",:;<>{}" or unicodedata.category(char) in ("Cc", "Co"):
                    words.append(" ")
                else:
                    words.append(char)
            replacement = "".join(words)
            if replacement == token:
                raise  # no progress: do not loop on a new, unhandled error
            text = source[:start] + " " + replacement + " " + source[end:]
    if warning:
        # One privacy-safe observation per normalization call, never caller text.
        try:
            import obs
            obs.event("normalization_fallback", level="warn", code=warning)
        except ImportError:
            print("[obs] " + json.dumps({"event": "normalization_fallback", "level": "warn", "code": warning}), flush=True)
    return guard_output(original, result, language_table(code)) if auto else result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Plain text; otherwise read stdin.")
    parser.add_argument("--date-order", choices=("DMY", "MDY"), default="DMY")
    parser.add_argument("--language", default="en-IN", help="Language code/name, including hi, te and en-IN.")
    parser.add_argument("--two-digit-year-base", type=int, choices=(1900, 2000))
    args = parser.parse_args()
    try:
        print(normalize(args.text if args.text is not None else sys.stdin.read(), language=args.language,
                        date_order=args.date_order, two_digit_year_base=args.two_digit_year_base))
    except NormalizationError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        sys.exit(2)
'''),
    'mithi_tts.text_norm.tn.script_guard': (False, r'''"""Deterministic, CPU-only locale inference and output script containment.

Count letters/marks, not shared danda punctuation or decimal digits. Devanagari
cannot distinguish Hindi/Marathi: auto uses Hindi; explicit Marathi is preferred.
"""
import unicodedata
from collections import Counter

BLOCKS = ((0x0900, 0x097F, 'hi'), (0x0980, 0x09FF, 'bn'),
          (0x0A00, 0x0A7F, 'pa'), (0x0A80, 0x0AFF, 'gu'),
          (0x0B00, 0x0B7F, 'or'), (0x0B80, 0x0BFF, 'ta'),
          (0x0C00, 0x0C7F, 'te'), (0x0C80, 0x0CFF, 'kn'),
          (0x0D00, 0x0D7F, 'ml'))


def script_counts(text):
    counts = Counter()
    for c in text:
        if unicodedata.category(c)[0] not in 'LM':
            continue
        if unicodedata.name(c, '').startswith('LATIN'):
            counts['en-IN'] += 1
            continue
        for low, high, code in BLOCKS:
            if low <= ord(c) <= high:
                counts[code] += 1
                break
    return counts


def detect_language(text):
    counts = script_counts(text)
    code = max(counts, key=counts.get) if counts else 'en-IN'
    if code == 'bn' and any(c in text for c in 'ৰৱ'):
        return 'as'
    return code


def guard_output(original, result, locale):
    allowed = set(script_counts(original))
    # A numeric/symbol-only input supplies no script; use the default locale.
    if not allowed or set(script_counts(result)) <= allowed:
        return result
    # Fail conservatively: preserve prose and read the original digits in the
    # chosen input script; never return a generated foreign-script expansion.
    if __package__:
        from .normalize import language_table
    else:
        from normalize import language_table
    locale = language_table(detect_language(original))
    result = ''.join(' ' + locale.digit(unicodedata.decimal(c)) + ' '
                     if c.isdecimal() else c for c in original)
    return ' '.join(result.split()) if set(script_counts(result)) <= allowed else original
'''),
    'mithi_tts.text_norm.tn.text_pipeline': (False, r'''"""CPU-only request preparation shared by HTTP and WebSocket serving.

No model imports, logging of caller text, or silent raw-text fallback.
"""
if __package__:
    from .normalize import LANGUAGE_NAMES, NormalizationError, validate_text, resolve_language
    from .script_guard import detect_language
else:
    from normalize import LANGUAGE_NAMES, NormalizationError, validate_text, resolve_language
    from script_guard import detect_language

NORMALIZATION_LANGUAGES = tuple(LANGUAGE_NAMES)


class PipelineError(Exception):
    def __init__(self, code, message, status=422):
        self.code, self.status = code, status
        super().__init__(message)

    def payload(self):
        return {"error": str(self), "code": self.code}


def prepare_text(text, *, normalizer, splitter, post_process, normalize_enabled=True,
                 chunk_enabled=True, date_order="DMY", two_digit_year_base=None, max_chunk_chars=None,
                 normalization_language="en-IN"):
    try:
        if not normalize_enabled:
            validate_text(text, date_order=date_order, two_digit_year_base=two_digit_year_base, language=detect_language(text) if normalization_language is None and isinstance(text, str) else normalization_language)
        if not isinstance(text, str):
            raise NormalizationError("invalid_text", "text must be a string.")
        t = text.strip()
        if not t:
            return []
        if normalize_enabled:
            if normalizer is None:
                raise PipelineError("normalizer_unavailable", "Text normalizer unavailable.", 503)
            t = normalizer(t, language="auto" if normalization_language is None else normalization_language, date_order=date_order, two_digit_year_base=two_digit_year_base)
            if not isinstance(t, str) or not any(c.isalnum() for c in t):
                raise PipelineError("empty_normalized_text", "No speakable text remains.")
        if chunk_enabled:
            if splitter is None or post_process is None:
                raise PipelineError("chunker_unavailable", "Text chunker unavailable.", 503)
            t = post_process(t)
            parts = splitter(t)
            if not parts or any(not isinstance(p, str) or not p.strip() for p in parts):
                raise PipelineError("invalid_chunks", "Text chunker returned invalid chunks.", 503)
            # Current chunker changes whitespace only. Detect dropped/reordered text
            # before audio starts; do not send a too-long whole-text fallback.
            if " ".join(" ".join(parts).split()) != " ".join(t.split()):
                raise PipelineError("chunk_content_mismatch", "Text chunker changed content.", 503)
            if max_chunk_chars and any(len(p) > max_chunk_chars for p in parts):
                raise PipelineError("unsplittable_text", "A text token exceeds the safe chunk size; shorten it.")
            return parts
        return [t]
    except NormalizationError as exc:
        raise PipelineError(exc.code, str(exc)) from None
    except PipelineError:
        raise
    except Exception:
        raise PipelineError("text_pipeline_failed", "Text preparation failed; no audio was generated.", 503) from None


def resolve_conditioning(settings, supported):
    """Never invent an untrained language tag. Legacy omission remains Hindi.

    `language`/`lang` requests a model tag unless `conditioning_language` is
    explicitly supplied. The latter is for a checkpoint trained with a tag that
    differs from the text language, and requires deployment-owner confirmation.
    """
    value = settings.get("conditioning_language") or settings.get("language") or settings.get("lang") or "Hindi"
    aliases = {"en": "English", "en-in": "English", "en-us": "English",
               "hi": "Hindi", "hi-in": "Hindi", "indian english": "English", "indian-english": "English"}
    if not isinstance(value, str):
        raise PipelineError("invalid_language", "language must be a string.")
    by_name = {x.lower(): x for x in supported}
    value = aliases.get(value.lower(), by_name.get(value.lower(), value))
    if value not in supported:
        raise PipelineError("unsupported_conditioning_language",
                            "No supported model tag for this language. Confirm the checkpoint conditioning_language.")
    return value


def resolve_normalization(settings):
    """Normalization locale and model conditioning are deliberately separate.

    Explicit normalization_language wins, then language/lang; omission preserves
    Indian English. Never guess Hindi versus Marathi from Devanagari text.
    """
    value = next((settings[key] for key in ("normalization_language", "language", "lang")
                  if key in settings), "en-IN")
    try:
        return resolve_language(value)
    except NormalizationError as exc:
        raise PipelineError(exc.code, str(exc)) from None


def merge_settings(current, incoming, sticky):
    """Snapshot per queued utterance; newer aliases replace older aliases."""
    result = dict(current)
    # A language change must not inherit a previous explicit override.
    if "language" in incoming or "lang" in incoming:
        for key in ("conditioning_language", "normalization_language"):
            if key not in incoming:
                result.pop(key, None)
    for a, b in (("voice_id", "voice"), ("language", "lang")):
        if a in incoming or b in incoming:
            result.pop(a, None)
            result.pop(b, None)
    result.update({k: incoming[k] for k in sticky if k in incoming})
    return result


_REQUEST_LANGUAGE_KEYS = frozenset(("language", "lang", "normalization_language", "conditioning_language"))


def retain_session_settings(current, incoming, sticky):
    """Text-frame locales belong to their context, not the next customer turn.

    start/config frames still use merge_settings directly and establish explicit
    session defaults. Other sticky options retain the existing protocol behavior.
    """
    return merge_settings(current, {k: v for k, v in incoming.items()
                                   if k not in _REQUEST_LANGUAGE_KEYS}, sticky)
'''),
    'mithi_tts.text_norm.tn.tts_chunker': (False, r'''"""Prosody-aware, Indic-aware, emotion-tag-safe chunker for TTS.

Python port of the TypeScript chunker (studio gateway, itself a port of
indic_chunker.py on branch local-chunker-normguard) that passes the I1-I6
invariant harness. Replaces the earlier sentence-only chunker, which had two
defects:
  * it split on any '.', '!' or '?', tearing "8.4", "example.com", "₹2,500"
    and quoted '"आओगे?"' apart;
  * it split ONLY on sentence punctuation, so a long comma-joined sentence
    became one oversized chunk that silently overran max_model_len and
    truncated mid-speech.

Splits normalized text preferring, in priority order:
  1. sentence boundaries  ( . ! ? … । ॥ — ONLY token-final, i.e. followed by
     whitespace, so decimals / URLs / currency / quoted '?' stay intact )
  2. clause boundaries    ( ; : , — –  and pan-Indic connector words )
  3. word boundaries      ( never splits a word / Devanagari cluster )

Emotion tags ([laughs], [clears throat], …) are atomic and glued to the word
they condition, so a boundary can never strand a tag into the wrong clip.

BUDGET: MAX chars is sized in SPEAKING SECONDS, but the real ceiling is the
engine's context window: a chunk costs ~50 audio tokens/sec plus ~1.0 (Hindi)
to ~1.5 (Telugu) prompt tokens/char. At max_model_len=1024 a chunk must stay
under ~13 s / ~180 chars; 25 s / 347 chars needs max_model_len=2048. Set
CHUNK_MAX_CHARS to match the deployed window.
"""
import os
import re

_DASH_ELLIPSIS = dict.fromkeys(map(ord, "―‒⸺⸻"), " ")   # bar/figure/2-3em dash -> space
                                                          # (em/en dash and … kept: clause cues)
def post_process(s):
    if not s:
        return s
    s = s.translate(_DASH_ELLIPSIS)
    s = re.sub(r"\.{2,}", "…", s)                # ASCII ".." / "..." -> ellipsis (a sentence cue)
    s = re.sub(r"\s+([,.;:!?।])", r"\1", s)      # drop space before punctuation
    s = re.sub(r"\s{2,}", " ", s)                # collapse runs of spaces
    return s.strip()


SPC = 0.060                                       # measured speaking rate, s/char: 40 clips,
                                                  # 10 languages x 2 voices, mean 0.058
                                                  # (ta/ml ~0.051, or ~0.062 -- 1.54x spread).
                                                  # 0.072 was a guess and over-split by ~20%.
MAX = int(os.environ.get("CHUNK_MAX_CHARS", str(round(20 / SPC))))   # 333 -> ~20.0 s (fits max_model_len=2048)
MIN = int(os.environ.get("CHUNK_MIN_CHARS", "60"))                   # ~4.3 s: merge runts
HARD = MAX                                        # absolute ceiling (except a lone long token)
CHUNK_GAP_S = 0.15                                # silence inserted between concatenated chunks
SECS_PER_CHAR = SPC                               # back-compat alias
CHUNK_MAX_SEC = MAX * SPC                         # back-compat alias

TAG_G = re.compile(r"\[[a-z][a-z ]*\]")                          # bracketed lowercase = TTS tag
SENT_END = re.compile(r"(?:[.!?।॥…]|\.{3})+[\"'”’»)\]]*$")
CLAUSE_END = re.compile(r"[,;:—–][\"'”’»)\]]*$")
_WS = re.compile(r"\s+")
STRIP = "\"'“”‘’«»()[],.;:!?।॥…-–—"
STRIP_LEAD = "\"'“”‘’«»("

CONNECTORS = {
    "and", "but", "because", "while", "so", "then",
    "और", "एवं", "तथा", "लेकिन", "पर", "मगर", "क्योंकि", "जबकि", "फिर", "आणि", "पण", "कारण",
    "এবং", "আর", "কিন্তু", "কারণ", "তারপর", "ଏବଂ", "କିନ୍ତୁ",
    "અને", "પણ", "કારણ", "ਅਤੇ", "ਪਰ", "ਕਿਉਂਕਿ",
    "మరియు", "కానీ", "ఎందుకంటే", "ಮತ್ತು", "ಆದರೆ", "ಏಕೆಂದರೆ",
    "மற்றும்", "ஆனால்", "ஏனெனில்", "പക്ഷേ", "എന്നാൽ", "കാരണം",
}


def _strip(s, chars):
    return s.strip(chars)


def _groups(para):
    """Whitespace tokens -> atomic groups, with emotion tags glued to their word."""
    prot = TAG_G.sub(lambda m: m.group(0).replace(" ", "\x00"), para)
    toks = [t.replace("\x00", " ") for t in prot.split() if t]
    out, glue_right = [], False
    for t in toks:
        bare = _strip(TAG_G.sub("", t).strip(), STRIP)
        is_tag = bare == ""                                   # token is only tag(+punctuation)
        if glue_right:
            out[-1] = out[-1] + " " + t
            glue_right = is_tag                               # consecutive tags keep gluing
            continue
        if is_tag and out and not SENT_END.search(out[-1]):
            out[-1] = out[-1] + " " + t                       # mid/end tag -> preceding word
        elif is_tag:
            out.append(t)                                     # sentence-leading tag -> next word
            glue_right = True
        else:
            out.append(t)
    return out


def _strength(group, nxt):
    """Boundary strength AFTER this group: 3 sentence, 2 clause, 1 word."""
    if SENT_END.search(group):
        return 3
    if CLAUSE_END.search(group):
        return 2
    if nxt is not None:
        parts = nxt.split()
        first = _strip(parts[0], STRIP_LEAD) if parts else ""
        if first in CONNECTORS:
            return 2                                          # break BEFORE a connector word
    return 1


def _runs(gs, min_strength):
    """Join groups into runs, breaking where boundary strength >= min_strength."""
    out, buf = [], []
    for i, g in enumerate(gs):
        buf.append(g)
        nxt = gs[i + 1] if i + 1 < len(gs) else None
        if nxt is not None and _strength(g, nxt) >= min_strength:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out


def _split_words(clause):
    out, buf = [], ""
    for g in _groups(clause):
        cand = (buf + " " + g) if buf else g
        if len(cand) <= HARD or not buf:
            buf = cand
        else:
            out.append(buf)
            buf = g
    if buf:
        out.append(buf)
    return out


def _split_long(sent):
    out, buf = [], ""
    for clause in _runs(_groups(sent), 2):
        if len(clause) > MAX:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split_words(clause))
            continue
        cand = (buf + " " + clause) if buf else clause
        if len(cand) <= MAX:
            buf = cand
        else:
            out.append(buf)
            buf = clause
    if buf:
        out.append(buf)
    return out


def _chunk_groups(gs):
    out, buf = [], ""
    for sent in _runs(gs, 3):
        if len(sent) > MAX:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split_long(sent))
            continue
        cand = (buf + " " + sent) if buf else sent
        if len(cand) <= MAX:
            buf = cand
        else:
            out.append(buf)
            buf = sent
    if buf:
        out.append(buf)
    return out


def _merge_runts(pieces):
    if not pieces:
        return []
    merged = []
    for p in pieces:
        if merged and len(p) < MIN and len(merged[-1] + " " + p) <= HARD:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    if len(merged) >= 2 and len(merged[0]) < MIN and len(merged[0] + " " + merged[1]) <= HARD:
        merged[1] = merged[0] + " " + merged[1]
        merged.pop(0)
    return [_WS.sub(" ", m).strip() for m in merged]


def split_for_tts(text, max_sec=None, spc=SPC, hard_sec=None):
    """Split normalized text into prosody-friendly, tag-safe TTS chunks.

    max_sec/spc/hard_sec are accepted for call-site compatibility with the old
    chunker; sizing is governed by MAX/MIN/HARD (chars) above.
    """
    if not text or not text.strip():
        return []
    pieces = []
    for para in text.split("\n"):
        p = para.strip()
        if not p:
            continue
        pieces.extend(_chunk_groups(_groups(p)))
    return _merge_runts(pieces) or [text.strip()]
'''),
    'mithi_tts.tts': (False, r'''"""Mithi text-to-speech: 11 Indian languages, 10 voices: Satvik, Anirban, Nakul, Vasant, Siddhant (male); Charvi, Ahana, Vaidehi, Malini, Jahnavi (female).

    text -> Maya text normaliser -> espeak-ng / misaki G2P (Kokoro phonemes)
         -> indic-bert word vectors -> prosody writer (durations, pitch, loudness, breaths per phoneme)
         -> Kokoro text encoder + iSTFTNet decoder -> 24 kHz audio

    from mithi_tts import Mithi
    tts = Mithi("path/to/mithi-v2-flash")          # the downloaded model folder
    audio = tts.synthesize("आज मौसम कितना अच्छा है!", lang="hi", voice="charvi")   # numpy array, 24 kHz
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file

from . import frontend as fe
from .indic_encoder import IndicTextEncoder
from .kokoro.istftnet import Decoder
from .kokoro.modules import TextEncoder
from .prosody import P5_COLS, WRITER_LANGS, ProsodyWriter, decode

SR = 24000
from ._here import HERE as DEFAULT_DIR


class KokoroDecoder(nn.Module):
    """The parts of Kokoro-82M this model uses: phoneme text encoder and iSTFTNet decoder."""

    def __init__(self, cfg):
        super().__init__()
        self.text_encoder = TextEncoder(channels=cfg["hidden_dim"], kernel_size=cfg["text_encoder_kernel_size"],
                                        depth=cfg["n_layer"], n_symbols=cfg["n_token"])
        self.decoder = Decoder(dim_in=cfg["hidden_dim"], style_dim=cfg["style_dim"], dim_out=cfg["n_mels"],
                               disable_complex=False, **cfg["istftnet"])


class Mithi:
    LANGS = fe.LANGS

    def __init__(self, model_dir=DEFAULT_DIR, device=None):
        self.dir = model_dir
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.backends.cuda.matmul.allow_tf32 = False
        m = os.path.join(model_dir, "model")
        if not os.path.isdir(m):                               # weights at the top of the model folder
            m = model_dir
        self.cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
        self.settings = dict(self.cfg["inference"])

        kcfg = self.cfg["kokoro"]
        self.vocab = kcfg["vocab"]
        self.kokoro = KokoroDecoder(kcfg)
        self.kokoro.load_state_dict(load_file(os.path.join(m, "decoder.safetensors")))

        with safe_open(os.path.join(m, "writer.safetensors"), "pt") as f:
            meta = f.metadata()
        self.voices = json.loads(meta["voices"])
        self.writer = ProsodyWriter(n_spk=len(self.voices))
        self.writer.load_state_dict(load_file(os.path.join(m, "writer.safetensors")))

        self.text_enc = IndicTextEncoder(os.path.join(m, "indic_encoder"))
        self.kokoro, self.writer, self.text_enc = (x.to(self.dev).eval() for x in (self.kokoro, self.writer, self.text_enc))
        self.packs = {v: t.float().to(self.dev) for v, t in load_file(os.path.join(m, "voices.safetensors")).items()}

    # ------------------------------------------------------------------ one sentence
    @torch.no_grad()
    def synth_sentence(self, text, lang, voice, seed=0):
        s, dev = self.settings, self.dev
        torch.manual_seed(seed)
        norm, ps = fe.phonemes(text, lang)
        ids = torch.LongTensor([0, *[i for i in map(self.vocab.get, ps) if i is not None], 0])
        words = fe.words_of(text, norm)
        tok_word, n_groups = fe.token_words(ids.tolist())
        if n_groups == len(words):
            wv, _ = self.text_enc.word_vectors(" ".join(words))
            tw = torch.tensor(tok_word)
            word_vec, has = wv[tw.clamp(min=0)][None], (tw >= 0)[None].to(dev)
        else:                                               # word / phoneme groups disagree: phonemes only
            word_vec = torch.zeros(1, len(ids), self.text_enc.dim, device=dev)
            has = torch.zeros(1, len(ids), dtype=torch.bool, device=dev)
        x = ids[None].to(dev)
        spk = torch.tensor([self.voices.index(voice)], device=dev)
        H = self.writer.read(x, word_vec, has, spk, torch.tensor([WRITER_LANGS.index(lang)], device=dev),
                             torch.zeros_like(x, dtype=torch.bool))
        head_temp = {k: s["voicing_temp"] for k in ("voiced", "uv0", "uv1")}
        dec_ids, codes = self.writer.sample(H, ids, fe.PAUSE_IDS, temp=s["writer_temp"], top_p=s["writer_top_p"],
                                            head_temp=head_temp)
        codes[[0, -1], 0] = codes[[0, -1], 0].clamp(max=s["edge_max"])      # short silence at both ends
        dur, f0, en = decode(codes[:, :P5_COLS].numpy())
        f0 = f0 * np.float32(2 ** (s["pitch_shift"] / 12))
        y = dec_ids[None].to(dev)
        T = int(dur.sum())
        aln = torch.zeros(1, y.shape[1], T, device=dev)
        aln[0, torch.repeat_interleave(torch.arange(y.shape[1], device=dev), torch.from_numpy(dur).to(dev)),
            torch.arange(T, device=dev)] = 1
        ref = self.packs[voice][min(len(ps), len(self.packs[voice])) - 1]
        h = self.kokoro.text_encoder(y, torch.tensor([y.shape[1]], device=dev), torch.zeros_like(y, dtype=torch.bool)) @ aln
        audio = self.kokoro.decoder(h, torch.from_numpy(f0)[None].to(dev), torch.from_numpy(en)[None].to(dev), ref[:, :128])
        return audio.squeeze().float().cpu().numpy()

    # ------------------------------------------------------------------ any length
    def split(self, text, lang):
        """The pieces synthesize() reads: the whole text, or sentences / comma pieces when it is long."""
        return fe.split_text(text, lang, self.settings["max_phonemes_per_piece"])

    def synthesize(self, text, lang="hi", voice="charvi", seed=None, gap=0.15):
        """Text of any length -> audio at 24 kHz in [-1, 1]. Long text is read sentence by sentence."""
        if lang not in fe.LANGS:
            raise ValueError(f"unknown language {lang} (one of {', '.join(fe.LANGS)})")
        if voice not in self.voices:
            raise ValueError(f"unknown voice {voice} (one of {', '.join(self.voices)})")
        if not text or not text.strip():
            raise ValueError("empty text")
        seed = self.settings["seed"] if seed is None else seed
        silence = np.zeros(int(gap * SR), np.float32)
        segs = []
        for p in self.split(text, lang):
            segs += ([silence] if segs else []) + [self.synth_sentence(p, lang, voice, seed).astype(np.float32)]
        return np.clip(np.concatenate(segs), -1.0, 1.0)

    def save(self, audio, path):
        import soundfile as sf
        sf.write(path, audio, SR, subtype="PCM_16")
'''),
}
_SOURCES["mithi_tts._here"] = (False, "HERE = " + repr(HERE) + "\n")


class _Embedded(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name not in _SOURCES:
            return None
        return importlib.util.spec_from_loader(name, self, is_package=_SOURCES[name][0])

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        is_pkg, src = _SOURCES[module.__name__]
        fname = os.path.join(HERE, *module.__name__.split(".")) + ("/__init__.py" if is_pkg else ".py")
        module.__file__ = fname
        if is_pkg:
            module.__path__ = [os.path.dirname(fname)]
        linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)     # readable tracebacks
        exec(compile(src, fname, "exec"), module.__dict__)


if not any(isinstance(f, _Embedded) for f in sys.meta_path):
    sys.meta_path.insert(0, _Embedded())

from mithi_tts import SR, Mithi  # noqa: E402,F401


# ---------------------------------------------------------------- command line
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="text to speak")
    ap.add_argument("--lang", default="hi", help="hi en bn gu kn ml mr or pa ta te")
    ap.add_argument("--voice", default="charvi", help="satvik anirban nakul vasant siddhant (male), charvi ahana vaidehi malini jahnavi (female)")
    ap.add_argument("--out", default="output.wav", help="output .wav file")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: PyTorch's choice, usually all cores)")
    args = ap.parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    tts = Mithi(HERE, device=args.device)
    t0 = time.time()
    audio = tts.synthesize(args.text, lang=args.lang, voice=args.voice)
    tts.save(audio, args.out)
    print(f"{args.out}: {len(audio) / SR:.2f} s of audio in {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
