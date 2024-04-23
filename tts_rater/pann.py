import math
import os
import pathlib

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def move_data_to_device(x, device):
    if "float" in str(x.dtype):
        x = torch.Tensor(x)
    elif "int" in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


class Interpolator(nn.Module):
    def __init__(self, ratio, interpolate_mode="nearest"):
        """Interpolate the sound event detection result along the time axis.

        Args:
            ratio: int
            interpolate_mode: str

        """
        super(Interpolator, self).__init__()

        if interpolate_mode == "nearest":
            self.interpolator = NearestInterpolator(ratio)

    def forward(self, x):
        """Interpolate the sound event detection result along the time axis.

        Args:
            x: (batch_size, time_steps, classes_num)

        Returns:
            (batch_size, new_time_steps, classes_num)
        """
        return self.interpolator(x)


class NearestInterpolator(nn.Module):
    def __init__(self, ratio):
        """Nearest interpolate the sound event detection result along the time axis.

        Args:
            ratio: int
        """
        super(NearestInterpolator, self).__init__()

        self.ratio = ratio

    def forward(self, x):
        """Interpolate the sound event detection result along the time axis.

        Args:
            x: (batch_size, time_steps, classes_num)

        Returns:
            upsampled: (batch_size, new_time_steps, classes_num)
        """
        (batch_size, time_steps, classes_num) = x.shape
        upsampled = x[:, :, None, :].repeat(1, 1, self.ratio, 1)
        upsampled = upsampled.reshape(batch_size, time_steps * self.ratio, classes_num)
        return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def do_mixup(x, mixup_lambda):
    out = x[0::2].transpose(0, -1) * mixup_lambda[0::2] + x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    return out.transpose(0, -1)


class DFTBase(nn.Module):
    def __init__(self):
        r"""Base class for DFT and IDFT matrix."""
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W


class DFT(DFTBase):
    def __init__(self, n, norm):
        r"""Calculate discrete Fourier transform (DFT), inverse DFT (IDFT,
        right DFT (RDFT) RDFT, and inverse RDFT (IRDFT.)

        Args:
          n: fft window size
          norm: None | 'ortho'
        """
        super(DFT, self).__init__()

        self.W = self.dft_matrix(n)
        self.inv_W = self.idft_matrix(n)

        self.W_real = torch.Tensor(np.real(self.W))
        self.W_imag = torch.Tensor(np.imag(self.W))
        self.inv_W_real = torch.Tensor(np.real(self.inv_W))
        self.inv_W_imag = torch.Tensor(np.imag(self.inv_W))

        self.n = n
        self.norm = norm

    def dft(self, x_real, x_imag):
        r"""Calculate DFT of a signal.

        Args:
            x_real: (n,), real part of a signal
            x_imag: (n,), imag part of a signal

        Returns:
            z_real: (n,), real part of output
            z_imag: (n,), imag part of output
        """
        z_real = torch.matmul(x_real, self.W_real) - torch.matmul(x_imag, self.W_imag)
        z_imag = torch.matmul(x_imag, self.W_real) + torch.matmul(x_real, self.W_imag)
        # shape: (n,)

        if self.norm is None:
            pass
        elif self.norm == "ortho":
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

    def idft(self, x_real, x_imag):
        r"""Calculate IDFT of a signal.

        Args:
            x_real: (n,), real part of a signal
            x_imag: (n,), imag part of a signal
        Returns:
            z_real: (n,), real part of output
            z_imag: (n,), imag part of output
        """
        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(x_imag, self.inv_W_imag)
        z_imag = torch.matmul(x_imag, self.inv_W_real) + torch.matmul(x_real, self.inv_W_imag)
        # shape: (n,)

        if self.norm is None:
            z_real /= self.n
        elif self.norm == "ortho":
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

    def rdft(self, x_real):
        r"""Calculate right RDFT of signal.

        Args:
            x_real: (n,), real part of a signal
            x_imag: (n,), imag part of a signal

        Returns:
            z_real: (n // 2 + 1,), real part of output
            z_imag: (n // 2 + 1,), imag part of output
        """
        n_rfft = self.n // 2 + 1
        z_real = torch.matmul(x_real, self.W_real[..., 0:n_rfft])
        z_imag = torch.matmul(x_real, self.W_imag[..., 0:n_rfft])
        # shape: (n // 2 + 1,)

        if self.norm is None:
            pass
        elif self.norm == "ortho":
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

    def irdft(self, x_real, x_imag):
        r"""Calculate IRDFT of signal.

        Args:
            x_real: (n // 2 + 1,), real part of a signal
            x_imag: (n // 2 + 1,), imag part of a signal

        Returns:
            z_real: (n,), real part of output
            z_imag: (n,), imag part of output
        """
        n_rfft = self.n // 2 + 1

        flip_x_real = torch.flip(x_real, dims=(-1,))
        flip_x_imag = torch.flip(x_imag, dims=(-1,))
        # shape: (n // 2 + 1,)

        x_real = torch.cat((x_real, flip_x_real[..., 1 : n_rfft - 1]), dim=-1)
        x_imag = torch.cat((x_imag, -1.0 * flip_x_imag[..., 1 : n_rfft - 1]), dim=-1)
        # shape: (n,)

        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(x_imag, self.inv_W_imag)
        # shape: (n,)

        if self.norm is None:
            z_real /= self.n
        elif self.norm == "ortho":
            z_real /= math.sqrt(self.n)

        return z_real


class STFT(DFTBase):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
    ):
        r"""PyTorch implementation of STFT with Conv1d. The function has the
        same output as librosa.stft.

        Args:
            n_fft: int, fft window size, e.g., 2048
            hop_length: int, hop length samples, e.g., 441
            win_length: int, window length e.g., 2048
            window: str, window function name, e.g., 'hann'
            center: bool
            pad_mode: str, e.g., 'reflect'
            freeze_parameters: bool, set to True to freeze all parameters. Set
                to False to finetune all parameters.
        """
        super(STFT, self).__init__()

        assert pad_mode in ["constant", "reflect"]

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = n_fft

        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        fft_window = librosa.filters.get_window(window, self.win_length, fftbins=True)

        # Pad the window out to n_fft size.
        fft_window = librosa.util.pad_center(data=fft_window, size=n_fft)

        # DFT & IDFT matrix.
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=n_fft,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.conv_imag = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=n_fft,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # Initialize Conv1d weights.
        self.conv_real.weight.data = torch.Tensor(np.real(self.W[:, 0:out_channels] * fft_window[:, None]).T)[
            :, None, :
        ].contiguous()
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(np.imag(self.W[:, 0:out_channels] * fft_window[:, None]).T)[
            :, None, :
        ].contiguous()
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        r"""Calculate STFT of batch of signals.

        Args:
            input: (batch_size, data_length), input signals.

        Returns:
            real: (batch_size, 1, time_steps, n_fft // 2 + 1)
            imag: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        x = input[:, None, :]  # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag


def magphase(real, imag):
    r"""Calculate magnitude and phase from real and imag part of signals.

    Args:
        real: tensor, real part of signals
        imag: tensor, imag part of signals

    Returns:
        mag: tensor, magnitude of signals
        cos: tensor, cosine of phases of signals
        sin: tensor, sine of phases of signals
    """
    mag = (real**2 + imag**2) ** 0.5
    cos = real / torch.clamp(mag, 1e-10, np.inf)
    sin = imag / torch.clamp(mag, 1e-10, np.inf)

    return mag, cos, sin


class ISTFT(DFTBase):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
        onnx=False,
        frames_num=None,
        device=None,
    ):
        """PyTorch implementation of ISTFT with Conv1d. The function has the
        same output as librosa.istft.

        Args:
            n_fft: int, fft window size, e.g., 2048
            hop_length: int, hop length samples, e.g., 441
            win_length: int, window length e.g., 2048
            window: str, window function name, e.g., 'hann'
            center: bool
            pad_mode: str, e.g., 'reflect'
            freeze_parameters: bool, set to True to freeze all parameters. Set
                to False to finetune all parameters.
            onnx: bool, set to True when exporting trained model to ONNX. This
                will replace several operations to operators supported by ONNX.
            frames_num: None | int, number of frames of audio clips to be
                inferneced. Only useable when onnx=True.
            device: None | str, device of ONNX. Only useable when onnx=True.
        """
        super(ISTFT, self).__init__()

        assert pad_mode in ["constant", "reflect"]

        if not onnx:
            assert frames_num is None, "When onnx=False, frames_num must be None!"
            assert device is None, "When onnx=False, device must be None!"

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.onnx = onnx

        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = self.n_fft

        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        # Initialize Conv1d modules for calculating real and imag part of DFT.
        self.init_real_imag_conv()

        # Initialize overlap add window for reconstruct time domain signals.
        self.init_overlap_add_window()

        if self.onnx:
            # Initialize ONNX modules.
            self.init_onnx_modules(frames_num, device)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def init_real_imag_conv(self):
        r"""Initialize Conv1d for calculating real and imag part of DFT."""
        self.W = self.idft_matrix(self.n_fft) / self.n_fft

        self.conv_real = nn.Conv1d(
            in_channels=self.n_fft,
            out_channels=self.n_fft,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.conv_imag = nn.Conv1d(
            in_channels=self.n_fft,
            out_channels=self.n_fft,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        ifft_window = librosa.filters.get_window(self.window, self.win_length, fftbins=True)
        # (win_length,)

        # Pad the window to n_fft
        ifft_window = librosa.util.pad_center(data=ifft_window, size=self.n_fft)

        self.conv_real.weight.data = torch.Tensor(np.real(self.W * ifft_window[None, :]).T)[:, :, None].contiguous()
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(np.imag(self.W * ifft_window[None, :]).T)[:, :, None].contiguous()
        # (n_fft // 2 + 1, 1, n_fft)

    def init_overlap_add_window(self):
        r"""Initialize overlap add window for reconstruct time domain signals."""

        ola_window = librosa.filters.get_window(self.window, self.win_length, fftbins=True)
        # (win_length,)

        ola_window = librosa.util.normalize(ola_window, norm=None) ** 2
        ola_window = librosa.util.pad_center(data=ola_window, size=self.n_fft)
        ola_window = torch.Tensor(ola_window)

        self.register_buffer("ola_window", ola_window)
        # (win_length,)

    def init_onnx_modules(self, frames_num, device):
        r"""Initialize ONNX modules.

        Args:
            frames_num: int
            device: str | None
        """

        # Use Conv1d to implement torch.flip(), because torch.flip() is not
        # supported by ONNX.
        self.reverse = nn.Conv1d(
            in_channels=self.n_fft // 2 + 1, out_channels=self.n_fft // 2 - 1, kernel_size=1, bias=False
        )

        tmp = np.zeros((self.n_fft // 2 - 1, self.n_fft // 2 + 1, 1))
        tmp[:, 1:-1, 0] = np.array(np.eye(self.n_fft // 2 - 1)[::-1])
        self.reverse.weight.data = torch.Tensor(tmp).contiguous()
        # (n_fft // 2 - 1, n_fft // 2 + 1, 1)

        # Use nn.ConvTranspose2d to implement torch.nn.functional.fold(),
        # because torch.nn.functional.fold() is not supported by ONNX.
        self.overlap_add = nn.ConvTranspose2d(
            in_channels=self.n_fft, out_channels=1, kernel_size=(self.n_fft, 1), stride=(self.hop_length, 1), bias=False
        )

        self.overlap_add.weight.data = torch.Tensor(np.eye(self.n_fft)[:, None, :, None]).contiguous()
        # (n_fft, 1, n_fft, 1)

        if frames_num:
            # Pre-calculate overlap-add window sum for reconstructing signals
            # when using ONNX.
            self.ifft_window_sum = self._get_ifft_window_sum_onnx(frames_num, device)
        else:
            self.ifft_window_sum = []

    def forward(self, real_stft, imag_stft, length):
        r"""Calculate inverse STFT.

        Args:
            real_stft: (batch_size, channels=1, time_steps, n_fft // 2 + 1)
            imag_stft: (batch_size, channels=1, time_steps, n_fft // 2 + 1)
            length: int

        Returns:
            real: (batch_size, data_length), output signals.
        """
        assert real_stft.ndimension() == 4 and imag_stft.ndimension() == 4
        batch_size, _, frames_num, _ = real_stft.shape

        real_stft = real_stft[:, 0, :, :].transpose(1, 2)
        imag_stft = imag_stft[:, 0, :, :].transpose(1, 2)
        # (batch_size, n_fft // 2 + 1, time_steps)

        # Get full stft representation from spectrum using symmetry attribute.
        if self.onnx:
            full_real_stft, full_imag_stft = self._get_full_stft_onnx(real_stft, imag_stft)
        else:
            full_real_stft, full_imag_stft = self._get_full_stft(real_stft, imag_stft)
        # full_real_stft: (batch_size, n_fft, time_steps)
        # full_imag_stft: (batch_size, n_fft, time_steps)

        # Calculate IDFT frame by frame.
        s_real = self.conv_real(full_real_stft) - self.conv_imag(full_imag_stft)
        # (batch_size, n_fft, time_steps)

        # Overlap add signals in frames to reconstruct signals.
        if self.onnx:
            y = self._overlap_add_divide_window_sum_onnx(s_real, frames_num)
        else:
            y = self._overlap_add_divide_window_sum(s_real, frames_num)
        # y: (batch_size, audio_samples + win_length,)

        y = self._trim_edges(y, length)
        # (batch_size, audio_samples,)

        return y

    def _get_full_stft(self, real_stft, imag_stft):
        r"""Get full stft representation from spectrum using symmetry attribute.

        Args:
            real_stft: (batch_size, n_fft // 2 + 1, time_steps)
            imag_stft: (batch_size, n_fft // 2 + 1, time_steps)

        Returns:
            full_real_stft: (batch_size, n_fft, time_steps)
            full_imag_stft: (batch_size, n_fft, time_steps)
        """
        full_real_stft = torch.cat((real_stft, torch.flip(real_stft[:, 1:-1, :], dims=[1])), dim=1)
        full_imag_stft = torch.cat((imag_stft, -torch.flip(imag_stft[:, 1:-1, :], dims=[1])), dim=1)

        return full_real_stft, full_imag_stft

    def _get_full_stft_onnx(self, real_stft, imag_stft):
        r"""Get full stft representation from spectrum using symmetry attribute
        for ONNX. Replace several pytorch operations in self._get_full_stft()
        that are not supported by ONNX.

        Args:
            real_stft: (batch_size, n_fft // 2 + 1, time_steps)
            imag_stft: (batch_size, n_fft // 2 + 1, time_steps)

        Returns:
            full_real_stft: (batch_size, n_fft, time_steps)
            full_imag_stft: (batch_size, n_fft, time_steps)
        """

        # Implement torch.flip() with Conv1d.
        full_real_stft = torch.cat((real_stft, self.reverse(real_stft)), dim=1)
        full_imag_stft = torch.cat((imag_stft, -self.reverse(imag_stft)), dim=1)

        return full_real_stft, full_imag_stft

    def _overlap_add_divide_window_sum(self, s_real, frames_num):
        r"""Overlap add signals in frames to reconstruct signals.

        Args:
            s_real: (batch_size, n_fft, time_steps), signals in frames
            frames_num: int

        Returns:
            y: (batch_size, audio_samples)
        """

        output_samples = (s_real.shape[-1] - 1) * self.hop_length + self.win_length
        # (audio_samples,)

        # Overlap-add signals in frames to signals. Ref:
        # asteroid_filterbanks.torch_stft_fb.torch_stft_fb() from
        # https://github.com/asteroid-team/asteroid-filterbanks
        y = torch.nn.functional.fold(
            input=s_real, output_size=(1, output_samples), kernel_size=(1, self.win_length), stride=(1, self.hop_length)
        )
        # (batch_size, 1, 1, audio_samples,)

        y = y[:, 0, 0, :]
        # (batch_size, audio_samples)

        # Get overlap-add window sum to be divided.
        ifft_window_sum = self._get_ifft_window(frames_num)
        # (audio_samples,)

        # Following code is abandaned for divide overlap-add window, because
        # not supported by half precision training and ONNX.
        # min_mask = ifft_window_sum.abs() < 1e-11
        # y[:, ~min_mask] = y[:, ~min_mask] / ifft_window_sum[None, ~min_mask]
        # # (batch_size, audio_samples)

        ifft_window_sum = torch.clamp(ifft_window_sum, 1e-11, np.inf)
        # (audio_samples,)

        y = y / ifft_window_sum[None, :]
        # (batch_size, audio_samples,)

        return y

    def _get_ifft_window(self, frames_num):
        r"""Get overlap-add window sum to be divided.

        Args:
            frames_num: int

        Returns:
            ifft_window_sum: (audio_samlpes,), overlap-add window sum to be
            divided.
        """

        output_samples = (frames_num - 1) * self.hop_length + self.win_length
        # (audio_samples,)

        window_matrix = self.ola_window[None, :, None].repeat(1, 1, frames_num)
        # (batch_size, win_length, time_steps)

        ifft_window_sum = F.fold(
            input=window_matrix,
            output_size=(1, output_samples),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )
        # (1, 1, 1, audio_samples)

        ifft_window_sum = ifft_window_sum.squeeze()
        # (audio_samlpes,)

        return ifft_window_sum

    def _overlap_add_divide_window_sum_onnx(self, s_real, frames_num):
        r"""Overlap add signals in frames to reconstruct signals for ONNX.
        Replace several pytorch operations in
        self._overlap_add_divide_window_sum() that are not supported by ONNX.

        Args:
            s_real: (batch_size, n_fft, time_steps), signals in frames
            frames_num: int

        Returns:
            y: (batch_size, audio_samples)
        """

        s_real = s_real[..., None]
        # (batch_size, n_fft, time_steps, 1)

        # Implement overlap-add with Conv1d, because torch.nn.functional.fold()
        # is not supported by ONNX.
        y = self.overlap_add(s_real)[:, 0, :, 0]
        # y: (batch_size, samples_num)

        if len(self.ifft_window_sum) != y.shape[1]:
            device = s_real.device

            self.ifft_window_sum = self._get_ifft_window_sum_onnx(frames_num, device)
            # (audio_samples,)

        # Use torch.clamp() to prevent from underflow to make sure all
        # operations are supported by ONNX.
        ifft_window_sum = torch.clamp(self.ifft_window_sum, 1e-11, np.inf)
        # (audio_samples,)

        y = y / ifft_window_sum[None, :]
        # (batch_size, audio_samples,)

        return y

    def _get_ifft_window_sum_onnx(self, frames_num, device):
        r"""Pre-calculate overlap-add window sum for reconstructing signals when
        using ONNX.

        Args:
            frames_num: int
            device: str | None

        Returns:
            ifft_window_sum: (audio_samples,)
        """

        ifft_window_sum = librosa.filters.window_sumsquare(
            window=self.window,
            n_frames=frames_num,
            win_length=self.win_length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        # (audio_samples,)

        ifft_window_sum = torch.Tensor(ifft_window_sum)

        if device:
            ifft_window_sum = ifft_window_sum.to(device)

        return ifft_window_sum

    def _trim_edges(self, y, length):
        r"""Trim audio.

        Args:
            y: (audio_samples,)
            length: int

        Returns:
            (trimmed_audio_samples,)
        """
        # Trim or pad to length
        if length is None:
            if self.center:
                y = y[:, self.n_fft // 2 : -self.n_fft // 2]
        else:
            if self.center:
                start = self.n_fft // 2
            else:
                start = 0

            y = y[:, start : start + length]

        return y


class Spectrogram(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        freeze_parameters=True,
    ):
        r"""Calculate spectrogram using pytorch. The STFT is implemented with
        Conv1d. The function has the same output of librosa.stft
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def forward(self, input):
        r"""Calculate spectrogram of input signals.
        Args:
            input: (batch_size, data_length)

        Returns:
            spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real**2 + imag**2

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (self.power / 2.0)

        return spectrogram


class LogmelFilterBank(nn.Module):
    def __init__(
        self,
        sr=22050,
        n_fft=2048,
        n_mels=64,
        fmin=0.0,
        fmax=None,
        is_log=True,
        ref=1.0,
        amin=1e-10,
        top_db=80.0,
        freeze_parameters=True,
    ):
        r"""Calculate logmel spectrogram using pytorch. The mel filter bank is
        the pytorch implementation of as librosa.filters.mel
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        if fmax == None:
            fmax = sr // 2

        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        r"""Calculate (log) mel spectrogram from spectrogram.

        Args:
            input: (*, n_fft), spectrogram

        Returns:
            output: (*, mel_bins), (log) mel spectrogram
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)
        # (*, mel_bins)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output

    def power_to_db(self, input):
        r"""Power to db, this function is the pytorch implementation of
        librosa.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise librosa.util.exceptions.ParameterError("top_db must be non-negative")
            log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db, max=np.inf)

        return log_spec


class Enframe(nn.Module):
    def __init__(self, frame_length=2048, hop_length=512):
        r"""Enframe a time sequence. This function is the pytorch implementation
        of librosa.util.frame
        """
        super(Enframe, self).__init__()

        self.enframe_conv = nn.Conv1d(
            in_channels=1, out_channels=frame_length, kernel_size=frame_length, stride=hop_length, padding=0, bias=False
        )

        self.enframe_conv.weight.data = torch.Tensor(torch.eye(frame_length)[:, None, :]).contiguous()
        self.enframe_conv.weight.requires_grad = False

    def forward(self, input):
        r"""Enframe signals into frames.
        Args:
            input: (batch_size, samples)

        Returns:
            output: (batch_size, window_length, frames_num)
        """
        output = self.enframe_conv(input[:, None, :])
        return output

    def power_to_db(self, input):
        r"""Power to db, this function is the pytorch implementation of
        librosa.power_to_lb.
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise librosa.util.exceptions.ParameterError("top_db must be non-negative")
            log_spec = torch.clamp(log_spec, min=log_spec.max() - self.top_db, max=np.inf)

        return log_spec


class Scalar(nn.Module):
    def __init__(self, scalar, freeze_parameters):
        super(Scalar, self).__init__()

        self.scalar_mean = Parameter(torch.Tensor(scalar["mean"]))
        self.scalar_std = Parameter(torch.Tensor(scalar["std"]))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        return (input - self.scalar_mean) / self.scalar_std


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):

        super(Cnn14, self).__init__()

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)

        return embedding


class PANNModel:
    def __init__(self, checkpoint_path=None, device="cuda"):
        """Audio tagging inference wrapper."""
        if not checkpoint_path:
            checkpoint_path = pathlib.Path(__file__).parent / "pann/Cnn14_mAP=0.431.pth"
        if not checkpoint_path.exists():
            checkpoint_path.parent.mkdir(exist_ok=True)
            dl_path = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
            os.system('wget -O "{}" "{}"'.format(checkpoint_path, dl_path))

        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Model
        self.model = Cnn14(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

        # Parallel
        if "cuda" in str(self.device):
            self.model.to(self.device)
            self.model = torch.nn.DataParallel(self.model)
        else:
            print("Using CPU.")

    def get_embedding(self, audio):
        audio = move_data_to_device(audio, self.device)

        with torch.inference_mode():
            self.model.eval()
            embedding = self.model(audio)

        return embedding
