# signal_utils.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.signal import welch, butter, filtfilt
from typing import Dict
from scipy.signal import welch
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.signal import find_peaks

def detrend(signal, Lambda):
    """
    Tarvainen et al. (2002)의 detrending 방법.

    Parameters:
      signal (1d np.array): 입력 신호.
      Lambda (int): 스무딩 파라미터.

    Returns:
      filtered_signal (1d np.array): detrend된 신호.
    """
    signal_length = len(signal)
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def BPF(input_val, fs=30, low=0.75, high=3.0):
    """
    Butterworth 밴드패스 필터.

    Parameters:
      input_val (np.array): 입력 신호.
      fs (int): 샘플링 주파수 (Hz).
      low (float): low cutoff (Hz).
      high (float): high cutoff (Hz).

    Returns:
      filtered_signal (np.array): 필터링된 신호.
    """
    low_norm = low / (0.5 * fs)
    high_norm = high / (0.5 * fs)
    b, a = butter(6, [low_norm, high_norm], btype='bandpass')
    return filtfilt(b, a, np.double(input_val))


def z_normalize(x):
    """z-normalization: (x - 평균) / 표준편차"""
    return (x - np.mean(x)) / np.std(x)


def process_signal(signal, fs=30, detrend_on=True, bpf_on=True, Lambda=100):
    """
    신호 전처리 함수.

    Args:
      signal (np.array): 입력 신호.
      fs (int): 샘플링 주파수.
      detrend_on (bool): True이면 detrend 적용.
      bpf_on (bool): True이면 BPF 적용.
      Lambda (int): detrend 파라미터.

    Returns:
      np.array: 전처리 및 z-normalization된 신호.
    """
    proc = signal.copy()
    if detrend_on:
        proc = detrend(proc, Lambda)
    if bpf_on:
        proc = BPF(proc, fs=fs)
    return z_normalize(proc)


def split_by_subjects(preds, labels, subject_counts, segment_length=300):
    """
    preds와 labels 배열을 subject별로 분할.

    각 subject의 길이는 subject_counts에 기록된 숫자에 segment_length를 곱한 값입니다.

    Args:
      preds (np.array): 전체 예측 배열.
      labels (np.array): 전체 레이블 배열.
      subject_counts (list[int]): subject별 곱셈 숫자 리스트.
      segment_length (int): 단위 길이.

    Returns:
      tuple: (subject별 preds 리스트, subject별 labels 리스트)
    """
    split_preds = []
    split_labels = []
    start = 0
    for count in subject_counts:
        length = count * segment_length
        end = start + length
        split_preds.append(preds[start:end])
        split_labels.append(labels[start:end])
        start = end
    return split_preds, split_labels


def plot_subject_signals(subject_id, p_signal, l_signal, fs=30):
    """
    subject의 전처리된 예측 및 레이블 신호에 대해 시간 도메인 및 PSD 플롯을 생성합니다.

    Args:
      subject_id (int): subject 번호.
      p_signal (np.array): 전처리된 예측 신호 (z-normalized).
      l_signal (np.array): 전처리된 레이블 신호 (z-normalized).
      fs (int): 샘플링 주파수.
    """
    nperseg = 256 if len(p_signal) >= 256 else len(p_signal)
    freqs_pred, psd_pred = welch(p_signal, fs=fs, nperseg=nperseg)
    freqs_label, psd_label = welch(l_signal, fs=fs, nperseg=nperseg)

    peak_idx_pred = np.argmax(psd_pred)
    dominant_freq_pred = freqs_pred[peak_idx_pred]
    heart_rate_pred = dominant_freq_pred * 60

    peak_idx_label = np.argmax(psd_label)
    dominant_freq_label = freqs_label[peak_idx_label]
    heart_rate_label = dominant_freq_label * 60

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(p_signal, label='Prediction', color='blue')
    axs[0].plot(l_signal, label='Label', color='red', alpha=0.7)
    axs[0].set_title(f"Subject {subject_id} - Time Domain (z-normalized)")
    axs[0].set_xlabel("Time (frame)")
    axs[0].set_ylabel("Amplitude (z-score)")
    axs[0].legend()

    axs[1].plot(freqs_pred, psd_pred, label='Prediction PSD', color='blue')
    axs[1].plot(freqs_label, psd_label, label='Label PSD', color='red', alpha=0.7)
    axs[1].axvline(x=dominant_freq_pred, color='green', linestyle='--',
                   label=f'Pred Dominant: {dominant_freq_pred:.2f} Hz\nHR: {heart_rate_pred:.2f} BPM')
    axs[1].axvline(x=dominant_freq_label, color='magenta', linestyle='--',
                   label=f'Label Dominant: {dominant_freq_label:.2f} Hz\nHR: {heart_rate_label:.2f} BPM')
    axs[1].set_title("Power Spectral Density (PSD)")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("PSD")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plot_subject_signals_chunks(subject_id, p_signal, l_signal, fs=30, eval_seconds=-1,
                                detrend_on=True, bpf_on=True, Lambda=100):
    """
    초 단위(eval_seconds)로 청크 분할하여 각 청크별로 전처리 후 시간 도메인 및 PSD 플롯을 생성합니다.
    만약 eval_seconds가 -1이면 전체 신호에 대해 플롯합니다.

    Args:
      subject_id (int): subject 번호.
      p_signal (np.array): 원본 예측 신호.
      l_signal (np.array): 원본 레이블 신호.
      fs (int): 샘플링 주파수.
      eval_seconds (int): 청크 길이(초). -1이면 전체 신호.
      detrend_on (bool): detrend 적용 여부.
      bpf_on (bool): BPF 적용 여부.
      Lambda (int): detrend 파라미터.
    """
    if eval_seconds == -1:
        p_processed = process_signal(p_signal, fs=fs, detrend_on=detrend_on, bpf_on=bpf_on, Lambda=Lambda)
        l_processed = z_normalize(l_signal)
        plot_subject_signals(subject_id, p_processed, l_processed, fs=fs)
    else:
        chunk_length = eval_seconds * fs
        n_chunks = int(np.ceil(len(p_signal) / chunk_length))
        fig, axs = plt.subplots(n_chunks, 2, figsize=(12, 4 * n_chunks))
        if n_chunks == 1:
            axs = np.array([axs])
        for i in range(n_chunks):
            start = i * chunk_length
            end = min((i + 1) * chunk_length, len(p_signal))
            p_chunk = p_signal[start:end]
            l_chunk = l_signal[start:end]
            p_processed = process_signal(p_chunk, fs=fs, detrend_on=detrend_on, bpf_on=bpf_on, Lambda=Lambda)
            l_processed = z_normalize(l_chunk)

            # 시간 도메인 플롯
            axs[i, 0].plot(p_processed, label='Prediction', color='blue')
            axs[i, 0].plot(l_processed, label='Label', color='red', alpha=0.7)
            axs[i, 0].set_title(f"Subject {subject_id} - Time Domain (Segment {i + 1})")
            axs[i, 0].set_xlabel("Time (frame)")
            axs[i, 0].set_ylabel("Amplitude (z-score)")
            axs[i, 0].legend()

            # PSD 플롯
            nperseg = 256 if len(p_processed) >= 256 else len(p_processed)
            freqs_pred, psd_pred = welch(p_processed, fs=fs, nperseg=nperseg)
            freqs_label, psd_label = welch(l_processed, fs=fs, nperseg=nperseg)
            peak_idx_pred = np.argmax(psd_pred)
            dominant_freq_pred = freqs_pred[peak_idx_pred]
            heart_rate_pred = dominant_freq_pred * 60
            peak_idx_label = np.argmax(psd_label)
            dominant_freq_label = freqs_label[peak_idx_label]
            heart_rate_label = dominant_freq_label * 60

            axs[i, 1].plot(freqs_pred, psd_pred, label='Prediction PSD', color='blue')
            axs[i, 1].plot(freqs_label, psd_label, label='Label PSD', color='red', alpha=0.7)
            axs[i, 1].axvline(x=dominant_freq_pred, color='green', linestyle='--',
                              label=f'Pred Dominant: {dominant_freq_pred:.2f} Hz\nHR: {heart_rate_pred:.2f} BPM')
            axs[i, 1].axvline(x=dominant_freq_label, color='magenta', linestyle='--',
                              label=f'Label Dominant: {dominant_freq_label:.2f} Hz\nHR: {heart_rate_label:.2f} BPM')
            axs[i, 1].set_title(f"Subject {subject_id} - PSD (Segment {i + 1})")
            axs[i, 1].set_xlabel("Frequency (Hz)")
            axs[i, 1].set_ylabel("PSD")
            axs[i, 1].legend()
        plt.tight_layout()
        plt.show()

def plot_multi_subject_signals_chunks(
    subject_id: int,
    preds_dict: Dict[str, np.ndarray],
    l_signal: np.ndarray,
    fs: int = 30,
    eval_seconds: int = -1,
    detrend_on: bool = True,
    bpf_on: bool = True,
    Lambda: int = 100
):
    """
    여러 모델(preds_dict)의 예측과 단일 label을,
    청크별로 Time-domain & PSD 비교 플롯.
    """
    cmap = get_cmap("tab10")
    model_names = list(preds_dict.keys())


    # 같은 파일 내 정의된 process_signal, z_normalize 함수를 직접 사용
    def proc(x):
        return process_signal(x, fs=fs, detrend_on=detrend_on, bpf_on=bpf_on, Lambda=Lambda)

    # 전체 또는 청크 개수 계산
    if eval_seconds == -1:
        # 전체 시그널
        fig, axs = plt.subplots(2, 1, figsize=(10,8))
        # Time-domain
        for i,name in enumerate(model_names):
            axs[0].plot(proc(preds_dict[name]), label=name, color=cmap(i))
        axs[0].plot(z_normalize(l_signal), '--', label="Label", color='black')
        axs[0].set_title(f"Subject {subject_id} - Time Domain (전체)")
        axs[0].legend()

        # PSD
        for i,name in enumerate(model_names):
            f,p = welch(proc(preds_dict[name]), fs=fs, nperseg=min(256,len(l_signal)))
            axs[1].plot(f, p, label=f"{name} PSD", color=cmap(i))
        f_l, p_l = welch(z_normalize(l_signal), fs=fs, nperseg=min(256,len(l_signal)))
        axs[1].plot(f_l, p_l, '--', label="Label PSD", color='black')
        axs[1].set_title("Power Spectral Density")
        axs[1].legend()
        plt.tight_layout()
        plt.show()

    else:
        chunk_len = eval_seconds * fs
        n_chunks = int(np.ceil(len(l_signal)/chunk_len))
        fig, axs = plt.subplots(n_chunks, 2, figsize=(12,4*n_chunks))
        if n_chunks==1: axs = np.array([axs])
        for seg in range(n_chunks):
            s,e = seg*chunk_len, min((seg+1)*chunk_len, len(l_signal))
            # Time-domain
            for i,name in enumerate(model_names):
                axs[seg,0].plot(proc(preds_dict[name][s:e]), label=name, color=cmap(i))
            axs[seg,0].plot(z_normalize(l_signal[s:e]), '--', label="Label", color='black')
            axs[seg,0].set_title(f"Subject {subject_id} - Segment {seg+1} Time")
            axs[seg,0].legend()

            # PSD
            for i,name in enumerate(model_names):
                f,p = welch(proc(preds_dict[name][s:e]), fs=fs, nperseg=min(256,e-s))
                axs[seg,1].plot(f,p, label=f"{name} PSD", color=cmap(i))
            f_l, p_l = welch(z_normalize(l_signal[s:e]), fs=fs, nperseg=min(256,e-s))
            axs[seg,1].plot(f_l,p_l,'--', label="Label PSD", color='black')
            axs[seg,1].set_title(f"Subject {subject_id} - Segment {seg+1} PSD")
            axs[seg,1].legend()

        plt.tight_layout()
        plt.show()


def compute_psnr(ref_signal: np.ndarray, pred_signal: np.ndarray) -> float:
    """
    PSNR (Peak Signal-to-Noise Ratio) 계산

    Args:
      ref_signal: 레퍼런스 신호 (label)
      pred_signal: 예측 신호

    Returns:
      PSNR (dB)
    """
    mse = np.mean((ref_signal - pred_signal) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(ref_signal)
    return 10 * np.log10((max_val ** 2) / mse)


def compute_hrv_metrics(signal: np.ndarray, fs: int) -> Dict[str, float]:
    """
    RR 간격으로부터 AVNN과 SDNN을 계산.

    Args:
      signal (np.ndarray): 원본 R-R 시계열 신호.
      fs (int): 샘플링 주파수(Hz).

    Returns:
      Dict[str, float]: {'AVNN(ms)': value, 'SDNN(ms)': value}
    """
    peaks, _ = find_peaks(signal)
    if len(peaks) < 2:
        return {'AVNN(ms)': np.nan, 'SDNN(ms)': np.nan}
    # RR intervals in ms
    rr_intervals = np.diff(peaks) / fs * 1000.0
    avnn = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    return {'AVNN(ms)': avnn, 'SDNN(ms)': sdnn}


def compute_metrics(
    l_signal: np.ndarray,
    p_signal: np.ndarray,
    fs: int
) -> Dict[str, float]:
    # Basic error metrics
    err = p_signal - l_signal
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mask = l_signal != 0
    mape = np.mean(np.abs(err[mask] / l_signal[mask])) * 100 if np.any(mask) else np.nan

    valid = ~np.isnan(l_signal) & ~np.isnan(p_signal)
    pearson = np.corrcoef(p_signal[valid], l_signal[valid])[0,1] if valid.sum() > 1 else np.nan

    # HRV metrics for reference and prediction
    hrv_ref = compute_hrv_metrics(l_signal, fs)
    hrv_pred = compute_hrv_metrics(p_signal, fs)

    # Errors in HRV metrics
    mae_avnn = np.abs(hrv_pred['AVNN(ms)'] - hrv_ref['AVNN(ms)'])
    mae_sdnn = np.abs(hrv_pred['SDNN(ms)'] - hrv_ref['SDNN(ms)'])

    # SNR and PSNR
    var_s = np.var(l_signal)
    var_e = np.var(err)
    snr = 10 * np.log10(var_s / var_e) if var_e > 0 else np.inf
    psnr = compute_psnr(l_signal, p_signal)

    # Combine all metrics
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE(%)": mape,
        "Pearson": pearson,
        # AVNN/SDNN for label
        "Label_AVNN(ms)": hrv_ref['AVNN(ms)'],
        "Label_SDNN(ms)": hrv_ref['SDNN(ms)'],
        # AVNN/SDNN for prediction
        "Pred_AVNN(ms)": hrv_pred['AVNN(ms)'],
        "Pred_SDNN(ms)": hrv_pred['SDNN(ms)'],
        # HRV errors
        "MAE_AVNN(ms)": mae_avnn,
        "MAE_SDNN(ms)": mae_sdnn,
        "SNR(dB)": snr,
        "PSNR(dB)": psnr
    }
    return metrics