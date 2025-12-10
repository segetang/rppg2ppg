import os
import numpy as np
from signal_utils import (
    split_by_subjects,
    plot_subject_signals_chunks,
    plot_multi_subject_signals_chunks
)
from signal_utils import compute_metrics

# 추가 라이브러리
from scipy.signal import find_peaks
from scipy.stats import spearmanr, entropy

# ----------------------------------
# 설정 변수 (실험용)
# ----------------------------------
result_save_path = "/code/rppg2ppg/newdata"
test_dataset     = "PURE"    # ["UBFC", "PURE", "cohface"]
multiple_preds   = True      # False 면 기존 단일 모델 모드
model_names      = ["POS"]  # 비교할 모델 리스트
eval_seconds     = 10        # 평가 청크 길이 (초). -1 이면 전체 신호 평가.
fs               = 30        # 샘플링 주파수
apply_detrend    = True      # True이면 detrend 적용
apply_BPF        = True      # True이면 BPF 적용
Lambda_value     = 100       # detrend 파라미터

# ----------------------------------
# 파일 경로 설정 & 로드
# ----------------------------------
label_fp = os.path.join(result_save_path, f"{test_dataset}_label.npy")
if not os.path.exists(label_fp):
    raise FileNotFoundError(f"레이블 파일이 없습니다: {label_fp}")
labels = np.load(label_fp)

if multiple_preds:
    # 여러 모델 예측을 dict 형태로 로드
    preds_dict = {}
    for m in model_names:
        p_fp = os.path.join(result_save_path, f"{m}.npy")
        if not os.path.exists(p_fp):
            raise FileNotFoundError(f"예측 파일이 없습니다: {p_fp}")
        preds_dict[m] = np.load(p_fp)
else:
    # 단일 모델 예측 (원래 방식)
    model_name = model_names[0]
    pred_fp = os.path.join(result_save_path, f"{test_dataset}_{model_name}_prediction.npy")
    if not os.path.exists(pred_fp):
        raise FileNotFoundError(f"예측 파일이 없습니다: {pred_fp}")
    preds = np.load(pred_fp)

print("\n=== Overall Metrics per Model (기존 compute_metrics 호출) ===")
if multiple_preds:
    iter_models = preds_dict.items()
else:
    iter_models = {model_name: preds}.items()
for model_name, p in iter_models:
    metrics = compute_metrics(
        l_signal = labels,
        p_signal = p,
        fs       = fs
    )
    # 예쁘게 고정소수점 표기 출력
    print(f"\n>> {model_name}")
    for k, v in metrics.items():
        try:
            print(f"   {k:10s}: {v:.4f}")
        except Exception:
            print(f"   {k:10s}: {v}")

# ----------------------------------
# subject별 segment 개수 설정
# ----------------------------------
if test_dataset.upper() == "UBFC":
    subject_counts = [6,6,6,6,6,6,6,6,5,6,6,6,6,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                      6,6,6,6,4,6,6]
elif test_dataset.upper() == "PURE":
    subject_counts = [6,6,8,8,6,6,6,6,7,7,6,6,6,6,7,7,6,6,6,6,
                      7,7,6,6,6,6,7,7,6,6,6,7,7,6,6,6,7,7,7,6,6,
                      6,6,7,7,6,6,6,6,7,8,6,6,6,6,7,7,6,6]
elif test_dataset.lower() == "cohface":
    subject_counts = [4,4,4,4,4,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                      4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                      4,4,4,4,4,4,4,4,4,4,4,4,4,3,4,4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,1,4,4,4,4,4,
                      4,4,4,4,4,4,4,4,4,4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,4,4,3,4,4,4,
                      4,4,4,4,4,4,4,4,4,4,4,4]
else:
    raise ValueError("알 수 없는 데이터셋입니다.")

# ----------------------------------
# subject별 분할
# ----------------------------------
if multiple_preds:
    # 첫 모델로부터 split_labels만 추출
    first = model_names[0]
    _, split_labels = split_by_subjects(preds_dict[first], labels, subject_counts, segment_length=fs*eval_seconds)
    # 모델별로 split_preds 저장
    split_preds_dict = {
        m: split_by_subjects(preds_dict[m], labels, subject_counts, segment_length=fs*eval_seconds)[0]
        for m in model_names
    }
else:
    split_preds, split_labels = split_by_subjects(preds, labels, subject_counts, segment_length=fs*eval_seconds)

# ----------------------------------
# --- 새 지표 계산을 위한 함수들 (BWMD, SRE, WCR, EDD) + HRV 추가
# ----------------------------------

def dtw_distance(a, b):
    na, nb = len(a), len(b)
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = abs(a[i-1] - b[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    dist = D[na, nb]
    norm = dist / max(na + nb, 1)
    return float(norm)


def find_peaks_times(sig, fs):
    min_dist = max(1, int(0.4 * fs))
    peaks, _ = find_peaks(sig, distance=min_dist)
    times = peaks / float(fs)
    return np.array(times)


def safe_median_rr(peak_times):
    if len(peak_times) < 2:
        return None
    rr = np.diff(peak_times)
    return float(np.median(rr))


def compute_bwmd_for_chunk(y, yhat, fs, lambda_p=1.0, lambda_a=0.5):
    """
    한 chunk에 대한 BWMD만 계산.
    """
    max_y = np.max(np.abs(y)) + 1e-8
    max_yhat = np.max(np.abs(yhat)) + 1e-8
    a = y / max_y
    b = yhat / max_yhat

    dtw_norm = dtw_distance(a, b)

    pt_y = find_peaks_times(y, fs)
    pt_yhat = find_peaks_times(yhat, fs)
    if len(pt_y) >= 2 and len(pt_yhat) >= 2:
        k = min(len(pt_y), len(pt_yhat))
        diffs = np.abs(pt_y[:k] - pt_yhat[:k])
        peak_time_diff_mean = float(np.mean(diffs))
        rr = safe_median_rr(pt_y)
        rr_norm = rr if (rr is not None and rr > 1e-6) else (len(y) / float(fs))
        peak_time_diff_norm = peak_time_diff_mean / rr_norm
    else:
        if len(pt_y) >= 1 and len(pt_yhat) >= 1:
            diff = abs(pt_y[0] - pt_yhat[0])
            peak_time_diff_norm = diff / (len(y) / float(fs))
        else:
            peak_time_diff_norm = 0.5

    A_y = float(np.trapz(np.abs(y)))
    A_yhat = float(np.trapz(np.abs(yhat)))
    area_rel = abs(A_y - A_yhat) / (A_y + 1e-8)

    bwmd = (dtw_norm + lambda_p * peak_time_diff_norm) / (1.0 + lambda_a * area_rel)
    return float(bwmd)


def concordance_ccc(x, y):
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    if x.size == 0 or y.size == 0:
        return 0.0
    mx = np.mean(x); my = np.mean(y)
    sx2 = np.var(x, ddof=1) if x.size > 1 else 0.0
    sy2 = np.var(y, ddof=1) if y.size > 1 else 0.0
    cov = np.cov(x, y, ddof=1)[0, 1] if x.size > 1 and y.size > 1 else 0.0
    denom = sx2 + sy2 + (mx - my) ** 2
    if denom == 0:
        return 0.0
    ccc = 2.0 * cov / denom
    return float(np.clip(ccc, -1.0, 1.0))


def compute_wcr(y_concat, yhat_concat, beta=0.6):
    x = np.ravel(y_concat)
    y = np.ravel(yhat_concat)
    ccc = concordance_ccc(x, y)
    try:
        rho_s, _ = spearmanr(x, y)
        if np.isnan(rho_s):
            rho_s = 0.0
    except Exception:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        rho_s = np.corrcoef(rx, ry)[0, 1] if len(rx) > 1 else 0.0
    wcr = beta * ccc + (1.0 - beta) * rho_s
    return float(np.clip(wcr, -1.0, 1.0))


def jsd_from_histograms(p, q):
    p = p / (np.sum(p) + 1e-12)
    q = q / (np.sum(q) + 1e-12)
    m = 0.5 * (p + q)
    kl_pm = entropy(p, m)
    kl_qm = entropy(q, m)
    jsd = 0.5 * (kl_pm + kl_qm)
    jsd_norm = float(jsd / np.log(2.0)) if np.log(2.0) > 0 else float(jsd)
    return float(np.clip(jsd_norm, 0.0, 1.0))


def compute_edd(y_concat, yhat_concat, bins=60):
    e = np.ravel(yhat_concat) - np.ravel(y_concat)
    if e.size == 0:
        return 0.0
    hist, edges = np.histogram(e, bins=bins, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    P = hist.astype(float) + 1e-12
    sigma = np.std(e, ddof=1) if e.size > 1 else 1.0
    if sigma < 1e-8:
        sigma = 1e-8
    Q_pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * (centers ** 2) / (sigma ** 2))
    Q = Q_pdf + 1e-12
    P_norm = P / np.sum(P)
    Q_norm = Q / np.sum(Q)
    jsd = jsd_from_histograms(P_norm, Q_norm)
    return float(jsd)

# --- SRE 관련 함수들 ---

def compute_rmse(y, yhat):
    e = np.ravel(yhat) - np.ravel(y)
    if e.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(e**2)))


def compute_snr_db(y, yhat, eps=1e-12):
    # signal variance vs noise variance, dB로 반환
    y = np.ravel(y).astype(float)
    e = np.ravel(yhat).astype(float) - y
    if y.size == 0:
        return -100.0
    signal_var = float(np.var(y, ddof=1)) if y.size > 1 else float(np.var(y))
    noise_var = float(np.var(e, ddof=1)) if e.size > 1 else float(np.var(e))
    if noise_var < eps:
        return 100.0  # 매우 높은 SNR
    if signal_var < eps:
        return -100.0  # 거의 신호가 없음
    snr_linear = signal_var / noise_var
    snr_db = 10.0 * np.log10(snr_linear)
    return float(snr_db)


def sigmoid(x):
    # 안정적 sigmoid
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def compute_sre(y, yhat, alpha_c=1.0, alpha_s=1.0, w_c=2.0, w_s=0.5):
    """
    SRE = RMSE * (1 + w_c*CRI + w_s*SI)
    CRI = sigmoid(alpha_c * (3 - SNR_dB))
    SI  = sigmoid(alpha_s * (SNR_dB - 15))
    """
    rmse = compute_rmse(y, yhat)
    snr_db = compute_snr_db(y, yhat)
    cri = sigmoid(alpha_c * (3.0 - snr_db))
    si = sigmoid(alpha_s * (snr_db - 15.0))
    sre = rmse * (1.0 + w_c * cri + w_s * si)
    return float(sre), float(rmse), float(snr_db), float(cri), float(si)

# 정규화 유틸 (퍼센타일 클램프 후 0..1)

def percentile_normalize(val, ref_vals, p_lo=5, p_hi=95, lower_is_better=True, eps=1e-8):
    ref = np.asarray(ref_vals)
    if ref.size < 3:
        if lower_is_better:
            lo, hi = 0.0, max(1.0, float(val))
        else:
            lo, hi = 0.0, max(1.0, float(val))
    else:
        lo = np.percentile(ref, p_lo)
        hi = np.percentile(ref, p_hi)
        if np.isclose(hi, lo):
            hi = lo + eps
    v = float(val)
    v_clamped = np.clip(v, lo, hi)
    norm = (v_clamped - lo) / (hi - lo + eps)
    return float(np.clip(norm, 0.0, 1.0))


def composite_v2_from_metrics(m_bwmd, m_sre, m_wcr, m_edd, ref_bwmd, ref_sre, ref_wcr, ref_edd,
                              weights=None):
    """
    Composite on BWMD, SRE, WCR, EDD
    weights default sum to 1.
    """
    if weights is None:
        weights = {'BWMD':0.3, 'SRE':0.2, 'WCR':0.3, 'EDD':0.2}

    n_bwmd = percentile_normalize(m_bwmd, ref_bwmd, lower_is_better=True)
    n_sre  = percentile_normalize(m_sre,  ref_sre,  lower_is_better=True)
    n_edd  = percentile_normalize(m_edd,  ref_edd,  lower_is_better=True)
    n_wcr  = percentile_normalize(m_wcr,  ref_wcr,  lower_is_better=False)

    s_bwmd = 1.0 - n_bwmd
    s_sre  = 1.0 - n_sre
    s_edd  = 1.0 - n_edd
    s_wcr  = n_wcr

    wsum = sum(weights.values()) + 1e-12
    agg = (weights['BWMD']*s_bwmd + weights['SRE']*s_sre +
           weights['WCR']*s_wcr + weights['EDD']*s_edd) / wsum
    composite = float(np.clip(agg, 0.0, 1.0) * 100.0)
    breakdown = {'s_bwmd': s_bwmd, 's_sre': s_sre, 's_wcr': s_wcr, 's_edd': s_edd, 'agg': agg}
    return composite, breakdown

# --- HRV 관련 보조 함수들 ---

def compute_rr_intervals_from_peak_times(peak_times):
    """peak_times: array of peak times in seconds. returns rr intervals in seconds."""
    if peak_times is None or len(peak_times) < 2:
        return np.array([])
    rr = np.diff(peak_times)
    return np.asarray(rr, dtype=float)


def compute_hrv_metrics_from_peak_times(peak_times):
    """Return dict with HRV measures computed from peak times (seconds).
    Measures: mean_rr, median_rr, sdnn, rmssd, pnn50, mean_hr_bpm
    """
    rr = compute_rr_intervals_from_peak_times(peak_times)
    if rr.size == 0:
        return {
            'mean_rr': np.nan,
            'median_rr': np.nan,
            'sdnn': np.nan,
            'rmssd': np.nan,
            'pnn50': np.nan,
            'mean_hr': np.nan
        }
    mean_rr = float(np.mean(rr))
    median_rr = float(np.median(rr))
    sdnn = float(np.std(rr, ddof=1)) if rr.size > 1 else 0.0
    diff_rr = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff_rr**2))) if diff_rr.size > 0 else 0.0
    nn50 = np.sum(np.abs(diff_rr) > 0.05) if diff_rr.size > 0 else 0
    pnn50 = float(nn50) / float(diff_rr.size) if diff_rr.size > 0 else 0.0
    mean_hr = 60.0 / mean_rr if mean_rr > 0 else np.nan
    return {
        'mean_rr': mean_rr,
        'median_rr': median_rr,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'pnn50': pnn50,
        'mean_hr': mean_hr
    }

# ---------- 모델별/주체별 지표 계산 루프 ----------
print("\n=== New rPPG Metrics (BWMD, SRE, WCR, EDD) & Composite v2 & HRV ===")

all_bwmds = []
all_sres = []
all_wcrs = []
all_edds = []

metrics_store = {m: [] for m in (model_names if multiple_preds else [model_name])}

models_to_process = (model_names if multiple_preds else [model_name])
for m in models_to_process:
    preds_per_subject = split_preds_dict[m] if multiple_preds else split_preds
    for subj_idx, chunks_pred in enumerate(preds_per_subject):
        chunks_label = split_labels[subj_idx]
        n_chunks = len(chunks_label)
        bwmd_list = []
        y_concat = []
        yhat_concat = []
        for c in range(n_chunks):
            raw_y = chunks_label[c]
            raw_yhat = chunks_pred[c]

            # None 또는 비정상 값이면 건너뜀
            if raw_y is None or raw_yhat is None:
                continue

            # 항상 1D numpy array로 보장 (스칼라 -> 길이1 array)
            y_chunk = np.atleast_1d(np.asarray(raw_y, dtype=float))
            yhat_chunk = np.atleast_1d(np.asarray(raw_yhat, dtype=float))

            # 길이 확보 (size 사용)
            L = min(y_chunk.size, yhat_chunk.size)
            if L == 0:
                continue

            # 같은 길이로 자름
            y_chunk = y_chunk[:L]
            yhat_chunk = yhat_chunk[:L]

            # BWMD 계산 (예외 처리 포함)
            try:
                b = compute_bwmd_for_chunk(y_chunk, yhat_chunk, fs)
            except Exception as e:
                # 경고만 남기고 해당 청크 건너뜀
                # print(f"Warning: compute_bwmd_for_chunk error for subject {subj_idx} chunk {c}: {e}")
                continue

            bwmd_list.append(b)
            y_concat.append(y_chunk)
            yhat_concat.append(yhat_chunk)

        subj_bwmd = float(np.mean(bwmd_list)) if len(bwmd_list) > 0 else 0.0
        if len(y_concat) > 0:
            y_concat = np.concatenate(y_concat)
            yhat_concat = np.concatenate(yhat_concat)
        else:
            y_concat = np.array([])
            yhat_concat = np.array([])

        # WCR, EDD 계산
        subj_wcr = compute_wcr(y_concat, yhat_concat)
        subj_edd = compute_edd(y_concat, yhat_concat)

        # SRE 계산 (주체 전체 concat 기반)
        subj_sre, subj_rmse, subj_snr_db, subj_cri, subj_si = compute_sre(y_concat, yhat_concat)

        # --- HRV 계산 (label vs pred) ---
        # peak 탐지는 concat 신호에서 수행
        pt_y = find_peaks_times(y_concat, fs) if y_concat.size > 0 else np.array([])
        pt_yhat = find_peaks_times(yhat_concat, fs) if yhat_concat.size > 0 else np.array([])

        hrv_label = compute_hrv_metrics_from_peak_times(pt_y)
        hrv_pred  = compute_hrv_metrics_from_peak_times(pt_yhat)

        # HRV 차이 (절대값)
        hrv_diff = {}
        for k_hrv in hrv_label.keys():
            a = hrv_label.get(k_hrv, np.nan)
            b = hrv_pred.get(k_hrv, np.nan)
            try:
                if np.isnan(a) or np.isnan(b):
                    hrv_diff[k_hrv] = np.nan
                else:
                    hrv_diff[k_hrv] = float(abs(a - b))
            except Exception:
                hrv_diff[k_hrv] = np.nan

        metrics_store[m].append({
            'subject_idx': subj_idx,
            'BWMD': subj_bwmd,
            'SRE' : subj_sre,
            'WCR': subj_wcr,
            'EDD': subj_edd,
            'RMSE': subj_rmse,
            'SNR_dB': subj_snr_db,
            'CRI': subj_cri,
            'SI': subj_si,
            'HRV_label': hrv_label,
            'HRV_pred' : hrv_pred,
            'HRV_diff' : hrv_diff
        })
        all_bwmds.append(subj_bwmd)
        all_sres.append(subj_sre)
        all_wcrs.append(subj_wcr)
        all_edds.append(subj_edd)

# ref 분포 준비
ref_bwmd = np.array(all_bwmds) if len(all_bwmds) > 0 else np.array([0.0])
ref_sre  = np.array(all_sres)  if len(all_sres)  > 0 else np.array([0.0])
ref_wcr  = np.array(all_wcrs)  if len(all_wcrs)  > 0 else np.array([0.0])
ref_edd  = np.array(all_edds)  if len(all_edds)  > 0 else np.array([0.0])

# 모델별 평균 및 Composite 계산, 출력 (고정 소수점 표기)
for m in models_to_process:
    subj_metrics = metrics_store[m]
    bwmd_vals = np.array([d['BWMD'] for d in subj_metrics])
    sre_vals  = np.array([d['SRE']  for d in subj_metrics])
    wcr_vals  = np.array([d['WCR']  for d in subj_metrics])
    edd_vals  = np.array([d['EDD']  for d in subj_metrics])

    mean_bwmd = float(np.nanmean(bwmd_vals)) if bwmd_vals.size > 0 else 0.0
    mean_sre  = float(np.nanmean(sre_vals))  if sre_vals.size > 0 else 0.0
    mean_wcr  = float(np.nanmean(wcr_vals))  if wcr_vals.size > 0 else 0.0
    mean_edd  = float(np.nanmean(edd_vals))  if edd_vals.size > 0 else 0.0

    composite, breakdown = composite_v2_from_metrics(
        mean_bwmd, mean_sre, mean_wcr, mean_edd,
        ref_bwmd=ref_bwmd, ref_sre=ref_sre, ref_wcr=ref_wcr, ref_edd=ref_edd
    )

    # HRV 평균/오차 집계
    hrv_keys = ['mean_rr', 'median_rr', 'sdnn', 'rmssd', 'pnn50', 'mean_hr']
    hrv_label_means = {k: np.nanmean([s['HRV_label'][k] for s in subj_metrics if not np.isnan(s['HRV_label'][k])]) if any([not np.isnan(s['HRV_label'][k]) for s in subj_metrics]) else np.nan for k in hrv_keys}
    hrv_pred_means  = {k: np.nanmean([s['HRV_pred'][k]  for s in subj_metrics if not np.isnan(s['HRV_pred'][k])])  if any([not np.isnan(s['HRV_pred'][k])  for s in subj_metrics]) else np.nan for k in hrv_keys}
    hrv_diff_means  = {k: np.nanmean([s['HRV_diff'][k]  for s in subj_metrics if not np.isnan(s['HRV_diff'][k])])  if any([not np.isnan(s['HRV_diff'][k])  for s in subj_metrics]) else np.nan for k in hrv_keys}

    print(f"\n>> {m}  (new)")
    print(f"   mean_BWMD : {mean_bwmd:.5f}  (Smaller is better)")
    print(f"   mean_SRE  : {mean_sre:.5f}  (Smaller is better)")
    print(f"   mean_WCR  : {mean_wcr:.5f}  (Bigger is better)")
    print(f"   mean_EDD  : {mean_edd:.5f}  (Smaller is better)")
    print(f"   Composite_v2 (0-100): {composite:.2f}")
    print("   Breakdown (scores 0..1):")
    for bk_k, bk_v in breakdown.items():
        try:
            print(f"      {bk_k:12s}: {bk_v:.5f}")
        except Exception:
            print(f"      {bk_k:12s}: {bk_v}")

    # HRV 출력
    print("\n   HRV (label avg | pred avg | mean abs diff)")
    for k in hrv_keys:
        lab = hrv_label_means.get(k, np.nan)
        pr  = hrv_pred_means.get(k, np.nan)
        df  = hrv_diff_means.get(k, np.nan)
        lab_s = f"{lab:.4f}" if not np.isnan(lab) else "nan"
        pr_s  = f"{pr:.4f}"  if not np.isnan(pr)  else "nan"
        df_s  = f"{df:.4f}"  if not np.isnan(df)  else "nan"
        print(f"      {k:8s}: {lab_s:>8s} | {pr_s:>8s} | {df_s:>8s}")

# ----------------------------------
# 평가 & 플롯 (subject별) - 기존 코드 재사용
# ----------------------------------
for idx, l_sub in enumerate(split_labels):
    if multiple_preds:
        subj_preds = {m: split_preds_dict[m][idx] for m in model_names}
        plot_multi_subject_signals_chunks(
            subject_id = idx+1,
            preds_dict  = subj_preds,
            l_signal    = l_sub,
            fs          = fs,
            eval_seconds= eval_seconds,
            detrend_on  = apply_detrend,
            bpf_on      = apply_BPF,
            Lambda      = Lambda_value
        )
    else:
        p_sub = split_preds[idx]
        plot_subject_signals_chunks(
            subject_id = idx+1,
            p_signal   = p_sub,
            l_signal   = l_sub,
            fs         = fs,
            eval_seconds = eval_seconds,
            detrend_on = apply_detrend,
            bpf_on     = apply_BPF,
            Lambda     = Lambda_value
        )
