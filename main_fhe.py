import os
import time

import numpy as np
from sklearn.metrics import confusion_matrix
from concrete.ml.sklearn import XGBClassifier, LogisticRegression

from config import *
from preprocessing import get_data, get_data_v2

from concrete.fhe import Configuration

from tqdm import tqdm


MODE = "execute"
P_ERROR = 1e-2

debug_config = Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location="~/.cml_keycache",
)


def slide_window(B, S, y=None):
    """
    inputs
        - B (np.array: base probabilities of shape (N,W,A)
        - S (int): smoother window size

    """
    N, W, A = B.shape

    # pad it.
    pad = (S + 1) // 2
    pad_left = np.flip(B[:, 0:pad, :], axis=1)
    pad_right = np.flip(B[:, -pad:, :], axis=1)
    B_padded = np.concatenate([pad_left, B, pad_right], axis=1)

    # window it.
    X_slide = np.zeros((N, W, A * S), dtype="float32")
    for ppl, dat in enumerate(B_padded):
        for w in range(W):
            X_slide[ppl, w, :] = dat[w:w + S].ravel()

    # reshape
    X_slide = X_slide.reshape(N * W, A * S)
    y_slide = None if y is None else y.reshape(N * W)

    return X_slide, y_slide


if __name__ == '__main__':
    print("Training with FHE")
    print("Getting data...")
    data, meta = get_data_v2()
    (X_t1, y_t1), (X_t2, y_t2), (X_v, y_v) = data
    print(X_t1.shape, y_t1.shape, X_t2.shape, y_t2.shape, X_v.shape, y_v.shape)

    print("Constructing model...")
    print(meta)
    n_windows = meta["C"] // meta["M"]
    context = int(meta["M"] * CONTEXT_RATIO)
    assert n_windows >= 2 * SMOOTH_SIZE, "Smoother size to large for given window size. "

    base_models = [LogisticRegression(n_bits=8, penalty="l2", C=3., solver="liblinear", max_iter=1000) for _ in range(n_windows)]
    smoother = XGBClassifier(
        n_bits=4, n_estimators=100, max_depth=4,
        learning_rate=0.1, reg_lambda=1, reg_alpha=0,
        n_jobs=N_JOBS, random_state=SEED,
        use_label_encoder=False, objective='multi:softprob',
    )
    print("Concrete does not support `n_jobs` in `XGBClassifier` yet, the training might be slow")

    def train_base_models(models, X_t, y_t):
        if context != 0.:
            pad_left = np.flip(X_t[:, 0:context], axis=1)
            pad_right = np.flip(X_t[:, -context:], axis=1)
            X_t = np.concatenate([pad_left, X_t, pad_right], axis=1)

        M_ = meta["M"] + 2 * context
        idx = np.arange(0, meta["C"], meta["M"])[:-2]
        X_b = np.lib.stride_tricks.sliding_window_view(X_t, M_, axis=1)[:, idx, :]

        train_args = tuple(zip(models[:-1], np.swapaxes(X_b, 0, 1), np.swapaxes(y_t, 0, 1)[:-1]))
        rem = meta["C"] - meta["M"] * n_windows
        train_args += ((models[-1], X_t[:, X_t.shape[1] - (M_ + rem):], y_t[:, -1]),)

        log_iter = tqdm(train_args, total=n_windows, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}', position=0,
                        leave=True)
        return [b[0].fit(b[1], b[2]) for b in log_iter]


    def predict_base_models(models, X_p, fhe="disable"):
        if context != 0.:
            pad_left = np.flip(X_p[:, 0:context], axis=1)
            pad_right = np.flip(X_p[:, -context:], axis=1)
            X_p = np.concatenate([pad_left, X_p, pad_right], axis=1)

        M_ = meta["M"] + 2 * context
        idx = np.arange(0, meta["C"], meta["M"])[:-2]
        X_b = np.lib.stride_tricks.sliding_window_view(X_p, M_, axis=1)[:, idx, :]

        base_args = tuple(zip(models[:-1], np.swapaxes(X_b, 0, 1)))
        rem = meta["C"] - meta["M"] * n_windows
        base_args += ((models[-1], X_p[:, X_p.shape[1] - (M_ + rem):]),)

        log_iter = tqdm(base_args, total=n_windows, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}', position=0, leave=True)

        B = np.array([b[0].predict_proba(b[1], fhe=fhe) for b in log_iter])
        B = np.swapaxes(B, 0, 1)

        return B

    def compile_base_models(models, X_t):
        if context != 0.:
            pad_left = np.flip(X_t[:, 0:context], axis=1)
            pad_right = np.flip(X_t[:, -context:], axis=1)
            X_t = np.concatenate([pad_left, X_t, pad_right], axis=1)

        M_ = meta["M"] + 2 * context
        idx = np.arange(0, meta["C"], meta["M"])[:-2]
        X_b = np.lib.stride_tricks.sliding_window_view(X_t, M_, axis=1)[:, idx, :]

        train_args = tuple(zip(models[:-1], np.swapaxes(X_b, 0, 1)))
        rem = meta["C"] - meta["M"] * n_windows
        train_args += ((models[-1], X_t[:, X_t.shape[1] - (M_ + rem):]),)

        log_iter = tqdm(train_args, total=n_windows, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}', position=0,
                        leave=True)
        return [b[0].compile(b[1], configuration=debug_config, p_error=P_ERROR) for b in log_iter]

    def train_smoother(smoother, X_t, y_t):
        X_slide, y_slide = slide_window(X_t, SMOOTH_SIZE, y_t)
        return smoother.fit(X_slide, y_slide)


    def predict_smoother(smoother, X_p, fhe="disable"):
        X_slide, _ = slide_window(X_p, SMOOTH_SIZE)
        prob = smoother.predict_proba(X_slide, fhe=fhe)

        return prob.reshape(-1, n_windows, meta["A"])

    def compile_smoother(smoother, X_t):
        X_slide, _ = slide_window(X_t, SMOOTH_SIZE)
        return smoother.compile(X_slide, configuration=debug_config, p_error=P_ERROR)

    print("Stage 1: training base models...")
    t_begin = time.time()
    base_models = train_base_models(base_models, X_t1, y_t1)
    prob_X_t2 = predict_base_models(base_models, X_t2)
    print(f"Stage 1: {time.time() - t_begin:.2f} seconds")

    print("Stage 2: training smoother...")
    t_begin = time.time()
    smoother = train_smoother(smoother, prob_X_t2, y_t2)
    print(f"Stage 2: {time.time() - t_begin:.2f} seconds")

    print("Wrap-up: compiling...")
    t_begin = time.time()
    compile_base_models(base_models, X_t1)
    compile_smoother(smoother, prob_X_t2)
    print(f"Compiling: {time.time() - t_begin:.2f} seconds")

    print("Evaluating model (simulate)...")
    t_begin = time.time()
    prob_X_v = predict_base_models(base_models, X_v, fhe="simulate")
    X_slide, _ = slide_window(prob_X_v, SMOOTH_SIZE)
    pbar = tqdm(range(len(X_slide)), total=len(X_slide))
    total, correct = 0, 0
    y_v_predict_all = []
    y_v_flatten = y_v.reshape(-1)
    for i in pbar:
        y_v_pred = np.argmax(smoother.predict_proba(X_slide[i].reshape(1, -1), fhe="simulate"), axis=-1)[0]
        y_v_predict_all.append(y_v_pred)
        correct += np.sum(y_v_flatten[i] == y_v_pred)
        total += 1
        pbar.set_description(f"accuracy={correct / total:.4f}")
    print(f"Accuracy on validation set: {correct / total:.4f}")
    conf_matrix_v = confusion_matrix(y_v_flatten, np.array(y_v_predict_all))
    indices_v = sorted(np.unique(np.concatenate((y_v_flatten, np.array(y_v_predict_all)))))
    print("Confusion matrix on validation set:")
    print(conf_matrix_v)
    print(f"Evaluating (simulate): {time.time() - t_begin:.2f} seconds")

    print("Evaluating model (execute)...")
    t_begin = time.time()
    X_execute = X_v[0].reshape(1, -1)
    prob_X_execute_v = predict_base_models(base_models, X_execute, fhe="execute")
    X_execute_slide, _ = slide_window(prob_X_execute_v, SMOOTH_SIZE)
    pbar = tqdm(range(len(X_execute_slide)), total=len(X_execute_slide))
    total, correct = 0, 0
    y_v_execute_predict_all = []
    y_v_execute_flatten = y_v[0].reshape(-1)
    for i in pbar:
        y_v_execute_pred = np.argmax(smoother.predict_proba(X_execute_slide[i].reshape(1, -1), fhe="execute"), axis=-1)[0]
        y_v_execute_predict_all.append(y_v_execute_pred)
        correct += np.sum(y_v_execute_flatten[i] == y_v_execute_pred)
        total += 1
        pbar.set_description(f"accuracy={correct / total:.4f}")
    print(f"Accuracy on validation set: {correct / total:.4f}")
    print(f"Evaluating (execute): {time.time() - t_begin:.2f} seconds")
