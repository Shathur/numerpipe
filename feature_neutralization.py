import numpy as np
import pandas as pd
import scipy.stats
import torch
import joblib
from tqdm.notebook import tqdm

TARGET_NAME = f"target"
PREDICTION_NAME = f"prediction"
PREDICTION_NAME_NEUTRALIZED = f"prediction_neutralized"


def neutralize_short(df, prediction_name=None, by=None, proportion=1.0):
    if by is None:
        by = [x for x in df.columns if x.startswith("feature")]

    scores = df[prediction_name].copy()
    # if you dont take .copy() it changes the original df
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack(
        (exposures, np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1))
    )

    scores = scores - proportion * (
        exposures @ (np.linalg.pinv(exposures) @ scores.values)
    )
    return scores / scores.std()


# to neutralize a column in a df by many other columns on a per-era basis
def neutralize(
    df, columns, extra_neutralizers=None, proportion=1.0, normalize=True, era_col="era"
):
    # need to do this for lint to be happy bc [] is a "dangerous argument"
    if extra_neutralizers is None:
        extra_neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        print(u, end="\r")
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (pd.Series(x).rank(method="first").values - 0.5) / len(x)
                scores2.append(x)
            scores = np.array(scores2).T
            extra = df_era[extra_neutralizers].values
            exposures = np.concatenate([extra], axis=1)
        else:
            exposures = df_era[extra_neutralizers].values

        scores = scores - proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32))
        )

        scores = scores / scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed), columns=columns, index=df.index)


# to neutralize any series by any other series
def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1))
    )

    correction = proportion * (
        exposures.dot(np.linalg.lstsq(exposures, scores, rcond=None)[0])
    )
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


# original code written by JRB at JTaylor's github
# https://github.com/jonrtaylor/twitch/blob/master/FE_Clipping_Script.ipynb
# here we gather the functions and with minor changes implement them into our pipeline
# NOTE: ported from TensorFlow to PyTorch (the previous TF version is kept in
# feature_neutralization_prev6.py). The linear model is a zero-initialized
# `nn.Linear(bias=False)` (i.e. `y = X @ w`, identical to
# tf.keras.experimental.LinearModel(use_bias=False)) and `Adamax` below is a
# faithful re-implementation of the keras.optimizers.Adamax update rule
# (m = b1*m + (1-b1)*g; u = max(b2*u, |g|); w -= lr/(1-b1**t) * m/(u+eps)
# with keras defaults lr=0.001, b1=0.9, b2=0.999, eps=1e-7), so the trimming
# results match those of the previous TF implementation.
def exposures(x, y):
    x = x - torch.mean(x, dim=0, keepdim=True)
    x = x / torch.norm(x, dim=0)
    y = y - torch.mean(y, dim=0, keepdim=True)
    y = y / torch.norm(y, dim=0)
    return torch.matmul(x.transpose(0, 1), y)


class Adamax:
    """Faithful re-implementation of keras.optimizers.Adamax (dense variant).

    torch.optim.Adamax is NOT used because it applies epsilon inside the
    infinity-norm max (u = max(b2*u, |g| + eps)), which yields a different
    trajectory from keras. This class mirrors keras exactly.
    """

    def __init__(self, model, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.var = model.weight
        self.m = torch.zeros_like(self.var)
        self.u = torch.zeros_like(self.var)
        self.t = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def apply_gradients(self, grad):
        self.t += 1
        m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        u = torch.maximum(self.beta_2 * self.u, grad.abs())
        beta_1_power = np.power(np.float32(self.beta_1), np.float32(self.t))
        update = (self.lr * m) / ((1 - beta_1_power) * (u + self.epsilon))
        with torch.no_grad():
            self.var -= update
        self.m, self.u = m, u


def train_loop_body(model, feats, pred, target_exps):
    exps = exposures(feats, pred[:, None] - model(feats))
    loss = torch.sum(
        torch.relu(torch.relu(exps) - torch.relu(target_exps))
        + torch.relu(torch.relu(-exps) - torch.relu(-target_exps))
    )
    return loss


def train_loop(model, optimizer, feats, pred, target_exps, era, max_iters=100000, losses=None):
    # optionally record the loss trajectory (one float per iteration, pre-update)
    for i in range(max_iters):
        loss = train_loop_body(model, feats, pred, target_exps)
        model.zero_grad(set_to_none=False)
        loss.backward()
        optimizer.apply_gradients(model.weight.grad)
        loss = loss.item()
        if losses is not None:
            losses.append(loss)
        if loss < 1e-7:
            print(f"broke at era: {int(era)} with loss: {loss:0.7f} at iteration: {i}")
            break
        if i % 10000 == 0:
            tqdm.write(f"era: {int(era)} loss: {loss:0.7f}", end="\r")


def get_weights(model):
    # (n_features, 1) kernel like a keras Dense kernel, so joblib lm caches
    # written by the previous TF implementation stay loadable
    return [model.weight.detach().numpy().T]


def set_weights(model, weights):
    with torch.no_grad():
        model.weight.copy_(
            torch.as_tensor(np.asarray(weights[0].T), dtype=torch.float32)
        )


def reduce_exposure(prediction, features, max_exp, era, weights=None, max_iters=100000, losses=None):
    model = torch.nn.Linear(features.shape[1], 1, bias=False)
    # kernel_initializer='zeros' like tf.keras.experimental.LinearModel
    torch.nn.init.zeros_(model.weight)
    feats = torch.as_tensor(features - 0.5, dtype=torch.float32)
    pred = torch.as_tensor(prediction, dtype=torch.float32)
    if weights is None:
        optimizer = Adamax(model)
        start_exp = exposures(feats, pred[:, None])
        target_exps = torch.clamp(start_exp, -max_exp, max_exp)
        train_loop(model, optimizer, feats, pred, target_exps, era, max_iters=max_iters, losses=losses)
    else:
        set_weights(model, weights)
    return pred[:, None] - model(feats), get_weights(model)


def reduce_all_exposures(
    df,
    column=["prediction"],
    neutralizers=None,
    lm_cache_file=None,
    normalize=True,
    gaussianize=True,
    era_col="era",
    max_exp=0.1,
    max_iters=100000,
    loss_history=None,
):  ###<-----SELECT YOUR MAXIMUM FEATURE EXPOSURE HERE###
    if neutralizers is None:
        neutralizers = [x for x in df.columns if x.startswith("feature")]
    neutralized = []
    if loss_history is None:
        loss_history = {}
    if lm_cache_file.is_file():
        cache = joblib.load(lm_cache_file)
        # Remove weights for eraX if we'd accidentally saved it in the past.
        cache.pop("eraX", None)
    else:
        cache = {}
    for era in tqdm(df[era_col].unique()):
        tqdm.write(str(era), end="\r")
        df_era = df[df[era_col] == era]
        scores = df_era[column].values
        exposure_values = df_era[neutralizers].values

        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method="ordinal") - 0.5) / len(x)
                if gaussianize:
                    x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2)[0]

        era_losses = loss_history.setdefault(era, [])
        scores, weights = reduce_exposure(
            scores, exposure_values, max_exp, era, cache.get(era),
            max_iters=max_iters, losses=era_losses,
        )
        if era not in cache and era != "eraX":
            cache[era] = weights
            joblib.dump(cache, lm_cache_file)
        scores = scores.detach()
        scores = scores / torch.std(scores, correction=0)
        scores = scores - torch.min(scores)
        scores = scores / torch.max(scores)
        neutralized.append(scores.numpy())

    predictions = pd.DataFrame(
        np.concatenate(neutralized), columns=column, index=df.index
    )
    return predictions


# Put together the code above into a pipeline
def neutralization_trimming(tour_df=None, preds=None, lm_cache_file=None, max_exp=None):
    tour_df[PREDICTION_NAME] = preds

    # save predictions to transfer model either for inference or feature exposure clipping
    predictions_df = tour_df.index.to_frame().reset_index(drop=True)
    predictions_df["prediction_kazutsugi"] = tour_df[PREDICTION_NAME].reset_index(
        drop=True
    )
    predictions_df.to_csv("predictions_saved.csv", index=False)

    # Predictions we want to trim
    predictions_df = pd.read_csv("predictions_saved.csv", index_col=0)

    # we merge the tournament data with our predictions so we have already dropped the indexes
    full_df = pd.merge(tour_df, predictions_df, left_index=True, right_index=True)

    full_df.rename(columns={"prediction_kazutsugi": "prediction"}, inplace=True)

    neutralized_predictions_df = reduce_all_exposures(
        df=full_df, lm_cache_file=lm_cache_file, max_exp=0.10
    )

    tour_df[PREDICTION_NAME_NEUTRALIZED] = neutralized_predictions_df["prediction"]

    return neutralized_predictions_df[PREDICTION_NAME].values
