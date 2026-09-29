"""EmbedPlan on your own data, the scikit-learn way.

A transition is (state, action) -> next state, each written as text. Give EmbedPlan transitions
from any domain, it learns to predict the next state in a frozen encoder's embedding space and
returns the closest real state:

    from embedplan import EmbedPlan, split_transitions
    from embedplan.datasets import load_toy_ferry

    data = load_toy_ferry()
    X_train, X_test, y_train, y_test, g_train, g_test = split_transitions(
        data.X, data.y, data.groups, protocol="extrapolation")      # hold out whole problems
    model = EmbedPlan(encoder="hashing").fit(X_train, y_train, groups=g_train)
    model.evaluate(X_test, y_test, groups=g_test)   # Hit@1/5/10 among 128 candidates, and chance
    model.predict(X_test[:3])                        # the most likely next states, as text

Evaluation follows the paper's protocol: each query ranks its true next state among 128
candidates (the truth and 127 distractors, drawn without replacement and never equal to the
truth), distractors come from the query's own group (its problem) when groups are given, and a
candidate that scores exactly as high as the truth ranks above it. Checkpoints are selected on a
validation split carved out of the training data, never on the test data.
"""

import copy
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.utils.validation import check_is_fitted

from embedplan.encoders import get_encoder
from embedplan.models import build_model
from embedplan.training import train_transition

__all__ = ["EmbedPlan", "split_transitions", "transitions_from_trajectories"]

Candidates = Optional[Union[Sequence[str], Sequence[Sequence[str]]]]

# The settings behind the paper's main tables (experiments/train.py grid).
PAPER_PARAMS = dict(projection_dim=128, projection_layers=2, hidden_size=128, n_layers=2, tau=0.07,
                    action_weight=2.0, lr=4e-5, batch_size=128, max_epochs=200, early_stopping=False)


def _check_pairs(X, y=None) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """(states, actions, next_states) from pairs or a DataFrame with state/action(/next_state) columns."""
    if isinstance(X, pd.DataFrame):
        missing = {"state", "action"} - set(X.columns)
        if missing:
            raise ValueError(f"a DataFrame X needs the columns 'state' and 'action' (missing: {sorted(missing)})")
        states, actions = X["state"].tolist(), X["action"].tolist()
        if y is None and "next_state" in X.columns:
            y = X["next_state"].tolist()
    else:
        try:
            pairs = [tuple(p) for p in X]
        except TypeError:
            raise TypeError("X must be a sequence of (state, action) pairs or a DataFrame") from None
        if any(len(p) != 2 for p in pairs):
            raise ValueError("every item of X must be a (state, action) pair")
        states, actions = [p[0] for p in pairs], [p[1] for p in pairs]
    if not states:
        raise ValueError("X is empty")
    nxt = None if y is None else list(y)
    if nxt is not None and len(nxt) != len(states):
        raise ValueError(f"X has {len(states)} transitions but y has {len(nxt)} next states")
    for name, col in (("state", states), ("action", actions), ("next state", nxt or [])):
        for v in col:
            if not isinstance(v, str) or not v.strip():
                raise TypeError(f"every {name} must be a non-empty string, got {v!r}")
    return states, actions, nxt


def transitions_from_trajectories(trajectories, groups=None):
    """Cut trajectories into transitions.

    Each trajectory is (states, actions) with len(states) == len(actions) + 1, all text: the state
    before each action and the state it leads to. `groups` gives one id per trajectory (e.g. its
    problem); by default each trajectory is its own group. Returns X, y, groups for fit().
    """
    X, y, g = [], [], []
    for i, traj in enumerate(trajectories):
        states, actions = traj
        if len(states) != len(actions) + 1:
            raise ValueError(f"trajectory {i}: {len(states)} states for {len(actions)} actions; "
                             "expected one more state than actions")
        gid = i if groups is None else groups[i]
        for t, action in enumerate(actions):
            X.append((states[t], action))
            y.append(states[t + 1])
            g.append(gid)
    return X, y, np.asarray(g)


def split_transitions(X, y, groups=None, protocol: str = "extrapolation", test_size: float = 0.2,
                      random_state: int = 0):
    """Train/test split under one of the paper's protocols.

    "extrapolation"  whole groups (problems) are held out: the test problems are unseen
    "interpolation"  transitions are held out at random: the same problems appear on both sides

    Returns X_train, X_test, y_train, y_test, groups_train, groups_test (groups are None if not given).
    """
    states, actions, nxt = _check_pairs(X, y)
    n = len(states)
    if protocol == "extrapolation":
        if groups is None:
            raise ValueError("protocol='extrapolation' holds out whole groups: pass groups (e.g. problem ids)")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr, te = next(splitter.split(np.zeros(n), groups=np.asarray(groups)))
    elif protocol == "interpolation":
        tr, te = next(ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state).split(np.zeros(n)))
    else:
        raise ValueError(f"unknown protocol {protocol!r}: use 'extrapolation' or 'interpolation'")
    pairs = list(zip(states, actions))
    take = lambda seq, idx: [seq[i] for i in idx]  # noqa: E731
    g = None if groups is None else np.asarray(groups)
    return (take(pairs, tr), take(pairs, te), take(nxt, tr), take(nxt, te),
            None if g is None else g[tr], None if g is None else g[te])


class EmbedPlan(BaseEstimator):
    """A transition model learned on top of a frozen text encoder.

    Parameters
    ----------
    encoder : str or callable, default "hashing"
        "hashing" (word n-grams, no download), a sentence-transformers or Hugging Face model
        name (e.g. "BAAI/bge-m3", "Qwen/Qwen2.5-7B-Instruct"), or your own function mapping a list
        of texts to an array of shape (n_texts, dim), e.g. a lookup into precomputed embeddings.
    projection_dim, projection_layers : size and depth of the learned heads that project the
        state and action embeddings into the space where the transition is computed.
    hidden_size, n_layers, dropout : the residual MLP that predicts the next-state embedding.
    tau : InfoNCE temperature. action_weight : weight of the action-disambiguation term.
    lr, batch_size, max_epochs : AdamW learning rate, batch size, epoch budget.
    early_stopping : off by default (train for max_epochs, as the paper does). When on, hold out
        `validation_fraction` of the training transitions (whole groups when groups are given),
        compute the validation mean reciprocal rank every `eval_every` epochs, stop after
        `patience` epochs without improvement, and keep the best weights.
    loss_space : "absolute" (the paper's objective) or "delta" (scores the displacement).
    device : "auto" (CUDA when available, else CPU), "cpu" or "cuda".
    random_state : seed for initialization, batching, validation split and evaluation distractors.

    `EmbedPlan(**EmbedPlan.paper_params())` uses the settings behind the paper's main tables.
    """

    def __init__(self, encoder: Union[str, Callable] = "hashing", projection_dim: int = 128,
                 projection_layers: int = 2, hidden_size: int = 128, n_layers: int = 2, dropout: float = 0.0,
                 tau: float = 0.07, action_weight: float = 2.0, lr: float = 1e-3, batch_size: int = 128,
                 max_epochs: int = 200, early_stopping: bool = False, validation_fraction: float = 0.1,
                 patience: int = 30, eval_every: int = 5, loss_space: str = "absolute", device: str = "auto",
                 random_state: Optional[int] = 0, verbose: bool = False):
        self.encoder = encoder
        self.projection_dim = projection_dim
        self.projection_layers = projection_layers
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.tau = tau
        self.action_weight = action_weight
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.eval_every = eval_every
        self.loss_space = loss_space
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def paper_params() -> Dict:
        """The hyperparameters behind the paper's main tables (lr 4e-5, 200 epochs, no early stopping)."""
        return dict(PAPER_PARAMS)

    # ---- text -> rows of the embedding tables -------------------------------------------------

    def _rows(self, texts: Sequence[str], kind: str) -> torch.Tensor:
        """Row of each text in the state or action table, encoding (once) the texts not seen yet."""
        ids, table = (self._state_ids, "_S") if kind == "state" else (self._action_ids, "_A")
        new = list(dict.fromkeys(t for t in texts if t not in ids))
        if new:
            emb = torch.from_numpy(self.encoder_(new))
            if emb.shape[1] != getattr(self, table).shape[1]:
                raise ValueError(f"the encoder returned dim {emb.shape[1]}, the model was fit with "
                                 f"{getattr(self, table).shape[1]}")
            start = getattr(self, table).shape[0]
            for i, t in enumerate(new):
                ids[t] = start + i
            setattr(self, table, torch.cat([getattr(self, table), emb.float()]))
            if kind == "state":
                self._is_candidate = torch.cat([self._is_candidate, torch.zeros(len(new), dtype=torch.bool)])
        return torch.as_tensor([ids[t] for t in texts], dtype=torch.long)

    @property
    def state_texts_(self) -> List[str]:
        return list(self._state_ids)

    @property
    def candidate_states_(self) -> List[str]:
        check_is_fitted(self, "model_")
        texts = self.state_texts_
        return [texts[i] for i in torch.nonzero(self._is_candidate).flatten().tolist()]

    def _device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _seed(self):
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

    # ---- fitting -------------------------------------------------------------------------------

    def fit(self, X, y=None, groups=None):
        """Learn the transition model from (state, action) pairs X and next states y, all text.

        groups (optional): one id per transition, e.g. the problem it comes from. With groups,
        training batches are drawn within a group (hard, same-problem negatives, as in the paper)
        when groups are large enough, and the early-stopping split holds out whole groups.
        """
        states, actions, nxt = _check_pairs(X, y)
        if nxt is None:
            raise ValueError("fit needs the next states: pass y, or a DataFrame with a 'next_state' column")
        if self.loss_space not in ("absolute", "delta"):
            raise ValueError("loss_space must be 'absolute' or 'delta'")
        self._seed()
        self.device_ = self._device()
        self.encoder_ = get_encoder(self.encoder, device=None if self.device == "auto" else self.device)

        texts = list(dict.fromkeys(states + nxt))
        S0 = self.encoder_(texts)
        self._state_ids, self._S = {t: i for i, t in enumerate(texts)}, torch.from_numpy(S0).float()
        self._is_candidate = torch.ones(len(texts), dtype=torch.bool)
        acts = list(dict.fromkeys(actions))
        self._action_ids, self._A = {a: i for i, a in enumerate(acts)}, torch.from_numpy(self.encoder_(acts)).float()

        s_rows, a_rows, p_rows = self._rows(states, "state"), self._rows(actions, "action"), self._rows(nxt, "state")
        labels = _labels(groups, len(states))
        codes = np.zeros(len(states), dtype=np.int64) if labels is None else pd.factorize(np.asarray(labels))[0]
        tri = pd.DataFrame({"s_emb_idx": s_rows.numpy(), "a_idx": a_rows.numpy(),
                            "sp_emb_idx": p_rows.numpy(), "problem_idx": codes.astype(np.int64)})
        self._group_rows = _group_rows(labels, s_rows.numpy(), p_rows.numpy())

        n = len(tri)
        train_idx, val_idx = np.arange(n), np.array([], dtype=np.int64)
        if self.early_stopping and n >= 20:
            if groups is not None and len(np.unique(codes)) >= 3:
                sp = GroupShuffleSplit(n_splits=1, test_size=self.validation_fraction, random_state=self.random_state)
                train_idx, val_idx = next(sp.split(np.zeros(n), groups=codes))
            else:
                sp = ShuffleSplit(n_splits=1, test_size=self.validation_fraction, random_state=self.random_state)
                train_idx, val_idx = next(sp.split(np.zeros(n)))

        group_sizes = np.bincount(codes[train_idx]) if groups is not None else np.array([0])
        grouped = groups is not None and np.median(group_sizes[group_sizes > 0]) >= 16
        args = _Args(lr=self.lr, seed=self.random_state or 0, batch_size=self.batch_size, epochs=self.max_epochs,
                     tau=self.tau, action_weight=self.action_weight, loss_space=self.loss_space,
                     split="problem_grouped" if grouped else "random", projection_dim=self.projection_dim,
                     projection_layers=self.projection_layers, hidden_size=self.hidden_size,
                     n_layers=self.n_layers, dropout=self.dropout, transition="mlp")
        self.model_ = build_model(self._S.shape[1], self._A.shape[1], args, self.device_)
        S, A = self._S.to(self.device_), self._A.to(self.device_)

        self.history_ = []
        best = {"score": -1.0, "epoch": 0, "state": None}
        val = None
        if len(val_idx):
            val = (s_rows.numpy()[val_idx], a_rows.numpy()[val_idx], p_rows.numpy()[val_idx],
                   None if labels is None else [labels[i] for i in val_idx])

        def on_epoch(epoch: int, loss: float) -> bool:
            record = {"epoch": epoch, "loss": loss}
            if val is not None and (epoch % self.eval_every == 0 or epoch == self.max_epochs):
                h = self._evaluate_rows(*val, ks=(1,), pool_size=128, seed=0)["mrr"]
                record["val_mrr"] = h
                self.model_.train()
                if h > best["score"]:
                    best.update(score=h, epoch=epoch, state=copy.deepcopy(self.model_.state_dict()))
                elif epoch - best["epoch"] >= self.patience:
                    self.history_.append(record)
                    return True
            self.history_.append(record)
            return False

        train_transition(self.model_, S, A, tri, train_idx.tolist(), args, self.device_,
                         log_every=max(1, self.eval_every), verbose=self.verbose, callback=on_epoch)
        if best["state"] is not None:
            self.model_.load_state_dict(best["state"])
        self.best_epoch_ = best["epoch"] if best["state"] is not None else len(self.history_)
        self.best_validation_score_ = best["score"] if best["state"] is not None else None
        self.model_.eval()
        self.n_transitions_ = n
        return self

    # ---- prediction ----------------------------------------------------------------------------

    @torch.no_grad()
    def _latent(self, rows: torch.Tensor) -> torch.Tensor:
        S = self._S[rows].to(self.device_)
        return F.normalize(self.model_.state_projection_head(S), dim=-1)

    @torch.no_grad()
    def _predict_latent(self, s_rows: torch.Tensor, a_rows: torch.Tensor) -> torch.Tensor:
        self.model_.eval()
        out = []
        for i in range(0, len(s_rows), 1024):
            s = self._S[s_rows[i:i + 1024]].to(self.device_)
            a = self._A[a_rows[i:i + 1024]].to(self.device_)
            out.append(F.normalize(self.model_(s, a), dim=-1))
        return torch.cat(out) if out else torch.empty(0, self.projection_dim, device=self.device_)

    def add_states(self, texts: Sequence[str]):
        """Make these states candidates for predict() and rollout() (e.g. the states of a new problem)."""
        check_is_fitted(self, "model_")
        rows = self._rows(list(texts), "state")
        self._is_candidate[rows] = True
        return self

    def project(self, texts: Sequence[str]) -> np.ndarray:
        """The learned, normalized latent vectors of these states (for analysis and plots)."""
        check_is_fitted(self, "model_")
        return self._latent(self._rows(list(texts), "state")).cpu().numpy()

    def predict_scores(self, X, candidates: Candidates = None):
        """Cosine score of every candidate next state for every query.

        candidates: None (every state seen in fit or added with add_states), one shared list of
        texts, or one list per query. Returns (scores, candidate_texts): an array (n_queries,
        n_candidates) with a shared list, or lists of arrays and lists with per-query candidates.
        """
        check_is_fitted(self, "model_")
        states, actions, _ = _check_pairs(X)
        pred = self._predict_latent(self._rows(states, "state"), self._rows(actions, "action"))
        per_query = candidates is not None and len(candidates) > 0 and not isinstance(candidates[0], str)
        if per_query:
            if len(candidates) != len(states):
                raise ValueError("with one candidate list per query, pass as many lists as queries")
            scores = [(self._latent(self._rows(list(c), "state")) @ pred[i]).cpu().numpy()
                      for i, c in enumerate(candidates)]
            return scores, [list(c) for c in candidates]
        texts = self.candidate_states_ if candidates is None else list(candidates)
        if not texts:
            raise ValueError("no candidate states")
        return (pred @ self._latent(self._rows(texts, "state")).T).cpu().numpy(), texts

    def predict_topk(self, X, k: int = 5, candidates: Candidates = None) -> List[List[str]]:
        """The k most likely next states of each query, best first."""
        scores, texts = self.predict_scores(X, candidates)
        if isinstance(scores, list):
            return [[t[j] for j in np.argsort(-s, kind="stable")[:k]] for s, t in zip(scores, texts)]
        order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return [[texts[j] for j in row] for row in order]

    def predict(self, X, candidates: Candidates = None) -> List[str]:
        """The most likely next state of each (state, action) pair, among the candidate states.

        By default the candidates are the states seen in fit (and any added with add_states). For
        a problem the model never saw, pass that problem's states as `candidates`, or add them
        once with add_states: the prediction is always one of the candidates.
        """
        return [top[0] for top in self.predict_topk(X, k=1, candidates=candidates)]

    def rollout(self, state: str, actions: Sequence[str], candidates: Candidates = None) -> List[str]:
        """Apply a sequence of actions from `state`, feeding each predicted state back in.

        Each prediction is snapped to the nearest candidate state before the next step, as in the
        paper's multi-step experiments. Returns the predicted state after each action.
        """
        out, current = [], state
        for action in actions:
            current = self.predict([(current, action)], candidates=candidates)[0]
            out.append(current)
        return out

    # ---- evaluation ----------------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_rows(self, s_rows, a_rows, p_rows, labels: Optional[List[str]], ks=(1, 5, 10),
                       pool_size: Optional[int] = 128, seed: Optional[int] = 0,
                       max_queries: Optional[int] = None) -> Dict:
        """Rank each true next state among distractors: the known states of the query's group
        (fitted and evaluated) when labels are given, else every known state."""
        rng = np.random.default_rng(seed)
        s_rows, a_rows, p_rows = (np.asarray(r, dtype=np.int64) for r in (s_rows, a_rows, p_rows))
        q = np.arange(len(s_rows))
        if max_queries is not None and len(q) > max_queries:
            q = np.sort(rng.choice(q, max_queries, replace=False))
        if labels is None:
            known = torch.nonzero(self._is_candidate).flatten().numpy()
            pools = {None: np.unique(np.concatenate([known, s_rows, p_rows]))}
            key = [None] * len(s_rows)
        else:
            extra = _group_rows(labels, s_rows, p_rows)
            pools = {g: np.array(sorted(self._group_rows.get(g, set()) | extra[g])) for g in extra}
            key = labels
        all_rows = np.unique(np.concatenate(list(pools.values())))
        pos = {int(r): i for i, r in enumerate(all_rows)}
        latent = self._latent(torch.as_tensor(all_rows))
        pred = self._predict_latent(torch.as_tensor(s_rows[q]), torch.as_tensor(a_rows[q]))
        ranks, sizes = np.empty(len(q), dtype=np.int64), np.empty(len(q), dtype=np.int64)
        for j, i in enumerate(q):
            others = pools[key[i]]
            others = others[others != p_rows[i]]
            if pool_size is not None and len(others) > pool_size - 1:
                others = rng.choice(others, pool_size - 1, replace=False)
            cand = torch.as_tensor([pos[int(p_rows[i])]] + [pos[int(r)] for r in others], device=latent.device)
            sc = latent[cand] @ pred[j]
            ranks[j] = int((sc >= sc[0]).sum())          # ties count against the truth
            sizes[j] = len(cand)
        out = {f"hit@{k}": float((ranks <= k).mean()) for k in ks}
        out.update({f"chance@{k}": float(np.mean(np.minimum(k, sizes) / sizes)) for k in ks})
        out.update(mrr=float((1.0 / ranks).mean()), mean_rank=float(ranks.mean()),
                   n_queries=int(len(q)), mean_pool_size=float(sizes.mean()))
        return out

    def evaluate(self, X, y, groups=None, ks: Sequence[int] = (1, 5, 10), pool_size: Optional[int] = 128,
                 distractors: str = "auto", max_queries: Optional[int] = None) -> Dict:
        """Hit@k under the paper's protocol, with the chance level of the same pools.

        Each query ranks its true next state against up to `pool_size - 1` distractors, drawn
        without replacement and never equal to the truth; a tie counts against the truth.
        distractors="group": the known states of the query's own group (its problem), as for the
        paper's Extrapolation. distractors="all": every known state, as for its Interpolation.
        "auto" picks "group" when groups are given. pool_size=None ranks against the whole pool.
        Returns hit@k and chance@k for each k, mrr (mean reciprocal rank), mean_rank, n_queries
        and mean_pool_size.
        """
        check_is_fitted(self, "model_")
        states, actions, nxt = _check_pairs(X, y)
        if nxt is None:
            raise ValueError("evaluate needs the true next states y")
        if distractors == "auto":
            distractors = "group" if groups is not None else "all"
        if distractors not in ("group", "all"):
            raise ValueError("distractors must be 'group', 'all' or 'auto'")
        if distractors == "group" and groups is None:
            raise ValueError("distractors='group' needs groups")
        labels = _labels(groups, len(states)) if distractors == "group" else None
        return self._evaluate_rows(self._rows(states, "state").numpy(), self._rows(actions, "action").numpy(),
                                   self._rows(nxt, "state").numpy(), labels, ks=tuple(ks), pool_size=pool_size,
                                   seed=self.random_state, max_queries=max_queries)

    def score(self, X, y, groups=None, k: int = 5, pool_size: Optional[int] = 128,
              distractors: str = "auto") -> float:
        """Hit@k under the paper's protocol (see evaluate): higher is better."""
        return self.evaluate(X, y, groups=groups, ks=(k,), pool_size=pool_size,
                             distractors=distractors)[f"hit@{k}"]

    # ---- persistence ---------------------------------------------------------------------------

    def save(self, path) -> None:
        """Save the fitted model, its parameters and every encoded state and action."""
        check_is_fitted(self, "model_")
        params = self.get_params()
        params["encoder"] = self.encoder if isinstance(self.encoder, str) else None
        torch.save({"format": "embedplan-estimator/1", "params": params,
                    "model": {k: v.cpu() for k, v in self.model_.state_dict().items()},
                    "state_texts": self.state_texts_, "S": self._S, "is_candidate": self._is_candidate,
                    "action_texts": list(self._action_ids), "A": self._A, "history": self.history_,
                    "group_rows": {g: sorted(r) for g, r in self._group_rows.items()},
                    "best_epoch": self.best_epoch_, "n_transitions": self.n_transitions_}, path)

    @classmethod
    def load(cls, path, encoder: Union[str, Callable, None] = None, device: str = "auto") -> "EmbedPlan":
        """Load a model saved with save(). Pass `encoder` if it was fit with your own function."""
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob.get("format") != "embedplan-estimator/1":
            raise ValueError(f"{path} is not an EmbedPlan estimator file")
        params = dict(blob["params"], device=device)
        if encoder is not None:
            params["encoder"] = encoder
        if params["encoder"] is None:
            raise ValueError("this model was fit with a custom encoder function: pass it as encoder=")
        est = cls(**params)
        est.device_ = est._device()
        est.encoder_ = get_encoder(est.encoder)
        est._state_ids = {t: i for i, t in enumerate(blob["state_texts"])}
        est._S, est._is_candidate = blob["S"], blob["is_candidate"]
        est._action_ids = {a: i for i, a in enumerate(blob["action_texts"])}
        est._A = blob["A"]
        est._group_rows = {g: set(r) for g, r in blob["group_rows"].items()}
        args = _Args(projection_dim=est.projection_dim, projection_layers=est.projection_layers,
                     hidden_size=est.hidden_size, n_layers=est.n_layers, dropout=est.dropout, transition="mlp")
        est.model_ = build_model(est._S.shape[1], est._A.shape[1], args, est.device_)
        est.model_.load_state_dict(blob["model"])
        est.model_.eval()
        est.history_, est.best_epoch_, est.n_transitions_ = blob["history"], blob["best_epoch"], blob["n_transitions"]
        return est


def _labels(groups, n: int) -> Optional[List[str]]:
    """Group ids as strings (so ids given as ints, numpy ints or strings compare equal later)."""
    if groups is None:
        return None
    labels = [str(g) for g in np.asarray(groups).tolist()]
    if len(labels) != n:
        raise ValueError(f"groups has {len(labels)} ids for {n} transitions")
    return labels


def _group_rows(labels: Optional[List[str]], s_rows, p_rows) -> Dict[str, set]:
    """The state rows (current and next states) seen in each group."""
    out: Dict[str, set] = {}
    if labels is None:
        return out
    for g, s, p in zip(labels, s_rows, p_rows):
        out.setdefault(g, set()).update((int(s), int(p)))
    return out


class _Args:
    """The attribute bag build_model and train_transition read."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
