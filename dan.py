"""NumPy implementation of the paper's Dendritic Attention Network (DAN)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class TrainHistory:
    losses: list[float]


class AMSGrad:
    """Standard bias-corrected AMSGrad for a dictionary of NumPy arrays."""

    def __init__(
        self,
        params: dict[str, np.ndarray],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.params = params
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}
        self.v_max = {name: np.zeros_like(value) for name, value in params.items()}
        self.step_number = 0

    def step(self, gradients: dict[str, np.ndarray]) -> None:
        self.step_number += 1
        for name, gradient in gradients.items():
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * gradient
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * gradient**2
            self.v_max[name] = np.maximum(self.v_max[name], self.v[name])

            m_hat = self.m[name] / (1.0 - self.beta1**self.step_number)
            v_hat = self.v_max[name] / (1.0 - self.beta2**self.step_number)
            self.params[name] -= self.learning_rate * m_hat / (
                np.sqrt(v_hat) + self.epsilon
            )


class DAN:
    """DAN with D dendritic units, M branches, and I input features.

    Forward equations follow the paper. Gradients are derived from those forward
    equations; this corrects the sign error in the paper's derivative of theta.
    """

    def __init__(
        self,
        input_dim: int,
        branches: int,
        dendritic_units: int,
        kappa: float,
        theta_init: float,
        soma_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        if min(input_dim, branches, dendritic_units) < 1:
            raise ValueError("input_dim, branches and dendritic_units must be positive")
        self.input_dim = input_dim
        self.branches = branches
        self.dendritic_units = dendritic_units
        self.kappa = float(kappa)
        self.soma_threshold = float(soma_threshold)

        rng = np.random.default_rng(seed)
        shape = (dendritic_units, branches, input_dim)
        # With inputs scaled to [0, 1], weights centred near 2*theta place a
        # typical x~=0.5 close to the synaptic sigmoid's high-gradient region.
        # A zero-centred initialization makes every H tiny when theta is
        # 0.5--0.7; multiplying several H values then causes vanishing gradients.
        weight_center = 2.0 * float(theta_init)
        terminal_weights = rng.uniform(0.8, 1.2, size=dendritic_units)
        terminal_weights /= np.sum(terminal_weights)
        self.params: dict[str, np.ndarray] = {
            "w": rng.normal(weight_center, 0.25, size=shape),
            "theta": np.full(shape, float(theta_init)),
            "delta": rng.uniform(0.8, 1.2, size=(dendritic_units, branches)),
            "lambda": terminal_weights,
            "varsigma": np.full(1, 0.5),
        }

    def _forward_with_cache(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        p = self.params
        activation = self.kappa * (
            x[:, None, None, :] * p["w"][None, :, :, :]
            - p["theta"][None, :, :, :]
        )
        h = _sigmoid(activation)
        z = np.prod(h, axis=-1)
        v = np.sum(z * p["delta"][None, :, :], axis=-1)
        s = _sigmoid(self.kappa * (v - self.soma_threshold))
        output = _sigmoid(s @ p["lambda"] - p["varsigma"][0])
        return output, (x, h, z, s, output)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must have shape (n, {self.input_dim})")
        return self._forward_with_cache(x)[0]

    def loss_and_gradients(
        self, x: np.ndarray, target: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        prediction, cache = self._forward_with_cache(x)
        x, h, z, s, prediction = cache
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        batch_size = target.size
        residual = prediction - target
        loss = 0.5 * float(np.mean(residual**2))

        grad_q = residual * prediction * (1.0 - prediction) / batch_size
        grad_lambda = s.T @ grad_q
        grad_varsigma = np.array([-np.sum(grad_q)])

        grad_s = grad_q[:, None] * self.params["lambda"][None, :]
        grad_v = grad_s * self.kappa * s * (1.0 - s)
        grad_delta = np.einsum("bd,bdm->dm", grad_v, z)
        grad_z = grad_v[:, :, None] * self.params["delta"][None, :, :]

        # H is strictly positive after the clipped sigmoid, so Z/H gives the
        # product of all other synaptic outputs on the same branch.
        grad_h = grad_z[:, :, :, None] * z[:, :, :, None] / h
        grad_activation = grad_h * h * (1.0 - h)
        grad_w = np.sum(
            grad_activation * self.kappa * x[:, None, None, :], axis=0
        )
        grad_theta = np.sum(grad_activation * (-self.kappa), axis=0)

        gradients = {
            "w": grad_w,
            "theta": grad_theta,
            "delta": grad_delta,
            "lambda": grad_lambda,
            "varsigma": grad_varsigma,
        }
        return loss, gradients

    def fit(
        self,
        x: np.ndarray,
        target: np.ndarray,
        *,
        epochs: int = 1000,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        seed: int = 42,
        verbose_every: int = 100,
    ) -> TrainHistory:
        x = np.asarray(x, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(x) != len(target):
            raise ValueError("x and target must contain the same number of samples")

        optimizer = AMSGrad(
            self.params,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
        rng = np.random.default_rng(seed)
        losses: list[float] = []

        for epoch in range(1, epochs + 1):
            indices = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                batch = indices[start : start + batch_size]
                _, gradients = self.loss_and_gradients(x[batch], target[batch])
                optimizer.step(gradients)

            prediction = self.predict(x)
            loss = 0.5 * float(np.mean((prediction - target) ** 2))
            losses.append(loss)
            if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
                print(f"epoch={epoch:4d} loss={loss:.8f}")

        return TrainHistory(losses=losses)

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            **self.params,
            input_dim=self.input_dim,
            branches=self.branches,
            dendritic_units=self.dendritic_units,
            kappa=self.kappa,
            soma_threshold=self.soma_threshold,
        )

