#!/usr/bin/env python3
"""
Q15 Hallucination — tehnika: Kvantna detekcija halucinacija preko inverznog
CSV signala u aux superpoziciji (čisto kvantno, bez klasičnog softmax-a i bez hibrida).

Koncept:
  - „Hallucination rate“ H ∈ [0, 1] kontroliše balans između „grounded“
    (CSV-potkrepljenog) i „halucinatornog“ (inverznog CSV) režima preko 1 aux qubit-a.
  - Aux qubit: Ry(2α)|0⟩ = cos(α)|0⟩ + sin(α)|1⟩,  α = π·H/2.
    · H = 0 → aux = |0⟩ → pure grounded (CSV-signal).
    · H = 0.5 → aux = (|0⟩+|1⟩)/√2 → 50/50 grounded/hallucinated.
    · H = 1 → aux = |1⟩ → pure hallucination (inverse-CSV — retki brojevi dominiraju).

Kolo (nq + 1 qubit-a):
  1) Ry(2α) na aux.
  2) Kontrolisani StatePreparation(|ψ_CSV⟩) na state registar kad aux = 0
     (|ψ_CSV⟩ = amplitude-encoding freq_vector-a CELOG CSV-a).
  3) Kontrolisani StatePreparation(|ψ_INV⟩) na state registar kad aux = 1
     (|ψ_INV⟩ = amplitude-encoding INVERZNOG freq_vector-a: 1/(f + ε)).

Marginala nad state registrom:
  p[k] = cos²(α)·|ψ_CSV[k]|² + sin²(α)·|ψ_INV[k]|²  →  bias_39  →  NEXT (TOP-7).

Grid i izbor H:
  - Grid SAMO nad nq, po cos(bias@H=0, freq_csv) (grounded baseline).
  - Glavna predikcija na H_MAIN = 0.0 (strogo grounded).
  - Demonstracija „halucinatornog efekta“: NEXT i cos za niz H ∈ {0.0, 0.3, 0.5, 0.7, 1.0}
    — pokazuje kako sa rastom H predikcija klizi ka retkim brojevima iz CSV-a.

Sve deterministički: seed=39; amp_CSV i amp_INV iz CELOG CSV-a.

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39
EPS = 1.0

GRID_NQ = (5, 6)
H_MAIN = 0.0
H_DEMO = (0.0, 0.3, 0.5, 0.7, 1.0)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def inverse_freq_vector(f: np.ndarray, eps: float = EPS) -> np.ndarray:
    """1/(f + ε): retki brojevi (mali f) dobijaju velike vrednosti."""
    return 1.0 / (f + float(eps))


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# Hallucination kolo — aux superpozicija CSV / INV-CSV
# =========================
def build_hallucination_state(H_csv: np.ndarray, nq: int, H_rate: float) -> Statevector:
    """|Ψ⟩ = cos(α)|0⟩|ψ_CSV⟩ + sin(α)|1⟩|ψ_INV⟩,  α = π·H/2."""
    f_csv = freq_vector(H_csv)
    amp_csv = amp_from_freq(f_csv, nq)
    amp_inv = amp_from_freq(inverse_freq_vector(f_csv), nq)

    state = QuantumRegister(nq, name="s")
    aux = QuantumRegister(1, name="a")
    qc = QuantumCircuit(state, aux)

    alpha = float(np.pi * H_rate / 2.0)
    qc.ry(2.0 * alpha, aux[0])

    sp_csv = StatePreparation(amp_csv.tolist()).control(num_ctrl_qubits=1, ctrl_state=0)
    qc.append(sp_csv, [aux[0]] + list(state))

    sp_inv = StatePreparation(amp_inv.tolist()).control(num_ctrl_qubits=1, ctrl_state=1)
    qc.append(sp_inv, [aux[0]] + list(state))

    return Statevector(qc)


def hallucination_state_probs(H_csv: np.ndarray, nq: int, H_rate: float) -> np.ndarray:
    sv = build_hallucination_state(H_csv, nq, H_rate)
    p = np.abs(sv.data) ** 2
    dim_s = 2 ** nq
    mat = p.reshape(2, dim_s)
    p_s = mat.sum(axis=0)
    s = float(p_s.sum())
    return p_s / s if s > 0 else p_s


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija SAMO nad nq (H = 0 baseline)
# =========================
def optimize_nq(H_csv: np.ndarray):
    f_csv = freq_vector(H_csv)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        try:
            p = hallucination_state_probs(H_csv, nq, 0.0)
            b = bias_39(p)
            score = cosine(b, f_csv_n)
        except Exception:
            continue
        key = (score, -nq)
        if best is None or key > best[0]:
            best = (key, dict(nq=nq, score=float(score)))
    return best[1] if best else None


def main() -> int:
    H_csv = load_rows(CSV_PATH)
    if H_csv.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q15 Hallucination (aux superpozicija CSV / INV-CSV): CSV:", CSV_PATH)
    print("redova:", H_csv.shape[0], "| seed:", SEED, "| eps:", EPS)

    best = optimize_nq(H_csv)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST nq:", best["nq"],
        "| cos(bias@H=0, freq_csv):", round(float(best["score"]), 6),
    )

    nq_best = int(best["nq"])
    f_csv = freq_vector(H_csv)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX) / N_MAX

    print("--- demonstracija efekta halucinacije (isti nq, različito H) ---")
    for Hr in H_DEMO:
        p_H = hallucination_state_probs(H_csv, nq_best, float(Hr))
        pred_H = pick_next_combination(p_H)
        cos_H = cosine(bias_39(p_H), f_csv_n)
        print(f"H={Hr:.2f}  cos(bias, freq_csv)={cos_H:.6f}  NEXT={pred_H}")

    p_main = hallucination_state_probs(H_csv, nq_best, H_MAIN)
    pred_main = pick_next_combination(p_main)
    print("--- glavna predikcija ---")
    print("H_MAIN=", H_MAIN, "| nq=", nq_best)
    print("predikcija NEXT:", pred_main)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q15 Hallucination (aux superpozicija CSV / INV-CSV): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39 | eps: 1.0
BEST nq: 5 | cos(bias@H=0, freq_csv): 0.900351
--- demonstracija efekta halucinacije (isti nq, različito H) ---
H=0.00  cos(bias, freq_csv)=0.900351  NEXT=(7, 19, 22, 24, 27, 28, 31)
H=0.30  cos(bias, freq_csv)=0.901593  NEXT=(7, 19, 22, 24, 27, 28, 31)
H=0.50  cos(bias, freq_csv)=0.902004  NEXT=(1, 7, 14, 17, 19, 25, 30)
H=0.70  cos(bias, freq_csv)=0.900808  NEXT=(1, 13, 14, 17, 23, 25, 30)
H=1.00  cos(bias, freq_csv)=0.899018  NEXT=(1, 13, 14, 17, 23, 25, 30)
--- glavna predikcija ---
H_MAIN= 0.0 | nq= 5
predikcija NEXT: (7, 19, x, y, z, 28, 31)
"""



"""
Q15_Hallucination_inverz.py — tehnika: Kvantna detekcija halucinacija
preko inverznog CSV signala u aux superpoziciji.

Kolo (nq + 1):
  Aux: Ry(2α)|0⟩ = cos(α)|0⟩ + sin(α)|1⟩,  α = π·H/2.
  Kontrolisano SP|_{aux=0}: aux=0 → state postaje |ψ_CSV⟩  (grounded).
  Kontrolisano SP|_{aux=1}: aux=1 → state postaje |ψ_INV⟩  (hallucinated, 1/(f+ε)).
Marginala: p[k] = cos²(α)·|ψ_CSV[k]|² + sin²(α)·|ψ_INV[k]|².

H interpretacija (LLM semantika):
  H = 0 → grounded (potkrepljeno podacima iz CSV-a — najčešći brojevi dominiraju).
  H = 0.5 → 50/50 mix grounded/hallucinated.
  H = 1 → pure hallucination — retki brojevi iz CSV-a dominiraju (struktura,
          ali van preovlađujuće statistike).

Razlika od Q14 Temperature:
  Q14 (T = 1): high-side = UNIFORM (čisti „chaos“, bez strukture).
  Q15 (H = 1): high-side = INVERSE-CSV (strukturirana halucinacija u smeru
               suprotnom od CSV statistike).

Tehnike:
Amplitude encoding (StatePreparation) za oba režima.
Kontrolisana priprema oba režima iz jednog aux qubit-a (ctrl_state 0/1).
Egzaktni Statevector (bez uzorkovanja).
Deterministička grid-optimizacija nad nq.

Prednosti:
Čisto kvantno: bez klasičnog treninga, bez softmax-a, bez hibrida.
Samo 1 dodatni qubit; vrlo jeftino (nq + 1 qubit-a).
„Hallucination“ je strukturirana (inverse-CSV), ne random — omogućava jasnu
inspekciju u kom pravcu model odstupa od podataka.

Nedostaci:
Marginala je linearna konveksna kombinacija (aux ortogonalna) — nema interferencije.
INV-signal zavisi od ε (regularizator 1/(f+ε)) — izbor ε menja oštrinu halucinacije.
mod-39 readout meša stanja (dim 2^nq ≠ 39); pri nq = 5, 7 pozicija bias-a je 0.
Kao i u Q14, H se ne može pouzdano birati cos-metrikom nad skoro-uniformnim freq_csv.
"""
