# Encoder Benchmark

The benchmark compares different ways to **encode the process chain**. All models
share the same embedding, the same output head and the same training conditions;
only the encoder changes. The target is a vector of scalars that depends on the whole chain:

| Dataset | Target | $n_\text{out}$ |
|:---|:---|:---:|
| Dyconex (`dyconex_260701_seq2slope`) | IST slope of SenseA and SenseB | 2 |
| Ishigami (`ishigami_last_value_ds`, to be generated) | last value of the Ishigami sequence | 1 |

Code: `proT/baseline/baseline_models.py` (`BenchmarkModel`),
`proT/baseline/baseline_pl_modules.py` (`BenchmarkForecaster`),
configs in `config/benchmark/`.

---

## Notation

| Symbol | Description |
|:---:|:---|
| $L$ | Number of tokens in the chain (after padding) |
| $\mathbf{X}^{(i)}$ | $i$-th input token (semantic features) |
| $\mathbf{x}^{(i)} \in \mathbb{R}^{d}$ | $i$-th embedded token |
| $n$ | Number of valid (observed) tokens |
| $\mathbf{c} \in \mathbb{R}^{d_c}$ | Context vector (encoder output) |
| $d_\text{head}$ | Fixed input size of the shared head |

---

## Shared pipeline

$$
\hat{\mathbf{y}} = \text{Head}\Big(\mathbf{W}_a\, \text{Enc}_k\big(\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\big) + \mathbf{b}_a\Big) \in \mathbb{R}^{n_\text{out}}
$$

1. **Chain ordering.** Tokens are sorted by the order feature (`pass` module).
   Tokens with a missing value (padding or a missing measurement) are moved to
   the end and excluded by every encoder. All encoders therefore see the same
   sequence $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}$.
2. **Shared embedding** (`ModularEmbedding`, summation, fixed $d$):
   $\mathbf{x}^{(i)} = \sum_j \varphi_j[X_j^{(i)}]$. Allowed maps are
   `nn_embedding`, `linear` and `sinusoidal`. **No timestamp encoding**: the
   model rejects `time2vec`.
3. **Encoder** $\text{Enc}_k$: the only part that changes (see below). All
   encoders are **causal**: token $i$ depends only on tokens $j \le i$ in the
   sorted chain.
4. **Adapter** $\mathbf{W}_a \in \mathbb{R}^{d_\text{head} \times d_c}$: the only
   layer whose size depends on the encoder. It maps the context to the fixed head input.
5. **Shared head**: a fixed MLP, $d_\text{head} \to [64, 64] \to n_\text{out}$, with
   GELU activations. It is the same class and size for every encoder.

Training (identical for all encoders): MSE on the observed scalars, AdamW,
gradient clipping on the Trainer, k-fold CV, best checkpoint on `val_mae`.

---

## Encoders

### Transformer (`proT_bench`)
Causal self-attention encoder. The causal position is the token rank in the
sorted chain, so ties in the order feature cannot open a bidirectional window.
The readout is attention pooling with a learnable query $\mathbf{q}$:

$$
\mathbf{c} = \sum_{i=1}^{n} \text{softmax}_i\!\left(\frac{(\mathbf{W}_Q\mathbf{q})^\top \mathbf{W}_K\mathbf{h}^{(i)}}{\sqrt{d_{qk}}}\right)\mathbf{W}_V\mathbf{h}^{(i)}, \quad d_c = d
$$

### GRU / LSTM
Recurrent encoder over the packed sequence of valid tokens. The readout is the
last-layer hidden state after token $n$, with $d_c = d_h$.

### TCN
Stacked dilated causal convolutions (left padding only, dilation $2^{l-1}$,
ReLU). The readout is the output at token $n$, with $d_c = C$. The receptive field is
$R = 1 + (k-1)(2^{n_\text{layers}}-1)$.

### MLP (order-agnostic reference)
A token-wise MLP $g$ followed by a causal cumulative mean, read at token $n$:
$\mathbf{c} = \frac{1}{n}\sum_{i \le n} g(\mathbf{x}^{(i)})$, $d_c = h_\text{last}$.
It ignores the chain order, so it measures how much the order matters.

---

## Summary

| Encoder | Readout | $d_c$ | Missing tokens | Causal |
|:---:|:---|:---:|:---|:---:|
| proT_bench | attention pooling | $d$ | attention masks | yes (rank) |
| GRU / LSTM | hidden state at token $n$ | $d_h$ | packed sequence | yes |
| TCN | output at token $n$ | $C$ | moved to tail, ignored | yes |
| MLP | cumulative mean at token $n$ | $h_\text{last}$ | moved to tail, ignored | n/a |

## Hyperparameter optimization

Optuna (`proT/euler_optuna/cli.py`, `model_object: BENCH`) samples only
`model.kwargs.encoder.kwargs.*` (sizes, depth, dropout) and `training.lr`.
`check_bench_params` raises an error if any embedding or head key is sampled.

| Encoder | Sampled |
|:---|:---|
| proT_bench | `e_layers`, `n_heads`, `d_ff`, `d_qk`, `dropout` |
| GRU / LSTM | `d_hidden`, `n_layers`, `dropout` |
| TCN | `channels` (architecture), `kernel`, `dropout` |
| MLP | `hidden` (architecture), `dropout` |

## Experiment setup

```
python scripts/make_benchmark_configs.py   # writes experiments/benchmark/bench_<enc>_<ds>/
python -m proT.euler_optuna.cli paramsopt --exp_id benchmark/bench_GRU_dyconex --mode create
```
