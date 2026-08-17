# Benchmark Models

This document provides a formal description of the benchmark sequence models compared
against the Transformer in Scipione et al. (2026). All models are described using the
same notation adopted in the paper.

---

## Notation

| Symbol | Description |
|:---:|:---|
| $B$ | Batch size |
| $L_x$ | Input sequence length |
| $L_y$ | Target sequence length |
| $p_x$ | Number of input token features |
| $p_y$ | Number of target token features |
| $\mathbf{X}^{(i)} \in \mathbb{R}^{p_x}$ | $i$-th input token (semantic) |
| $\mathbf{Y}^{(i)} \in \mathbb{R}^{p_y}$ | $i$-th output token (semantic) |
| $\mathbf{x}^{(i)} \in \mathbb{R}^{d}$ | $i$-th embedded input vector |
| $\mathbf{y}^{(i)} \in \mathbb{R}^{d}$ | $i$-th embedded target context vector (value blanked) |
| $d$ | Embedding dimension (model space) |
| $\varphi_j[\cdot]$ | Feature embedding map for the $j$-th token feature |
| $\mathbf{c} \in \mathbb{R}^{d_c}$ | Context vector extracted from the input sequence |
| $d_h$ | Hidden dimension of the RNN or MLP |
| $n_\text{layers}$ | Number of stacked recurrent or SSM layers |
| $k_l$ | Kernel size of the TCN at layer $l$ |
| $C$ | Number of TCN output channels at the last layer |
| $d_\text{state}$ | SSM state dimension (S6/Mamba) |
| $d_\text{inner}$ | Expanded inner dimension of the SSM (= $d \cdot \text{expand}$) |
| $\mathbf{W}_\text{head}$ | Final linear projection weights |
| $\hat{y}^{(i)}$ | Scalar prediction for the $i$-th target token |

---

## Shared Input–Output Framework

All benchmark models share the same embedding infrastructure and output head. They differ
only in their sequence-modeling component.

### Input Embedding

Each input token $\mathbf{X}^{(i)}$ is embedded into $\mathbb{R}^d$ by applying
feature-specific maps $\varphi_j$ and summing the results (following Eq. 7 of the paper):

$$
\mathbf{x}^{(i)} = \varphi_1\!\left[X_1^{(i)}\right] + \varphi_2\!\left[X_2^{(i)}\right] + \cdots + \varphi_{p_x}\!\left[X_{p_x}^{(i)}\right]
$$

This produces the model input sequence $\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(L_x)}\}$.

### Target Embedding

The value feature $Y_1^{(i)}$ is the quantity to be predicted and is therefore blanked (set to
zero) before embedding (following Eq. 20 of the paper). Only context features are embedded:

$$
\mathbf{y}^{(i)} = \varphi_2\!\left[Y_2^{(i)}\right] + \cdots + \varphi_{p_y}\!\left[Y_{p_y}^{(i)}\right]
$$

### Context Broadcast and Output Head

Each model produces a single context vector $\mathbf{c} \in \mathbb{R}^{d_c}$ from the
input sequence. This vector is broadcast across the $L_y$ target positions and concatenated
with the target embeddings:

$$
\mathbf{z}^{(i)} = \left[\mathbf{y}^{(i)},\; \mathbf{c}\right] \in \mathbb{R}^{d + d_c}
$$

A shared linear head then maps each $\mathbf{z}^{(i)}$ to a scalar prediction:

$$
\hat{y}^{(i)} = \mathbf{W}_\text{head}\, \mathbf{z}^{(i)} + b_\text{head}, \quad i = 1, \ldots, L_y
$$

---

## 1. Gated Recurrent Unit (GRU)

### Architecture

The GRU is a recurrent neural network that processes the input sequence
$(\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(L_x)})$ by maintaining a hidden state updated
at each position. For a single-layer GRU, the update equations at position $i$ are:

$$
\mathbf{r}^{(i)} = \sigma\!\left(\mathbf{W}_r \mathbf{x}^{(i)} + \mathbf{U}_r \mathbf{h}^{(i-1)} + \mathbf{b}_r\right)
$$

$$
\mathbf{u}^{(i)} = \sigma\!\left(\mathbf{W}_u \mathbf{x}^{(i)} + \mathbf{U}_u \mathbf{h}^{(i-1)} + \mathbf{b}_u\right)
$$

$$
\tilde{\mathbf{h}}^{(i)} = \tanh\!\left(\mathbf{W}_h \mathbf{x}^{(i)} + \mathbf{U}_h \left(\mathbf{r}^{(i)} \odot \mathbf{h}^{(i-1)}\right) + \mathbf{b}_h\right)
$$

$$
\mathbf{h}^{(i)} = \left(1 - \mathbf{u}^{(i)}\right) \odot \mathbf{h}^{(i-1)} + \mathbf{u}^{(i)} \odot \tilde{\mathbf{h}}^{(i)}
$$

where $\sigma$ is the sigmoid function, $\odot$ is elementwise multiplication, and
$\mathbf{W}_r^{(l)}, \mathbf{W}_u^{(l)}, \mathbf{W}_h^{(l)} \in \mathbb{R}^{d_h \times d_h}$,
$\mathbf{U}_r^{(l)}, \mathbf{U}_u^{(l)}, \mathbf{U}_h^{(l)} \in \mathbb{R}^{d_h \times d_h}$
are learnable matrices specific to layer $l$ (for the first layer,
$\mathbf{W}_\star^{(1)} \in \mathbb{R}^{d_h \times d}$ since the input is $\mathbf{x}^{(i)} \in \mathbb{R}^d$).
We denote the mapping defined by the four equations above, with the weight matrices of
layer $l$, as $\text{GRU}_l$.

For $n_\text{layers}$ stacked layers, the hidden state of layer $l$ at position $i$ is
$\mathbf{h}_l^{(i)}$. Each layer receives as input the hidden-state sequence of the
layer below it, i.e.

$$
\mathbf{h}_l^{(i)} = \text{GRU}_l\!\left(\mathbf{h}_{l-1}^{(i)},\; \mathbf{h}_l^{(i-1)}\right), \quad l = 1, \ldots, n_\text{layers}
$$

with the base case $\mathbf{h}_0^{(i)} = \mathbf{x}^{(i)}$. Layers therefore interact
in a **temporal cascade along the feature axis**: layer $l+1$ processes the hidden-state
sequence emitted by layer $l$ over all positions, not the original input. This allows deeper
layers to build increasingly abstract representations, each still evolving sequentially over
the full input length $L_x$.

### Context Extraction

Missing values are handled via packed-sequence processing: positions flagged as missing
are excluded from the recurrent computation. The context vector is formed by **concatenating
the final hidden states of all layers** (at the last valid position $L_x$):

$$
\mathbf{c} = \left[\mathbf{h}_1^{(L_x)}, \mathbf{h}_2^{(L_x)}, \ldots, \mathbf{h}_{n_\text{layers}}^{(L_x)}\right] \in \mathbb{R}^{n_\text{layers} \cdot d_h}
$$

Each layer contributes an independent $d_h$-dimensional slice to the context, so the
context grows linearly with depth. Unlike the TCN (see below), the layers do not pool
information together into a single read-out channel; instead, each layer's final state
is read out independently.

### Output

The context vector is broadcast and concatenated with the target embeddings, and a linear
head produces the predictions:

$$
\hat{y}^{(i)} = \mathbf{W}_\text{head}\!\left[\mathbf{y}^{(i)},\; \mathbf{c}\right] + b_\text{head}
$$

where $\mathbf{W}_\text{head} \in \mathbb{R}^{1 \times (d + n_\text{layers} \cdot d_h)}$.

---

## 2. Long Short-Term Memory (LSTM)

### Architecture

The LSTM extends the GRU with an explicit cell state $\mathbf{s}^{(i)}$ that carries
long-range information. For a single-layer LSTM, the update equations at position $i$ are:

$$
\mathbf{f}^{(i)} = \sigma\!\left(\mathbf{W}_f \mathbf{x}^{(i)} + \mathbf{U}_f \mathbf{h}^{(i-1)} + \mathbf{b}_f\right)
$$

$$
\mathbf{g}^{(i)} = \sigma\!\left(\mathbf{W}_g \mathbf{x}^{(i)} + \mathbf{U}_g \mathbf{h}^{(i-1)} + \mathbf{b}_g\right)
$$

$$
\mathbf{o}^{(i)} = \sigma\!\left(\mathbf{W}_o \mathbf{x}^{(i)} + \mathbf{U}_o \mathbf{h}^{(i-1)} + \mathbf{b}_o\right)
$$

$$
\tilde{\mathbf{s}}^{(i)} = \tanh\!\left(\mathbf{W}_s \mathbf{x}^{(i)} + \mathbf{U}_s \mathbf{h}^{(i-1)} + \mathbf{b}_s\right)
$$

$$
\mathbf{s}^{(i)} = \mathbf{f}^{(i)} \odot \mathbf{s}^{(i-1)} + \mathbf{g}^{(i)} \odot \tilde{\mathbf{s}}^{(i)}
$$

$$
\mathbf{h}^{(i)} = \mathbf{o}^{(i)} \odot \tanh\!\left(\mathbf{s}^{(i)}\right)
$$

where $\mathbf{f}^{(i)}$, $\mathbf{g}^{(i)}$, $\mathbf{o}^{(i)}$ are the forget, input, and
output gates, and all $\mathbf{W}_\star \in \mathbb{R}^{d_h \times d}$,
$\mathbf{U}_\star \in \mathbb{R}^{d_h \times d_h}$.

For $n_\text{layers}$ stacked layers, the depth works identically to the GRU: each layer
$l$ takes the hidden-state sequence of layer $l-1$ as input, with
$\mathbf{h}_0^{(i)} = \mathbf{x}^{(i)}$. Both the hidden state $\mathbf{h}_l^{(i)}$ and
the cell state $\mathbf{s}_l^{(i)}$ are propagated independently per layer.

### Context Extraction

Identical to the GRU: missing positions are excluded via packed sequences, and the
context is formed from the final hidden states (at position $L_x$) across all layers:

$$
\mathbf{c} = \left[\mathbf{h}_1^{(L_x)}, \ldots, \mathbf{h}_{n_\text{layers}}^{(L_x)}\right] \in \mathbb{R}^{n_\text{layers} \cdot d_h}
$$

### Output

$$
\hat{y}^{(i)} = \mathbf{W}_\text{head}\!\left[\mathbf{y}^{(i)},\; \mathbf{c}\right] + b_\text{head}
$$

where $\mathbf{W}_\text{head} \in \mathbb{R}^{1 \times (d + n_\text{layers} \cdot d_h)}$.

---

## 3. Temporal Convolutional Network (TCN)

### Architecture

The TCN replaces recurrence with stacked causal 1D convolutions, each with exponentially
increasing dilation. Here $n_\text{layers}$ plays a structurally different role than in
the RNN: layers form a **strict cascade** where each layer consumes the entire output
sequence of the previous layer and produces a new sequence of the same length $L_x$,
potentially with a different number of channels.

#### Dilation

A standard 1D convolution of kernel size $k_l$ at position $i$ combines positions
$i, i-1, \ldots, i-(k_l-1)$, giving a local receptive field of $k_l$. A **dilated convolution**
with dilation factor $\delta_l$ instead combines positions $i, i-\delta_l, i-2\delta_l, \ldots,
i-(k_l-1)\delta_l$, spanning $(k_l-1)\delta_l+1$ positions with still only $k_l$ weights.
Setting $\delta_l = 2^{l-1}$ at layer $l$ (1-indexed) makes the receptive field grow
**exponentially** with depth. In the general case, each layer $l$ may have its own kernel
size $k_l$, and after $n_\text{layers}$ layers the cumulative receptive field is:

$$
R = 1 + \sum_{l=1}^{n_\text{layers}} (k_l - 1)\cdot 2^{l-1}
$$

**Uniform kernel size.** In this work we set $k_l = k$ for all layers. Using the geometric
series $\sum_{l=0}^{n-1} 2^l = 2^n - 1$, the receptive field simplifies to:

$$
R = 1 + (k-1)\left(2^{n_\text{layers}} - 1\right)
$$

For example, $k=3$ and $n_\text{layers}=8$ yields $R = 511$, covering sequences up to 500
positions long with only 8 layers and $O(k)$ parameters per layer.

#### Causal constraint

All convolutions use **stride 1**. At layer $l$, $(k_l - 1) \cdot 2^{l-1}$ zeros are
prepended (causal padding) so that $a_l^{(i)}$ depends only on positions $j \leq i$.
Stride 1 together with this padding ensures the output sequence length remains $L_x$
at every layer.

#### Layer-by-layer computation

Denote by $a_l^{(i)} \in \mathbb{R}^{C_l}$ the activation at position $i$ after the
$l$-th layer, with $C_l$ output channels. The cascade starts from the embedded input:
$a_0^{(i)} = \mathbf{x}^{(i)} \in \mathbb{R}^d$ ($C_0 = d$, the embedding dimension).
The output channel count of the last layer, $C = C_{n_\text{layers}}$, plays the same
structural role as the hidden dimension $d_h$ of the RNN: it is the width of the learned
representation read out as the context vector. Each subsequent layer applies:

$$
a_l^{(i)} = \text{Dropout}\!\left(\text{ReLU}\!\left(\sum_{m=0}^{k_l-1} \mathbf{K}_l^{(m)}\, a_{l-1}^{(i - m \cdot 2^{l-1})} + \mathbf{b}_l\right)\right), \quad i = 1, \ldots, L_x
$$

where $\mathbf{K}_l \in \mathbb{R}^{C_l \times C_{l-1} \times k_l}$ is the weight tensor of
layer $l$, and $\mathbf{K}_l^{(m)} \in \mathbb{R}^{C_l \times C_{l-1}}$ denotes its $m$-th
slice along the kernel-tap dimension ($m = 0, \ldots, k_l - 1$). The index $m$ identifies
the tap (lag $m \cdot 2^{l-1}$), **not** the sequence position: the same $\mathbf{K}_l^{(m)}$
is reused at every position $i$, which is precisely the weight-sharing property of
convolutions. Positions $i - m \cdot 2^{l-1} \leq 0$ are zero-padded.

The sequence length is preserved at all layers; only the feature (channel) dimension and the
receptive field change (entries below assume the uniform choice $k_l = k$):

| Layer $l$ | Input dim | Output dim | Dilation $\delta$ | Receptive field up to layer $l$ |
|:---:|:---:|:---:|:---:|:---:|
| 1 | $d$ | $C_1$ | $1$ | $k$ |
| 2 | $C_1$ | $C_2$ | $2$ | $k + (k-1)\cdot 2$ |
| 3 | $C_2$ | $C_3$ | $4$ | $k + (k-1)\cdot(2+4)$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| $n_\text{layers}$ | $C_{n_\text{layers}-1}$ | $C$ | $2^{n_\text{layers}-1}$ | $R$ |

At the last layer, $a_{n_\text{layers}}^{(L_x)}$ is the only output position that has
a full receptive field of $R$ positions (provided $L_x \geq R$).

### Context Extraction

The context is taken as the activation at the final position $L_x$ of the last
convolutional layer:

$$
\mathbf{c} = a_{n_\text{layers}}^{(L_x)} \in \mathbb{R}^{C}
$$

where $C = C_{n_\text{layers}}$ is the number of channels in the final layer. Unlike the
RNN, only a single layer contributes to the context, and only at a single position.

### Output

$$
\hat{y}^{(i)} = \mathbf{W}_\text{head}\!\left[\mathbf{y}^{(i)},\; \mathbf{c}\right] + b_\text{head}
$$

where $\mathbf{W}_\text{head} \in \mathbb{R}^{1 \times (d + C)}$.

---

## 4. Multi-Layer Perceptron (MLP)

The MLP serves as a non-sequential baseline. It makes no use of the ordering of the input
tokens: the $L_x$ embedded input vectors are collapsed into a single vector by mean pooling
before being processed by a fully connected network.

### Architecture

The pooled representation of the input sequence $(\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(L_x)})$ is:

$$
\bar{\mathbf{x}} = \frac{1}{L_x} \sum_{i=1}^{L_x} \mathbf{x}^{(i)} \in \mathbb{R}^d
$$

The MLP then applies $n_\text{layers}$ linear transformations with ReLU activations and
dropout:

$$
\mathbf{a}^{(0)} = \bar{\mathbf{x}}
$$

$$
\mathbf{a}^{(l)} = \text{Dropout}\!\left(\text{ReLU}\!\left(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)\right), \quad l = 1, \ldots, n_\text{layers}
$$

where $\mathbf{W}^{(l)} \in \mathbb{R}^{h_l \times h_{l-1}}$ and $h_0 = d$. The hidden
widths $h_1, \ldots, h_{n_\text{layers}}$ are hyperparameters.

### Context Extraction

The context is the output of the final MLP layer:

$$
\mathbf{c} = \mathbf{a}^{(n_\text{layers})} \in \mathbb{R}^{h_{n_\text{layers}}}
$$

### Output

$$
\hat{y}^{(i)} = \mathbf{W}_\text{head}\!\left[\mathbf{y}^{(i)},\; \mathbf{c}\right] + b_\text{head}
$$

where $\mathbf{W}_\text{head} \in \mathbb{R}^{1 \times (d + h_{n_\text{layers}})}$.

---

## 5. Bidirectional State Space Sequence Model (S6 / BiMamba)

### Architecture

The S6 model is based on the Mamba selective state space model (SSM), which processes
sequences with a data-dependent recurrence over a continuous latent state. A bidirectional
wrapper (BiMamba) processes the input sequence in both the forward and reverse directions,
enabling the context to capture both past and future information in the input.

#### Single Mamba block

The Mamba block first projects each input vector $\mathbf{x}^{(i)} \in \mathbb{R}^d$ to an
expanded dimension $d_\text{inner} = d \cdot \text{expand}$, splitting the result into two
branches:

$$
\left[{x'}^{(i)},\; z^{(i)}\right] = \mathbf{W}_\text{in}\, \mathbf{x}^{(i)}, \quad {x'}^{(i)},\; z^{(i)} \in \mathbb{R}^{d_\text{inner}}
$$

A causal depthwise 1D convolution of kernel size $d_\text{conv}$ is applied to the sequence
$({x'}^{(1)}, \ldots, {x'}^{(L_x)})$, followed by SiLU activation. The convolved sequence
is passed through the selective scan to produce $y^{(i)} \in \mathbb{R}^{d_\text{inner}}$.

The SSM state transition is governed by input-dependent matrices
$\bar{\mathbf{A}}^{(i)} \in \mathbb{R}^{d_\text{inner} \times d_\text{state}}$ and
$\bar{\mathbf{B}}^{(i)} \in \mathbb{R}^{d_\text{inner} \times d_\text{state}}$,
computed from ${x'}^{(i)}$ via learned projections. The state update at position $i$ reads:

$$
\mathbf{s}^{(i)} = \bar{\mathbf{A}}^{(i)} \odot \mathbf{s}^{(i-1)} + \bar{\mathbf{B}}^{(i)} \odot {x'}^{(i)}
$$

$$
y^{(i)} = \mathbf{s}^{(i)} \cdot \mathbf{1}_{d_\text{state}} + \mathbf{D} \odot {x'}^{(i)}
$$

where $\mathbf{D} \in \mathbb{R}^{d_\text{inner}}$ is a learned skip-connection parameter.

The gating operation applies a SiLU gate from the split branch $z^{(i)}$:

$$
\tilde{y}^{(i)} = y^{(i)} \odot \text{SiLU}\!\left(z^{(i)}\right)
$$

A final linear projection maps back to dimension $d$:

$$
\mathbf{x}_\text{out}^{(i)} = \mathbf{W}_\text{out}\, \tilde{y}^{(i)} \in \mathbb{R}^{d}
$$

Each Mamba block includes a residual connection with layer normalization:

$$
\text{MambaBlock}\!\left(\mathbf{x}^{(i)}\right) = \mathbf{x}^{(i)} + \text{SSM}\!\left(\text{LayerNorm}\!\left(\mathbf{x}^{(i)}\right)\right)
$$

#### Bidirectional wrapper

For $n_\text{layers}$ stacked blocks, the forward pass processes the sequence left to right,
and the backward pass processes the time-reversed sequence. Denote by $h_\text{fwd}^{(i)}$
and $h_\text{bwd}^{(i)} \in \mathbb{R}^d$ the output vectors at position $i$:

$$
h_\text{fwd}^{(i)} = f_\text{Mamba}^{n_\text{layers}}\!\left(\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(L_x)}\right)^{(i)}
$$

$$
h_\text{bwd}^{(i)} = \text{flip}\!\left(f_\text{Mamba}^{n_\text{layers}}\!\left(\mathbf{x}^{(L_x)}, \ldots, \mathbf{x}^{(1)}\right)\right)^{(i)}
$$

The two outputs are concatenated along the feature dimension at each position:

$$
\mathbf{h}_\text{bi}^{(i)} = \left[h_\text{fwd}^{(i)},\; h_\text{bwd}^{(i)}\right] \in \mathbb{R}^{2d}
$$

### Context Extraction

The context is obtained by mean-pooling the bidirectional output over all positions:

$$
\mathbf{c} = \frac{1}{L_x} \sum_{i=1}^{L_x} \mathbf{h}_\text{bi}^{(i)} \in \mathbb{R}^{2d}
$$

### Output

$$
\hat{y}^{(i)} = \mathbf{W}_\text{head}\!\left[\mathbf{y}^{(i)},\; \mathbf{c}\right] + b_\text{head}
$$

where $\mathbf{W}_\text{head} \in \mathbb{R}^{1 \times (d + 2d)} = \mathbb{R}^{1 \times 3d}$.

---

## Architecture Summary

| Model | Context type | Context dim $d_c$ | Handles missing values |
|:---:|:---:|:---:|:---:|
| GRU | Last RNN hidden state (all layers) | $n_\text{layers} \cdot d_h$ | via packed sequences |
| LSTM | Last RNN hidden state (all layers) | $n_\text{layers} \cdot d_h$ | via packed sequences |
| TCN | Last time-step of final conv. layer | $C$ | not explicitly |
| MLP | Mean-pooled + MLP | $h_{n_\text{layers}}$ | not explicitly |
| S6 | Mean-pooled BiMamba output | $2d$ | not explicitly |

All models share:
- The same `ModularEmbedding` block for input $\{\mathbf{x}^{(i)}\}$ and target context $\{\mathbf{y}^{(i)}\}$
- A single shared linear output head $\mathbf{W}_\text{head}$
- MSE training loss with AdamW optimizer and optional StepLR scheduler

---

## Appendix

### A. Derivation of the TCN receptive field formula

We derive the general formula

$$
R = 1 + \sum_{l=1}^{n_\text{layers}} (k_l - 1)\cdot \delta_l
$$

by tracking how far back in the original input sequence the activation $a_l^{(i)}$ can depend.

**Define** $R_l$ as the receptive field after layer $l$: the number of consecutive positions in
$a_0 = \{x^{(i)}\}$ that $a_l^{(i)}$ can depend on. The base case is $R_0 = 1$, since
$a_0^{(i)} = \mathbf{x}^{(i)}$ depends only on itself.

**Inductive step.** At layer $l$, the dilated convolution combines $k_l$ taps at lags
$0, \delta_l, 2\delta_l, \ldots, (k_l-1)\delta_l$ in its input sequence $\{a_{l-1}^{(i)}\}$.
The leftmost tap reaches position $a_{l-1}^{(i - (k_l-1)\delta_l)}$. By the inductive
hypothesis, this activation itself depends on $R_{l-1}$ original positions, spanning back to
$i - (k_l-1)\delta_l - (R_{l-1} - 1)$ in the original sequence. Therefore the total span of
original positions that $a_l^{(i)}$ can depend on is:

$$
R_l = R_{l-1} + (k_l - 1)\cdot\delta_l
$$

**Telescoping** from $l = 1$ to $l = n_\text{layers}$:

$$
R_{n_\text{layers}} = R_0 + \sum_{l=1}^{n_\text{layers}}(k_l - 1)\cdot\delta_l
= 1 + \sum_{l=1}^{n_\text{layers}}(k_l - 1)\cdot\delta_l
$$

**Specialisation.** With the exponentially growing dilation $\delta_l = 2^{l-1}$ and uniform
kernel size $k_l = k$:

$$
R = 1 + (k-1)\sum_{l=1}^{n_\text{layers}} 2^{l-1}
= 1 + (k-1)\left(2^{n_\text{layers}} - 1\right)
$$

**Sanity check** ($k = 3$, $n_\text{layers} = 3$, $\delta_l = 1, 2, 4$):

| Layer $l$ | Added span $(k-1)\delta_l$ | Cumulative $R_l$ |
|:---:|:---:|:---:|
| 1 | $2 \cdot 1 = 2$ | $3$ |
| 2 | $2 \cdot 2 = 4$ | $7$ |
| 3 | $2 \cdot 4 = 8$ | $15$ |

Formula: $1 + 2\cdot(2^3 - 1) = 1 + 14 = 15$. $\checkmark$
