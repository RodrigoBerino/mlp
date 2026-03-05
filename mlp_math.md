# Fundamentação Matemática

Referência matemática do projeto, alinhada com a implementação real em `src/`.

---

## Notação

| Símbolo | Significado |
|---|---|
| `L` | número de camadas (excluindo entrada) |
| `l` | índice da camada, `l ∈ {1, …, L}` |
| `W(l)` | matriz de pesos da camada `l`, shape `[nˡ × nˡ⁻¹]` |
| `b(l)` | vetor de bias da camada `l`, shape `[nˡ]` |
| `z(l)` | pré-ativação da camada `l` |
| `a(l)` | saída pós-ativação da camada `l`; `a(0) = x` (entrada) |
| `ŷ` | saída da rede após Softmax |
| `y` | label one-hot verdadeira |
| `φ` | função de ativação (Sigmoid, ReLU, Tanh) |
| `η` | taxa de aprendizado (learning rate) |
| `λ` | coeficiente de regularização L2 |
| `B` | tamanho do mini-batch |
| `⊙` | produto de Hadamard (element-wise) |

---

## Forward Pass

Para cada camada `l = 1, …, L`:

```
z(l)  = W(l) * a(l-1) + b(l)        ← pré-ativação
a(l)  = φ(z(l))                      ← ativação (camadas ocultas)
```

Na camada de saída (`l = L`), a ativação é **Softmax**:

```
ŷ = Softmax(z(L))
```

**Implementação:** `MLP::forward()` encadeia `Layer::forward()` em todas as camadas ocultas. A Softmax é aplicada diretamente pelo `MLP` ao `z(L)` da última layer (a ativação da última `Layer` é ignorada).

---

## Funções de Ativação

### Sigmoid

```
σ(x) = 1 / (1 + e^{-x})
σ'(x) = σ(x) · (1 - σ(x))
```

### ReLU

```
f(x) = max(0, x)
f'(x) = 1  se x > 0
        0  caso contrário
```

### Tanh

```
f(x) = tanh(x)
f'(x) = 1 - tanh²(x)
```

### Softmax (Numericamente Estável)

```
softmax(z)_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
```

A subtração de `max(z)` evita overflow numérico sem alterar o resultado (o fator cancela).

**Propriedades:**
- Σ_i softmax(z)_i = 1 (distribuição de probabilidade válida)
- softmax(z)_i ∈ (0, 1) para qualquer z

---

## Função de Custo: Cross Entropy

Para uma amostra com label one-hot `y` e predição `ŷ`:

```
L = −Σ_i y_i · log(ŷ_i)
```

Como `y` é one-hot (exatamente um índice `k` com `y_k = 1`):

```
L = −log(ŷ_k)
```

**Estabilidade numérica:** `log(max(ŷ_i, ε))` com `ε = 1e-12` previne `log(0)`.

### Gradiente Combinado Softmax + Cross Entropy

O gradiente de `L` em relação a `z(L)` (pré-ativação da última camada) é:

```
δ(L) = ∂L/∂z(L) = ŷ − y
```

Esta forma compacta resulta da cadeia de derivação aplicada conjuntamente ao Softmax e à Cross Entropy. Evita materializar o Jacobiano `∂ŷ/∂z` de dimensão `[n × n]`.

**Implementação:** `CrossEntropy::compute_delta()` retorna `ŷ − y`. O `MLP::backward()` passa esse vetor diretamente para `Layer::backward_output_z()`.

---

## Backpropagation

### Gradiente na Camada de Saída

```
δ(L) = ŷ − y
```

### Gradiente em Camadas Ocultas (l < L)

```
δ(l) = (W(l+1)^T · δ(l+1)) ⊙ φ'(z(l))
```

Propaga o erro da camada seguinte (ponderado pelos pesos) e multiplica element-wise pela derivada da ativação no ponto de operação.

### Gradientes dos Parâmetros

```
∂L/∂W(l) = δ(l) · a(l-1)^T     ← produto externo
∂L/∂b(l) = δ(l)
```

**Implementação:** `Layer::backward()` e `Layer::backward_output_z()` acumulam `∂L/∂W` e `∂L/∂b` internamente. A acumulação permite o uso com mini-batches (soma sobre o batch antes de aplicar).

---

## Atualização de Pesos (SGD)

### SGD Puro (batch_size = 1)

Para cada amostra:
```
W(l) ← W(l) − η · ∂L/∂W(l)
b(l) ← b(l) − η · ∂L/∂b(l)
```

### Mini-Batch SGD

Para um batch de `B` amostras:
```
W(l) ← W(l) − (η/B) · Σ_{b=1}^{B} ∂L_b/∂W(l)
b(l) ← b(l) − (η/B) · Σ_{b=1}^{B} ∂L_b/∂b(l)
```

**Implementação:** O `Trainer` acumula gradientes via `mlp.backward()` para cada amostra do batch, depois chama `mlp.step(η/B)` que aplica o gradiente médio e zera os acumuladores.

---

## Inicialização Xavier Uniform

Para uma camada com `fan_in` entradas e `fan_out` neurônios:

```
limit = √(6 / (fan_in + fan_out))
W[i,j] ~ Uniform(−limit, +limit)
b[i]   = 0
```

**Motivação:** Mantém a variância dos sinais (e dos gradientes) aproximadamente constante através das camadas, prevenindo vanishing/exploding gradients na inicialização.

---

## Regularização L2

### Função de Custo Regularizada

```
L_total = L_data + λ · Σ_l Σ_{i,j} W_l(i,j)²
```

Onde `L_data` é a Cross Entropy média sobre as amostras do batch.

### Gradiente L2

```
∂L_total/∂W(l) = ∂L_data/∂W(l) + λ · W(l)
```

### Equivalência com Weight Decay

A atualização SGD com L2 pode ser reescrita como:

```
W(l) ← W(l) − η · (∂L_data/∂W(l) + λ · W(l))
       = W(l) · (1 − η·λ)  −  η · ∂L_data/∂W(l)
```

**Implementação:** O `Trainer` aplica:
1. `mlp.step(η/B)` — gradiente dos dados
2. `apply_weight_decay(mlp, (η/B)·λ)` — multiplica cada peso por `(1 − (η/B)·λ)`

Biases **não** são regularizados (prática padrão — biais afetam apenas translação, não complexidade do modelo).

### Loss Reportada vs. Val Loss

- **Train loss** (em `train_loss_history`): `mean_data_loss + λ · Σ W²` — inclui penalidade
- **Val loss** (em `val_loss_history`): `mean_data_loss` puro — sem penalidade
- **Early stopping** baseia-se em `val_loss` (erro de generalização)

---

## Early Stopping

Após cada época, o `Trainer` avalia `val_loss` no conjunto de validação:

```
se val_loss < best_val_loss − min_delta:
    best_val_loss ← val_loss
    patience_counter ← 0
    salvar snapshot de W e b
senão:
    patience_counter ← patience_counter + 1
    se patience_counter ≥ patience:
        parar treino

restaurar pesos do melhor snapshot
```

**Parâmetros:**
- `patience` — épocas consecutivas sem melhora antes de parar
- `min_delta` — melhora mínima em `val_loss` para contar como progresso

---

## Gradient Checking (Verificação Numérica)

Usado nos testes para validar que `backward()` está correto. A derivada numérica de `f` em `θ` por diferenças centrais:

```
∂f/∂θ ≈ (f(θ + ε) − f(θ − ε)) / (2ε)
```

Com `ε = 1e-4`. O erro relativo entre o gradiente analítico e numérico deve ser `< 1e-3` para confirmar a correção da backpropagation.
