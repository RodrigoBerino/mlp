# Arquitetura do Projeto

## Visão Geral

O projeto é uma biblioteca *header-only* em C++20 organizada em domínios independentes. Cada módulo tem uma responsabilidade única e pode ser testado isoladamente.

```
src/
├── core/           # Álgebra linear primitiva
├── activations/    # Funções de ativação (functors)
├── layers/         # Camada fully-connected
├── loss/           # Funções de custo + regularização
├── mlp/            # Rede neural + loop de treinamento
├── data/           # Leitura de CSV + pré-processamento
└── evaluation/     # Métricas multiclasse
```

A biblioteca `mlp_core` é um alvo `INTERFACE` no CMake — acumula todos os `include_directories` sem produzir binários.

---

## Módulos

### `core` — Álgebra Linear

**Arquivos:** `vector.hpp`, `matrix.hpp`

```cpp
template<typename T> class Vector;
template<typename T> class Matrix;
```

`Vector<T>` e `Matrix<T>` são contêineres genéricos sobre tipo escalar `T` (tipicamente `float` ou `double`).

**Vector:**
- Armazenamento: `std::vector<T>` interno
- Operações: `+`, `-`, `*` (escalar e produto interno), `[]` com bounds checking
- Sem alocação dinâmica explícita — gerenciado pela STL

**Matrix:**
- Armazenamento: row-major em `std::vector<T>`
- Acesso: `operator()(row, col)` com validação
- Operações: multiplicação matriz-vetor (`*`), multiplicação matriz-matriz, transposta, Hadamard
- `broadcast_add(bias)` para adicionar vetor de bias a cada coluna

Ambas as classes usam `[[nodiscard]]` nos métodos de acesso e `std::invalid_argument` / `std::out_of_range` para erros de tamanho.

---

### `activations` — Funções de Ativação

**Arquivos:** `sigmoid.hpp`, `relu.hpp`, `tanh.hpp`, `softmax.hpp`

Todas as ativações escalares seguem a interface de functor:

```cpp
template<typename T>
struct Sigmoid {
    T operator()(T x) const;      // f(x)
    T derivative(T x) const;      // f'(x)
};
```

| Ativação | f(x) | f'(x) |
|---|---|---|
| `Sigmoid<T>` | 1 / (1 + e⁻ˣ) | σ(x) · (1 − σ(x)) |
| `ReLU<T>` | max(0, x) | 1 se x > 0, senão 0 |
| `Tanh<T>` | tanh(x) | 1 − tanh²(x) |

**Softmax** é especializado para `Vector<T>`:

```cpp
template<typename T>
struct Softmax {
    Vector<T> operator()(const Vector<T>& z) const;
};
```

Usa a fórmula numericamente estável `exp(z_i − max(z))` para evitar overflow.

> **Nota de design:** A Softmax da camada de saída é aplicada diretamente pelo `MLP`, não pela `Layer`. O gradiente combinado Softmax+CrossEntropy é `ŷ − y` (elimina a necessidade de materializar o Jacobiano).

---

### `layers` — Camada Fully-Connected

**Arquivo:** `layer.hpp`

```cpp
template<typename T, typename Activation>
class Layer;
```

Cada `Layer` encapsula:
- `Matrix<T> W_` — pesos `[fan_out × fan_in]`
- `Vector<T> b_` — biases `[fan_out]`
- `Vector<T> z_` — cache pré-ativação (para backward)
- `Vector<T> a_` — cache pós-ativação
- `Matrix<T> grad_W_`, `Vector<T> grad_b_` — gradientes acumulados

**Inicialização Xavier uniform:**
```
limit = √(6 / (fan_in + fan_out))
W[i,j] ~ Uniform(−limit, +limit)
b[i]   = 0
```

**Forward:**
```
z = W * a_prev + b
a = φ(z)         ← element-wise
```

**Backward (camada oculta):**
```
δ = (W_next^T * δ_next) ⊙ φ'(z)
∂L/∂W += δ * a_prev^T    ← acumulado
∂L/∂b += δ               ← acumulado
```

**Backward (camada de saída com Softmax+CE):**
```
δ = ŷ − y    ← gradiente combinado, passado diretamente
```

`update(lr)` aplica `W -= lr * grad_W`, zera gradientes.

---

### `loss` — Funções de Custo e Regularização

**Arquivos:** `cross_entropy.hpp`, `mse.hpp`, `l2_regularization.hpp`

**CrossEntropy** (primária):
```cpp
template<typename T>
struct CrossEntropy {
    T       compute_loss(y_hat, y)   → T
    Vector<T> compute_delta(y_hat, y) → ŷ − y
};
```
Numericamente estável via `max(ŷ_i, ε)` com `ε = 1e-12`.

**MSE** (secundária):
```cpp
template<typename T>
struct MSE {
    T       compute_loss(y_hat, y)
    Vector<T> compute_gradient(y_hat, y)
};
```

**L2 Regularization** (funções livres template):
```cpp
T    compute_l2_penalty(mlp, lambda)     → λ · Σ W²
void apply_weight_decay(mlp, eta_lambda)  → W ← W · (1 − η·λ)
T    compute_weight_norm_sq(mlp)         → Σ W²
```

---

### `mlp` — Rede Neural e Trainer

**Arquivos:** `mlp.hpp`, `trainer.hpp`

#### `MLP<T, Activation>`

O modelo principal. Recebe um vetor de tamanhos de camada e instancia layers internamente.

```cpp
MLP<float, Sigmoid<float>> model({4, 16, 8, 3}, /*seed=*/42u);
```

Internamente:
- `layers_[0]` → input→hidden₁ (com `Activation`)
- `layers_[1..L-2]` → hidden→hidden (com `Activation`)
- `layers_[L-1]` → hidden→output (**ativação bypassed**)
- Softmax aplicada diretamente a `z(L)` para produzir `y_hat_`

API pública:
```cpp
const Vector<T>& forward(const Vector<T>& x);
void backward(const Vector<T>& y_true);
void step(T learning_rate);
std::size_t num_layers() const;
Layer<T,Act>& layer(std::size_t i);         // mutable — para snapshot/restore
const Layer<T,Act>& layer(std::size_t i) const;
T compute_loss(const Vector<T>& y_true) const;
```

#### `Dataset<T>` e `EarlyStoppingConfig<T>`

`Dataset<T>` é uma struct simples:
```cpp
template<typename T>
struct Dataset {
    std::vector<Vector<T>> inputs;
    std::vector<Vector<T>> labels;   // one-hot
    std::size_t size() const;
    void validate() const;           // lança se sizes não batem
};
```

`EarlyStoppingConfig<T>`:
```cpp
template<typename T>
struct EarlyStoppingConfig {
    std::size_t patience  = 5;
    T           min_delta = T{1e-4};
};
```

#### `Trainer<T, Activation>`

Gerencia o loop de treinamento. Suporta dois modos:

**`train()`** — SGD / mini-batch simples:
```cpp
trainer.train(model, data, epochs, lr,
              batch_size=1, shuffle=false, seed=42, lambda=0.0f);
```

**`train_with_validation()`** — com validação, early stopping e restauração do melhor modelo:
```cpp
trainer.train_with_validation(
    model, train_data, val_data, epochs, lr,
    batch_size=1, shuffle=false, seed=42,
    EarlyStoppingConfig{patience=5, min_delta=1e-4},
    lambda=0.0f
);
```

Histórico disponível após treino:
```cpp
trainer.epoch_losses()          // train() → loss por época
trainer.epoch_accuracies()      // train() → accuracy por época
trainer.train_loss_history()    // train_with_validation() → inclui penalidade L2
trainer.val_loss_history()      // train_with_validation() → data loss puro
trainer.val_macro_f1_history()  // train_with_validation() → F1 macro por época
```

---

### `data` — Pipeline de Dados

**Arquivos:** `csv_reader.hpp`, `data_pipeline.hpp`, `dataset.hpp`

**`read_csv(path)`** — função inline que lê qualquer CSV:
```cpp
CsvData result = mlp::read_csv("train.csv");
// result.header: vector<string>
// result.rows: vector<vector<string>>
```

**`DataPipeline<T>`** — pipeline completa:
1. Lê CSV via `read_csv`
2. Extrai features numéricas e labels string
3. Coleta classes únicas → ordena alfabeticamente
4. Embaralha índices com seed fixa
5. Divide em train/val/test
6. Ajusta MinMax **apenas no train** (sem data leakage)
7. Aplica normalização nos três splits
8. Codifica labels em one-hot

```cpp
DataPipeline<float> pipe;
auto splits = pipe.load_and_split("train.csv",
    /*label_col=*/ SIZE_MAX,   // última coluna
    /*train_frac=*/ 0.70f,
    /*val_frac=*/   0.15f,
    /*seed=*/       42u
);
// splits.train, splits.val, splits.test : Dataset<float>
pipe.num_features();  // dimensão de cada input
pipe.num_classes();   // número de classes
```

**`train_validation_split()`** — divide um `Dataset<T>` existente:
```cpp
auto [train, val] = mlp::train_validation_split(dataset, 0.8f, 42u);
```

---

### `evaluation` — Métricas Multiclasse

**Arquivo:** `metrics.hpp`

`Metrics<T>` é uma classe com apenas métodos estáticos (sem estado):

```cpp
// Matriz de confusão: cm(i,j) = amostras com classe real i preditas como j
Matrix<size_t> cm = Metrics<float>::compute_confusion_matrix(y_true, y_pred);

float acc  = Metrics<float>::compute_accuracy(cm);
auto  prec = Metrics<float>::compute_precision_per_class(cm);
auto  rec  = Metrics<float>::compute_recall_per_class(cm);
auto  f1   = Metrics<float>::compute_f1_per_class(cm);
float mf1  = Metrics<float>::compute_macro_f1(cm);
```

Divisão por zero protegida: denominador zero retorna 0 para a métrica daquela classe.

---

## Diagrama de Dependências entre Módulos

```
evaluation/metrics.hpp
    └── core/matrix.hpp
        └── core/vector.hpp

loss/cross_entropy.hpp, loss/mse.hpp
    └── core/vector.hpp

loss/l2_regularization.hpp
    └── mlp/mlp.hpp

activations/*.hpp
    └── core/vector.hpp (apenas softmax.hpp)

layers/layer.hpp
    └── core/matrix.hpp
    └── core/vector.hpp

mlp/mlp.hpp
    └── layers/layer.hpp
    └── activations/softmax.hpp
    └── loss/cross_entropy.hpp

mlp/trainer.hpp
    └── mlp/mlp.hpp
    └── loss/cross_entropy.hpp
    └── loss/l2_regularization.hpp
    └── evaluation/metrics.hpp

data/data_pipeline.hpp
    └── data/csv_reader.hpp
    └── mlp/trainer.hpp      (para Dataset<T>)

data/dataset.hpp
    └── mlp/trainer.hpp      (para Dataset<T>)
```

---

## Decisões de Arquitetura

**1. Header-only**
Todo o código vive em `.hpp`. Facilita redistribuição e elimina problemas de linkagem para uma biblioteca de templates.

**2. Compile-time polymorphism**
`Layer<T, Activation>` e `MLP<T, Activation>` são parametrizados pela função de ativação — sem virtual functions, zero overhead de runtime.

**3. Softmax fora da Layer**
A última camada não aplica ativação; a Softmax é aplicada pelo `MLP` diretamente ao `z(L)`. Isso permite usar o gradiente combinado `δ = ŷ − y` sem materializar o Jacobiano `∂Softmax/∂z`.

**4. Gradientes acumulados**
`Layer` acumula `∂L/∂W` e `∂L/∂b` internamente. `mlp.step(lr)` aplica a média (`lr/B` por sample num batch de B) e zera.

**5. Sem data leakage no MinMax**
`DataPipeline` ajusta min/max **apenas** nos dados de treino e aplica a transformação nos demais splits.

**6. Snapshot de pesos para Early Stopping**
`Trainer` salva uma cópia de `(W, b)` por camada sempre que `val_loss` melhora. Ao final, restaura o melhor checkpoint — sem modificar `Layer` ou `MLP`.
