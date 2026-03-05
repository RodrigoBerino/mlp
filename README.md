# MLP Multiclasse com Softmax

Uma implementação de rede neural *fully-connected* (Perceptron Multicamadas) para classificação multiclasse, escrita do zero em C++20.

Nenhuma biblioteca de ML externa é usada — apenas a STL e GoogleTest para testes.

---

## Motivação

Este projeto implementa um MLP completo utilizando:

- **Templates C++20** — álgebra linear e camadas genéricas sobre tipo escalar `T`
- **Functors** — ativações (Sigmoid, ReLU, Tanh, Softmax) como objetos chamáveis
- **Separação por domínio** — cada módulo tem responsabilidade única e testes independentes
- **Paradigma acadêmico** — formulação matemática explícita em cada arquivo

---

## Funcionalidades

| Funcionalidade | Status |
|---|---|
| Álgebra linear (`Vector<T>`, `Matrix<T>`) | ✓ |
| Ativações: Sigmoid, ReLU, Tanh, Softmax | ✓ |
| Camada fully-connected com Xavier init | ✓ |
| Cross Entropy + MSE | ✓ |
| MLP multi-camada com backpropagation | ✓ |
| SGD / Mini-batch com shuffle | ✓ |
| Pipeline de dados (CSV, MinMax, One-hot) | ✓ |
| Métricas multiclasse (F1, precisão, recall) | ✓ |
| Validation split + Early stopping | ✓ |
| L2 Regularization (weight decay) | ✓ |
| CI/CD via GitHub Actions | ✓ |

---

## Dependências

| Dependência | Versão | Uso |
|---|---|---|
| C++ | 20 | Linguagem |
| CMake | ≥ 3.20 | Build system |
| GCC | ≥ 13 | Compilador |
| GoogleTest | 1.14.0 | Framework de testes (via FetchContent) |

Nenhuma outra dependência é necessária. GoogleTest é baixado automaticamente pelo CMake.

---

## Estrutura de Pastas

```
mlp/
├── src/
│   ├── core/               # Vector<T>, Matrix<T>
│   ├── activations/        # Sigmoid, ReLU, Tanh, Softmax
│   ├── layers/             # Layer<T, Activation>
│   ├── loss/               # CrossEntropy, MSE, L2Regularization
│   ├── mlp/                # MLP<T,Act>, Trainer<T,Act>, Dataset<T>
│   ├── data/               # CsvReader, DataPipeline, train_validation_split
│   └── evaluation/         # Metrics<T>: F1, precisão, recall, matriz de confusão
│
├── tests/                  # 305 testes GoogleTest
│   ├── test_vector.cpp
│   ├── test_matrix.cpp
│   ├── test_activations.cpp
│   ├── test_layer.cpp
│   ├── test_loss.cpp
│   ├── test_mlp.cpp
│   ├── test_training.cpp
│   ├── test_data.cpp
│   ├── test_metrics.cpp
│   ├── test_minibatch.cpp
│   ├── test_validation.cpp
│   └── test_regularization.cpp
│
├── docs/                   # Documentação técnica
│   ├── architecture.md
│   ├── mlp_math.md
│   └── training_pipeline.md
│
├── experiments/            # Experimentos versionados
├── .github/workflows/      # CI/CD GitHub Actions
├── train.csv               # Dataset de exemplo
├── CMakeLists.txt
├── plan.md
└── prd.md
```

---

## Como Compilar

```bash
# Clone o repositório
git clone <url-do-repositório>
cd mlp

# Configure e compile
cmake -B build
cmake --build build --parallel

# Alternativa com Ninja (mais rápido)
cmake -B build -G Ninja
cmake --build build --parallel
```

O build produz apenas executáveis de teste — a biblioteca é *header-only* e não gera `.so` ou `.a`.

---

## Como Rodar os Testes

### Todos os testes

```bash
ctest --test-dir build --output-on-failure
```

### Com paralelismo

```bash
ctest --test-dir build --output-on-failure --parallel 4
```

### Por módulo

```bash
# Apenas álgebra linear
ctest --test-dir build --output-on-failure -R "test_vector|test_matrix"

# Apenas ativações
ctest --test-dir build --output-on-failure -R "test_activations"

# Apenas treinamento
ctest --test-dir build --output-on-failure -R "test_training|test_minibatch"

# Apenas validação e early stopping
ctest --test-dir build --output-on-failure -R "test_validation"

# Apenas regularização
ctest --test-dir build --output-on-failure -R "test_regularization"
```

### Resultado esperado

```
305 tests, 0 failures, 0 warnings
```

---

## Quick Start — Exemplo Completo de Treino

```cpp
#include "mlp/mlp.hpp"
#include "mlp/trainer.hpp"
#include "data/dataset.hpp"
#include "activations/sigmoid.hpp"
#include "loss/l2_regularization.hpp"
#include "evaluation/metrics.hpp"

#include <iostream>

int main() {
    // 1. Montar dataset manualmente
    mlp::Dataset<float> train_data, val_data;

    // Cada amostra: Vector<float> de features + Vector<float> one-hot label
    mlp::Vector<float> x(4);   // 4 features
    x[0] = 0.5f; x[1] = 0.3f; x[2] = 0.8f; x[3] = 0.1f;

    mlp::Vector<float> y(3, 0.0f);  // 3 classes
    y[0] = 1.0f;                     // classe 0

    train_data.inputs.push_back(x);
    train_data.labels.push_back(y);
    // ... adicionar mais amostras

    // 2. Criar MLP: 4 entradas → 16 neurônios ocultos → 3 saídas
    mlp::MLP<float, mlp::Sigmoid<float>> model(
        {4, 16, 3},   // arquitetura: {input, hidden..., output}
        42u           // seed para reprodutibilidade
    );

    // 3. Configurar trainer
    mlp::Trainer<float, mlp::Sigmoid<float>> trainer;

    // 4. Treinar com validação e early stopping
    mlp::EarlyStoppingConfig<float> es;
    es.patience  = 10;
    es.min_delta = 1e-4f;

    trainer.train_with_validation(
        model,
        train_data,
        val_data,
        /*epochs=*/   200,
        /*lr=*/       0.01f,
        /*batch_size*/32,
        /*shuffle=*/  true,
        /*seed=*/     42u,
        es,
        /*lambda=*/   0.001f   // L2 regularization
    );

    // 5. Inspecionar histórico
    const auto& tl = trainer.train_loss_history();
    const auto& vl = trainer.val_loss_history();
    const auto& f1 = trainer.val_macro_f1_history();

    for (std::size_t e = 0; e < tl.size(); ++e) {
        std::cout << "epoch " << e
                  << "  train=" << tl[e]
                  << "  val="   << vl[e]
                  << "  f1="    << f1[e]
                  << '\n';
    }

    return 0;
}
```

---

## Como Usar com Dataset CSV

O projeto inclui uma pipeline completa para leitura de CSV com normalização e one-hot encoding:

```cpp
#include "data/data_pipeline.hpp"
#include "mlp/trainer.hpp"
#include "mlp/mlp.hpp"
#include "activations/sigmoid.hpp"

int main() {
    mlp::DataPipeline<float> pipe;

    // Carrega e divide o dataset em train / val / test
    // Parâmetros: path, coluna do label (SIZE_MAX = última), frac_train, frac_val, seed
    auto splits = pipe.load_and_split(
        "train.csv",
        std::numeric_limits<std::size_t>::max(),  // última coluna = label
        0.70f,   // 70% treino
        0.15f,   // 15% validação  → 15% teste implícito
        42u      // seed
    );

    std::cout << "Treino: "    << splits.train.size() << " amostras\n";
    std::cout << "Validação: " << splits.val.size()   << " amostras\n";
    std::cout << "Teste: "     << splits.test.size()  << " amostras\n";
    std::cout << "Features: "  << pipe.num_features() << '\n';
    std::cout << "Classes: "   << pipe.num_classes()  << '\n';

    // Criar MLP com a dimensão correta
    mlp::MLP<float, mlp::Sigmoid<float>> model(
        {pipe.num_features(), 32, 16, pipe.num_classes()},
        42u
    );

    mlp::Trainer<float, mlp::Sigmoid<float>> trainer;

    mlp::EarlyStoppingConfig<float> es;
    es.patience  = 15;
    es.min_delta = 1e-4f;

    trainer.train_with_validation(
        model, splits.train, splits.val,
        300, 0.01f, 32, true, 42u, es, 0.001f
    );

    // Avaliar no test set
    const float test_loss = trainer.evaluate(model, splits.test);
    std::cout << "Test loss: " << test_loss << '\n';

    return 0;
}
```

---

## Formato Esperado do Dataset CSV

O dataset deve ser um arquivo CSV com:

- **Primeira linha:** cabeçalho (nomes das colunas)
- **Colunas de features:** valores numéricos (inteiros ou decimais)
- **Coluna de label:** string com o nome da classe (ex: `"Low"`, `"Medium"`, `"High"`)
- **Posição do label:** por padrão, a última coluna

Exemplo (`train.csv`):
```
gender,raisedhands,VisITedResources,AnnouncementsView,Discussion,Class
M,15,16,2,20,L
M,20,20,3,25,M
F,70,80,65,19,H
```

As classes serão ordenadas alfabeticamente e mapeadas para índices one-hot.

> **Dataset incluído:** `train.csv` — Students' Academic Performance Dataset (Kaggle).
> 3 classes: `H` (High), `L` (Low), `M` (Medium). Features numéricas de engajamento estudantil.

---

## Cobertura de Testes

| Módulo | Arquivo de Teste | Testes |
|---|---|---|
| Infraestrutura | `test_placeholder.cpp` | 1 |
| Vector | `test_vector.cpp` | ~20 |
| Matrix | `test_matrix.cpp` | ~20 |
| Ativações | `test_activations.cpp` | ~32 |
| Layer | `test_layer.cpp` | ~27 |
| Loss | `test_loss.cpp` | ~20 |
| MLP | `test_mlp.cpp` | ~22 |
| Trainer/SGD | `test_training.cpp` | ~25 |
| Data Pipeline | `test_data.cpp` | ~15 |
| Métricas | `test_metrics.cpp` | ~20 |
| Mini-batch | `test_minibatch.cpp` | ~18 |
| Validação/ES | `test_validation.cpp` | 16 |
| Regularização | `test_regularization.cpp` | 18 |
| **Total** | | **305** |

---

## CI/CD

O pipeline de CI (GitHub Actions) executa automaticamente em todo push ou pull request para `main`/`develop`:

1. **Install** — cmake, ninja, g++ no Ubuntu 24.04
2. **Configure** — `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`
3. **Build** — `cmake --build build --parallel`
4. **Test** — `ctest --output-on-failure --parallel 4`

O pipeline falha se qualquer teste falhar ou se houver warnings de compilação.

---

## Documentação Técnica

| Arquivo | Conteúdo |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Arquitetura e descrição dos módulos |
| [`docs/mlp_math.md`](docs/mlp_math.md) | Fundamentação matemática completa |
| [`docs/training_pipeline.md`](docs/training_pipeline.md) | Fluxo de dados ponta-a-ponta |

---

## Licença

MIT License — veja [LICENSE](LICENSE).
