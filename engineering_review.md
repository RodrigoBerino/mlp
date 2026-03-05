# Engineering Review — MLP Project

**Data:** 2026-03-05
**Revisão:** 1.0
**Escopo:** Código-fonte completo em `src/` após Fase 12 + correções de auditoria
**Compilador de referência:** GCC 13.3, C++20, flags `-Wall -Wextra -Wpedantic`

---

## Sumário Executivo

O projeto implementa uma biblioteca *header-only* de rede neural MLP multiclasse em C++20
com templates, sem dependências externas de ML. Após auditoria completa e aplicação de
três correções críticas, o projeto não apresenta memory leaks, bugs de comportamento
indefinido em fluxos normais de uso, ou instabilidades numéricas relevantes. Os issues
residuais documentados neste relatório são de baixo risco e não afetam a correção dos
resultados de treinamento.

---

## 1. Análise de Arquitetura

### 1.1 Estrutura de Módulos

```
src/
├── core/           vector.hpp, matrix.hpp           — álgebra linear primitiva
├── activations/    sigmoid.hpp, relu.hpp,            — functors de ativação
│                   tanh.hpp, softmax.hpp
├── layers/         layer.hpp                         — camada fully-connected
├── loss/           cross_entropy.hpp, mse.hpp,       — funções de custo
│                   l2_regularization.hpp             — regularização
├── mlp/            mlp.hpp, trainer.hpp              — modelo + loop de treino
├── data/           csv_reader.hpp, data_pipeline.hpp — ingestão de dados
│                   dataset.hpp                       — utilitário de split
└── evaluation/     metrics.hpp                       — métricas multiclasse
```

### 1.2 Grafo de Dependências

```
core/vector.hpp             (folha)
core/matrix.hpp           → core/vector.hpp
activations/relu.hpp        (folha)
activations/sigmoid.hpp   → <cmath>
activations/tanh.hpp      → <cmath>
activations/softmax.hpp   → core/vector.hpp
layers/layer.hpp          → core/matrix.hpp · core/vector.hpp
loss/cross_entropy.hpp    → core/vector.hpp
loss/mse.hpp              → core/vector.hpp
mlp/mlp.hpp               → layers/layer.hpp · activations/softmax.hpp
                             loss/cross_entropy.hpp
evaluation/metrics.hpp    → core/matrix.hpp · core/vector.hpp
loss/l2_regularization.hpp → mlp/mlp.hpp
mlp/trainer.hpp           → mlp/mlp.hpp · loss/l2_regularization.hpp
                             evaluation/metrics.hpp
data/csv_reader.hpp         (folha)
data/data_pipeline.hpp    → data/csv_reader.hpp · mlp/trainer.hpp
data/dataset.hpp          → mlp/trainer.hpp
```

**Dependências circulares:** nenhuma. O grafo é um DAG puro.
**`#pragma once`:** presente em todos os 16 headers — inclusões múltiplas impossíveis.

### 1.3 Decisões de Design

**Header-only com compile-time polymorphism**
`Layer<T, Activation>` e `MLP<T, Activation>` são parametrizados pela função de ativação
via template. Sem virtual functions — zero overhead de runtime. Trade-off: tempos de
compilação maiores para cada combinação `(T, Activation)` instanciada.

**Softmax fora da Layer**
A última camada não aplica ativação interna; `MLP::forward()` aplica Softmax diretamente
ao `z(L)`. Isso permite usar o gradiente combinado `δ(L) = ŷ − y`, evitando materializar
o Jacobiano `∂Softmax/∂z` de dimensão `[n × n]`. Correto e eficiente.

**Gradientes acumulados**
`Layer` acumula `∂L/∂W` e `∂L/∂b` internamente. `mlp.step(lr/B)` aplica o gradiente
médio do batch e zera os acumuladores. Permite mini-batch SGD sem estrutura externa.

**Dataset<T> definido em `trainer.hpp`**
Convenção de projeto: `Dataset<T>` é definido onde é mais usado. `data_pipeline.hpp` e
`dataset.hpp` incluem `trainer.hpp` para obter a definição — acoplamento intencional,
não circular.

### 1.4 Inconsistências Identificadas

**[INC-1] `docs/architecture.md` menciona método `broadcast_add` inexistente**
A documentação descreve `Matrix::broadcast_add(bias)` para adição de vetor de bias.
O método não existe em `matrix.hpp`. A adição é feita via composição em `Layer::forward()`:
```cpp
z_ = W_ * input + b_;   // Matrix*Vector → Vector, depois Vector+Vector
```
O código está correto; a documentação está errada.

**[INC-2] `Dataset::validate()` não verifica tamanhos internos**
```cpp
void validate() const {
    if (inputs.size() != labels.size()) { throw ... }
    // Não verifica: inputs[i].size() consistente entre amostras
}
```
Um `Dataset` construído manualmente com vetores de tamanhos diferentes passa por
`validate()`. O erro só aparece dentro de `Layer::forward()` com mensagem sem contexto
de qual amostra causou o problema.

**[INC-3] `MLP::forward()` retorna referência para estado interno**
```cpp
const Vector<T>& forward(const Vector<T>& input);   // retorna &y_hat_
```
A referência é invalidada na próxima chamada a `forward()`. No código atual do `Trainer`
o padrão de uso é sempre seguro (referência consumida antes da próxima chamada). É um
footgun na API pública para consumidores externos da biblioteca.

---

## 2. Análise de Memória

### 2.1 Inventário de Alocações

| Recurso | Tipo | Gerenciamento | Status |
|---|---|---|---|
| `Vector<T>::data_` | `std::vector<T>` | RAII automático | ✅ |
| `Matrix<T>::data_` | `std::vector<T>` | RAII automático | ✅ |
| `Layer` (W, b, z, a, grad_W, grad_b, last_input_) | Membros por valor | RAII automático | ✅ |
| `MLP::layers_` | `std::vector<Layer<T,Act>>` | RAII automático | ✅ |
| `MLP::y_hat_` | `Vector<T>` membro | RAII automático | ✅ |
| `Trainer::WeightSnapshot` | `std::vector<pair<Matrix,Vector>>` | RAII automático | ✅ |
| `DataPipeline::min_`, `max_`, `class_names_` | `std::vector` | RAII automático | ✅ |
| `std::mt19937 rng` (todos os sites) | Stack-allocated | Destruído no escopo | ✅ |

**`new`/`delete` explícitos no codebase:** nenhum.
**Memory leaks detectados:** nenhum.

### 2.2 Custo de Memória Notável

`Trainer::train_with_validation()` armazena um `WeightSnapshot` — cópia profunda de
todos os pesos — a cada época em que `val_loss` melhora:

```cpp
best_snapshot = take_snapshot(mlp);   // copia W e b de todas as camadas
```

Para a arquitetura de referência `{16, 32, 16, 3}`:

```
Camada 0: 16×32 + 32 = 544 floats
Camada 1: 32×16 + 16 = 528 floats
Camada 2: 16×3  +  3 =  51 floats
Total: 1123 floats × 4 bytes ≈ 4.4 KB por snapshot
```

Para redes maiores como `{64, 256, 128, 10}`:

```
Camada 0: 64×256  + 256 = 16640 floats
Camada 1: 256×128 + 128 = 32896 floats
Camada 2: 128×10  +  10 =  1290 floats
Total: 50826 floats × 4 bytes ≈ 200 KB por snapshot
```

Apenas um snapshot é mantido por vez (substituído quando val_loss melhora). Consumo
de memória é O(parâmetros), não cresce com o número de épocas.

---

## 3. Estabilidade Numérica

### 3.1 Softmax

**Implementação:**
```cpp
T max_val = *max_element(z);           // shift para estabilidade
out[i] = std::exp(z[i] - max_val);    // z[i] - max_val ≤ 0 por construção
sum += out[i];
out[i] /= sum;
```

**Análise por caso:**

| Caso | Comportamento | Resultado |
|---|---|---|
| Entrada normal | `z[i] - max_val ∈ (-∞, 0]`, pelo menos um `= 0` | `sum ≥ exp(0) = 1 > 0` — nunca divide por zero ✅ |
| Entradas muito negativas (underflow) | `exp(x) → 0` para `x < -104` (float) | Classe recebe prob ≈ 0, matematicamente correto ✅ |
| Entradas idênticas | `exp(0) = 1` para todos, `sum = n` | Distribuição uniforme `1/n` ✅ |
| Entrada com um elemento muito maior | Elemento dominante → prob ≈ 1, demais → 0 | Correto ✅ |
| Vetor vazio | `throw std::invalid_argument` | Detectado ✅ |

**Veredito: Numericamente estável. `sum ≥ 1` sempre, divisão por zero impossível.**

### 3.2 Cross Entropy

**Implementação:**
```cpp
static constexpr T kEpsilon = T{1e-12};
if (y[i] > T{0}) {
    const T safe_p = y_hat[i] > kEpsilon ? y_hat[i] : kEpsilon;
    loss -= y[i] * std::log(safe_p);
}
```

**Análise:**

| Caso | Comportamento | Resultado |
|---|---|---|
| `y_hat[i] ∈ (0, 1]` normal | `log(y_hat[i]) ∈ (-∞, 0]`, finito | Loss ∈ [0, ∞) ✅ |
| `y_hat[i] → 0` (extremo) | Clamped para `kEpsilon`, `log(1e-12) ≈ -27.6` | Loss ≤ 27.6 — finito ✅ |
| `y_hat[i] = 1` (predição perfeita) | `log(1) = 0` | Loss = 0 ✅ |
| `y` one-hot | Loop processa apenas `y[k] = 1` | Sem acumulação desnecessária ✅ |

**Nota sobre `kEpsilon` com `float`:**
`kEpsilon = T{1e-12}` armazenado como `float` vira `≈ 9.99e-13f`, abaixo do epsilon de
float (`≈ 1.19e-7`). Na prática, Softmax nunca produz saídas abaixo de `1e-7` para
configurações razoáveis — o clamp raramente é ativado. Impacto prático: zero.

**Veredito: Numericamente estável. Sem risco de log(0) ou NaN.**

### 3.3 Sigmoid

**Implementação:**
```cpp
return T{1} / (T{1} + std::exp(-x));
```

**Análise para `T = float` (threshold: `|x| ≈ 88.7`):**

| Caso | Intermediário | Resultado final | Correto? |
|---|---|---|---|
| `x > 88.7` | `exp(-x) → 0` (denormal) | `1 / 1 = 1.0f` | ✅ σ(+∞) = 1 |
| `x ∈ [-88.7, 88.7]` | Nenhum overflow | Resultado preciso | ✅ |
| `x < -88.7` | `exp(-x) → inf` (overflow IEEE 754) | `1 / (1+inf) = 0.0f` | ✅ σ(-∞) = 0 |
| `x = NaN` | Propagado | `NaN` | Detectável ✅ |

O overflow intermediário (`inf`) ocorre na CPU por uma instrução FPU e é não-observável.
O resultado final está sempre em `[0, 1]`. Durante o backward: `σ'(x) = σ(x)(1-σ(x))`,
com `σ = 0` ou `σ = 1` nos extremos o gradiente é 0 — vanishing esperado e correto.

**Veredito: Resultado sempre em [0, 1]. Overflow intermediário é IEEE 754 well-defined
sem consequência para o resultado ou o gradiente.**

### 3.4 Weight Decay — Proteção Adicionada

**Antes da correção:** `factor = 1 - eta_lambda` podia ser negativo para
`eta_lambda ≥ 1`, invertendo silenciosamente os sinais de todos os pesos.

**Após a correção (NUM-1):**
```cpp
if (eta_lambda >= T{1}) {
    throw std::invalid_argument(
        "apply_weight_decay: eta_lambda must be < 1 to avoid weight sign "
        "inversion (got ...). Reduce learning_rate or lambda so their "
        "product per batch < 1.");
}
```

`eta_lambda = learning_rate / batch_size * lambda`. Para configurações típicas
(`lr=0.01, batch=32, lambda=0.001`): `eta_lambda = 3.1e-7` — muito longe do limite.
O limite `eta_lambda < 1` equivale a `lr * lambda < batch_size`, restrição fraca que
qualquer configuração razoável satisfaz.

---

## 4. Determinismo

### 4.1 Fontes de Aleatoriedade e Controle

| Fonte | Arquivo | Seed | Garantia |
|---|---|---|---|
| Xavier init por camada | `layer.hpp:218` | `base_seed + layer_index` | Determinístico ✅ |
| Shuffle em `train()` | `trainer.hpp:126` | `shuffle_seed` | Determinístico ✅ |
| Shuffle em `train_with_validation()` | `trainer.hpp:228` | `shuffle_seed` | Determinístico ✅ |
| Shuffle em `load_and_split()` | `data_pipeline.hpp:128` | `seed` | Determinístico ✅ |
| Shuffle em `train_validation_split()` | `dataset.hpp:50` | `seed` | Determinístico ✅ |

### 4.2 Sequência de Shuffles Entre Épocas

O `std::mt19937` é instanciado **uma vez** antes do loop de épocas e seu estado
progride continuamente:

```cpp
std::mt19937 rng(shuffle_seed);          // instanciado uma vez
for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
    std::shuffle(idx.begin(), idx.end(), rng);   // estado progride
}
```

Isso garante duas propriedades:
1. Épocas diferentes recebem permutações diferentes (sem repetição do padrão)
2. A sequência completa de todas as épocas é reproduzível com a mesma seed

### 4.3 Portabilidade de Sementes

`std::mt19937` é especificado pelo padrão C++ como Mersenne Twister 19937 — produz
a mesma sequência de inteiros em qualquer implementação conforme.

`std::uniform_real_distribution<T>`, entretanto, **não é especificado** pelo padrão
entre compiladores diferentes. A mesma seed com GCC 13 e MSVC pode produzir pesos
iniciais diferentes. Para este projeto (CI com GCC 13.3 / Ubuntu 24.04), o comportamento
é completamente determinístico.

### 4.4 Ressalva: Overflow de Seed

```cpp
seed + static_cast<std::uint32_t>(i)   // layer i recebe seed derivada
```

Para `seed = UINT32_MAX` e redes com 2+ camadas, a adição faz wrap-around (`UINT32_MAX
+ 1 = 0`). Duas camadas diferentes receberiam seeds idênticas, produzindo pesos iniciais
correlacionados. Overflow unsigned é bem-definido em C++, mas semanticamente indesejado.
Afeta apenas o caso extremo `seed = 0xFFFFFFFF`, improvável em uso real.

---

## 5. Issues Residuais

Os issues abaixo foram identificados na auditoria e **não foram corrigidos** — são de
baixo risco ou envolvem mudanças de API.

### 5.1 Issues de API Design

---

**[API-1] `MLP::forward()` retorna referência para estado interno**

- **Arquivo:** `src/mlp/mlp.hpp:76`
- **Risco:** Baixo — apenas para consumidores externos que armazenam a referência
- **Descrição:** `forward()` retorna `const Vector<T>&` apontando para `y_hat_` interno.
  Chamadas subsequentes a `forward()` invalidam a referência. No `Trainer` atual o
  padrão de uso é seguro (referência consumida antes da próxima chamada).
- **Mitigação sugerida:** Retornar por valor, ou documentar explicitamente o lifetime.

---

**[API-2] `Dataset::validate()` não verifica consistência dimensional interna**

- **Arquivo:** `src/mlp/trainer.hpp:59`
- **Risco:** Baixo — afeta apenas Datasets construídos manualmente fora de `DataPipeline`
- **Descrição:** `validate()` verifica apenas que `inputs.size() == labels.size()`. Não
  verifica que todos os vetores de input têm o mesmo tamanho, nem que não são vazios.
  Datasets mal formados falham com mensagem genérica em `Layer::forward()`.
- **Mitigação sugerida:** Adicionar checagem de `inputs[0].size()` consistente.

---

**[API-3] `Trainer::evaluate()` e `compute_accuracy()` modificam estado do MLP**

- **Arquivo:** `src/mlp/trainer.hpp:301, 318`
- **Risco:** Muito baixo — apenas para uso via `mlp.output()` após avaliação
- **Descrição:** Ambos recebem `MLP<T,Activation>&` (não-const) porque `mlp.forward()`
  é não-const. Após `train_with_validation()`, `restore_snapshot()` restaura W e b, mas
  `y_hat_` interno do MLP contém o resultado do último sample de validação — não do
  melhor snapshot restaurado.
- **Mitigação sugerida:** Documentar que `mlp.forward(x)` deve ser chamado antes de
  usar `mlp.output()` após o treino.

---

### 5.2 Issues de Robustez de Dados

---

**[DATA-1] `std::stod` sem contexto de diagnóstico**

- **Arquivo:** `src/data/data_pipeline.hpp:109`
- **Risco:** Baixo — afeta apenas CSVs malformados
- **Descrição:** Se uma célula contém texto não-numérico, `std::stod` lança
  `std::invalid_argument` sem indicar linha ou coluna do CSV.
- **Mitigação sugerida:** Envolver em try/catch com mensagem contextualizada.

---

**[DATA-2] `n_test = 0` silencioso para frações pequenas**

- **Arquivo:** `src/data/data_pipeline.hpp:136`
- **Risco:** Muito baixo — requer frações de validação intencionalmente grandes
- **Descrição:** `n_test = n - n_train - n_val`. Com arredondamento de ponto flutuante,
  pode resultar em `n_test = 0` (conjunto de teste vazio). Nenhum aviso é emitido.
- **Mitigação sugerida:** Adicionar `if (n_test == 0) throw ...` com mensagem informativa.

---

### 5.3 Issues de Documentação

---

**[DOC-1] `docs/architecture.md` descreve método inexistente**

- **Arquivo:** `docs/architecture.md` — seção `core`
- **Risco:** Nenhum para o código; confuso para leitores
- **Descrição:** A documentação menciona `broadcast_add(bias)` como método de `Matrix`.
  O método não existe. A funcionalidade existe via composição em `Layer::forward()`.
- **Correção:** Remover `broadcast_add` da documentação ou descrever o mecanismo real.

---

### 5.4 Issues de Qualidade de Código

---

**[CODE-1] `argmax` implementado em dois lugares com semânticas diferentes**

| Local | Visibilidade | Vetor vazio |
|---|---|---|
| `Trainer::argmax` — `trainer.hpp:389` | `public static` | Lança `std::invalid_argument` |
| `Metrics::argmax` — `metrics.hpp:174` | `private static` | Retorna `0` silenciosamente |

O `Metrics::argmax` nunca recebe vetor vazio em uso normal (a checagem de `y_true.empty()`
em `compute_confusion_matrix` o protege). Mas a divergência de comportamento é um risco
latente.

---

**[CODE-2] Validação dupla do conjunto de validação por época**

- **Arquivo:** `src/mlp/trainer.hpp:273-276`
- **Impacto:** Performance — não correção
- **Descrição:** Por época em `train_with_validation()`, o conjunto de validação é
  percorrido duas vezes com forward completo:
  1. `evaluate(mlp, val_data)` → computa val_loss
  2. `compute_val_macro_f1(mlp, val_data)` → computa macro F1

  Com 1000 amostras de validação e 300 épocas: 600.000 forwards desnecessários.
  As predições poderiam ser computadas uma vez e reutilizadas para ambas as métricas.

---

**[CODE-3] Loop de mini-batch duplicado entre `train()` e `train_with_validation()`**

- **Arquivo:** `src/mlp/trainer.hpp:136-162` e `trainer.hpp:243-262`
- **Impacto:** Manutenibilidade — alterações no loop precisam ser feitas em dois lugares.

---

### 5.5 Issues de Numerics (Baixo Risco)

---

**[NUM-2] Sigmoid com overflow intermediário para `|x| > 88.7` (float)**

- **Arquivo:** `src/activations/sigmoid.hpp:17`
- **Risco:** Nenhum — resultado final correto, overflow é IEEE 754 well-defined
- **Descrição:** `exp(-x)` resulta em `inf` para `x < -88.7f`. `1 / (1 + inf) = 0.0f`
  matematicamente correto. Formulação alternativa sem overflow intermediário:
  `0.5 + 0.5 * tanh(x / 2)`.

---

**[NUM-3] `kEpsilon = 1e-12` impreciso para `T = float`**

- **Arquivo:** `src/loss/cross_entropy.hpp:28`
- **Risco:** Nenhum — Softmax nunca produz saídas abaixo do epsilon de float em uso normal
- **Descrição:** `T{1e-12}` convertido para `float` vira `≈ 9.99e-13f`, abaixo de
  `std::numeric_limits<float>::epsilon()`. Valor type-safe seria
  `std::numeric_limits<T>::epsilon()`.

---

## 6. Correções Aplicadas Nesta Revisão

As seguintes correções foram aplicadas ao código antes da publicação deste relatório:

| ID | Arquivo | Problema | Correção |
|---|---|---|---|
| BUG-1 | `data_pipeline.hpp:104` | Acesso UB via `std::vector::operator[]` em linha CSV curta | Adicionada validação de `row.size() == n_cols` com mensagem contendo número da linha |
| BUG-3 | `csv_reader.hpp:33` | `\r` de CRLF Windows não removido, causando falha em `std::stod` | `line.pop_back()` se `line.back() == '\r'` antes do filtro de linha vazia |
| NUM-1 | `l2_regularization.hpp:65` | `eta_lambda ≥ 1` invertia sinais dos pesos silenciosamente | Guard explícito com `throw std::invalid_argument` e mensagem de diagnóstico |

---

## 7. Nota de Portabilidade

O projeto compila e passa todos os 305 testes com:

- **GCC 13.3** / Ubuntu 24.04 — ambiente de referência (CI)
- **C++20** obrigatório — uso de `std::iota`, `std::shuffle`, structured bindings

Comportamento de `std::uniform_real_distribution<T>` não é portável entre compiladores
— pesos iniciais podem diferir entre GCC e MSVC/Clang com a mesma seed. O comportamento
de treinamento (convergência, acurácia final) não é afetado, apenas a reprodutibilidade
exata entre compiladores distintos.

---

*Relatório gerado com base em análise estática completa de `src/` (16 headers, ~1400 LOC)
e execução local de `ctest` (305/305 testes passando, 0 warnings com `-Wall -Wextra -Wpedantic`).*
