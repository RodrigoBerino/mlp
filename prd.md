# PRD — Product Requirements Document

Projeto: MLP Multiclasse com Softmax
Versão: 2.0

Dataset:
Students' Academic Performance Dataset (Kaggle)

---

# 1. Objetivo

Desenvolver uma implementação genérica e extensível de um Perceptron Multicamadas (MLP) para classificação multiclasse utilizando:

- Softmax na camada final
- Cross Entropy como função de custo
- Backpropagation
- Templates e Functors em C++

---

# 2. Escopo

O sistema deve:

- Ler dados CSV
- Realizar pré-processamento
- Treinar MLP multiclasse
- Avaliar métricas
- Exportar resultados
- Executar testes automatizados
- Integrar CI/CD

---

# 3. Requisitos Funcionais

RF1 — Leitura de CSV  
RF2 — One-hot encoding  
RF3 — Normalização MinMax  
RF4 — Implementação de MLP L camadas  
RF5 — Suporte a ativações genéricas  
RF6 — Implementação de Softmax  
RF7 — Cross Entropy  
RF8 — Backpropagation  
RF9 — Treinamento via SGD  
RF10 — Métricas multiclasse  
RF11 — Exportação de métricas

---

# 4. Requisitos Não Funcionais

RNF1 — Código genérico com templates  
RNF2 — Ativações via functors  
RNF3 — Arquitetura modular  
RNF4 — Testes automatizados  
RNF5 — Cobertura mínima de 80%  
RNF6 — Build via CMake  
RNF7 — CI/CD obrigatório  
RNF8 — Sem dependências externas de ML

---

# 5. Requisitos Arquiteturais

RA1 — Separação por domínio (core, layers, loss, mlp, data)  
RA2 — Separação por skills para execução modular com Claude Code  
RA3 — Testes independentes por módulo  
RA4 — Pipeline CI obrigatório  
RA5 — Documentação técnica em docs/

---

# 6. Requisitos Matemáticos

Baseado na formulação:

z(l) = W(l)a(l-1) + b(l)  
a(l) = φ(z(l))

Saída final:
y_hat = Softmax(z(L))

Backprop:

δ(L) = y_hat - y

δ(l) = (W(l+1)^T δ(l+1)) ⊙ φ'(z(l))

Gradientes:

∂L/∂W(l) = δ(l)a(l-1)^T  
∂L/∂b(l) = δ(l)

Atualização:

W ← W - η∂L/∂W  
b ← b - η∂L/∂b

---

# 7. Métricas

- Accuracy
- Precision macro
- Recall macro
- F1 macro
- Matriz de confusão

---

# 8. Critérios de Aceitação

- Loss converge
- Accuracy superior ao baseline aleatório
- Gradientes validados numericamente
- Testes automatizados passando
- Pipeline CI validado

---

# 9. Skills Envolvidas

## Implementação

- C++20
- Templates avançados
- Functors
- Álgebra linear
- Otimização numérica

## Testes

- Testes unitários
- Testes de integração
- Gradient checking
- Análise de cobertura

## DevOps

- CMake
- GitHub Actions
- Versionamento semântico

---

# 10. Extensões Futuras

- Dropout
- L2 regularization
- Adam optimizer
- Serialização de modelo
- Interface gráfica
