# Deep-Learning — Lista de Exercícios

**Universidade Federal da Paraíba**
Centro de Informática — Departamento de Informática
**Disciplina:** Aprendizado Profundo · **Período:** 2026.1
**Professor:** Tiago Maritan

**Data de Entrega:** 07/07/2026

## Alunos:

| Aluno | Matrícula |
| --- | --- |
| Matheus Bruno da Silva Oliveira | 20230021519 |
| Micael Oliveira de Lima Toscano | 20230021537 |
| Sérgio Cauã dos Santos | 20230103033 |

---

## Questão 1 — MLP para Regressão

Implemente uma Rede Perceptron de Múltiplas Camadas (MLP) para resolver os problemas a seguir.

### 1a) Aproximação de função

Utilize a rede neural para aproximar a seguinte função:

$$f(x) = 10x^5 + 5x^4 + 2x^3 - 0.5x^2 + 3x + 2, \quad \text{onde } 0 \le x \le 5$$

Para aproximar essa função:

- gere conjuntos de treinamento e teste;
- utilize `x` como entrada e `f(x)` como saída desejada;
- treine a rede neural utilizando apenas o conjunto de treinamento;
- avalie o desempenho da rede durante o treinamento utilizando o conjunto de teste.

**Apresente:**

- gráfico da função real e da função aproximada;
- curva de erro de treinamento;
- curva de erro de validação;
- métricas MAE, MSE e RMSE;
- gráfico comparando as curvas real e predita.

### 1b) California Housing Dataset

Utilize a base de dados **California Housing Dataset**, disponível na biblioteca Scikit-Learn, e utilize a rede (MLP) para prever o valor médio de imóveis a partir de características socioeconômicas e geográficas das regiões analisadas.

**Apresente:**

- análise exploratória dos dados;
- tratamento e normalização dos atributos;
- arquitetura da rede utilizada;
- curvas de treinamento e validação;
- métricas MAE, MSE e RMSE;
- gráfico comparando valores reais e preditos.

---

## Questão 2 — MLP para Classificação (Titanic)

Implemente uma Rede Perceptron de Múltiplas Camadas (MLP) para resolver um problema de classificação utilizando a base **Titanic**, disponível em: <https://www.kaggle.com/c/titanic/data>

O objetivo é prever se um passageiro sobreviveu ou não ao naufrágio a partir de informações como sexo, idade, classe socioeconômica e demais atributos disponíveis.

Além do treinamento da rede neural, realize uma análise completa da base de dados e apresente:

- análise exploratória dos dados;
- identificação de valores ausentes;
- tratamento de dados faltantes;
- análise de atributos irrelevantes ou redundantes;
- transformação de atributos categóricos;
- normalização ou padronização dos atributos numéricos;
- análise da distribuição das classes;
- técnicas de balanceamento, caso necessário.

**Após o treinamento, apresente:**

- arquitetura da rede utilizada;
- curva de erro de treinamento e validação;
- matriz de confusão;
- métricas accuracy, precision, recall e f1-score.

---

## Questão 3 — MLP e CNN (Fashion-MNIST)

Utilize a base **Fashion-MNIST**, disponível em: <https://www.kaggle.com/datasets/zalando-research/fashionmnist>, e implemente dois classificadores distintos:

- **a)** uma Rede Perceptron de Múltiplas Camadas (MLP);
- **b)** uma Rede Neural Convolucional (CNN).

**Apresente:**

- arquitetura utilizada em cada modelo;
- curva de erro de treinamento e validação;
- matriz de confusão;
- acurácia final no conjunto de teste.

---

## Questão 4 — Interpretabilidade da CNN

Utilize a CNN desenvolvida na Questão 3 e investigue as representações aprendidas pela rede ao longo do treinamento.

### 4a) Filtros da primeira camada convolucional

Extraia e visualize os filtros aprendidos pela primeira camada convolucional da rede, apresentando uma visualização dos filtros aprendidos e, se possível, faça uma descrição qualitativa dos padrões observados.

### 4b) Mapas de ativação

Selecione pelo menos 5 imagens do conjunto de teste e visualize os mapas de ativação produzidos por uma camada convolucional intermediária da rede.

**Apresente:**

- imagem original;
- mapas de ativação gerados pela rede;
- análise qualitativa das regiões da imagem que mais contribuíram para a ativação dos neurônios.

### 4c) Acertos e erros de classificação

Selecione exemplos corretamente classificados e exemplos classificados incorretamente pela CNN. Para cada caso, apresente a imagem original, a classe verdadeira, a classe predita e a probabilidade associada à predição (ou score equivalente).

---

## Questão 5 — Autoencoders (Fashion-MNIST)

Utilize novamente o dataset **Fashion-MNIST** para realizar as atividades descritas a seguir.

### 5a) Autoencoder de reconstrução

Implemente e treine um autoencoder capaz de reconstruir as imagens do conjunto de dados. O modelo pode ser construído utilizando apenas camadas densas (fully connected), ou camadas convolucionais e deconvolucionais.

Treine **pelo menos três versões do modelo** utilizando diferentes tamanhos para o espaço latente.

**Após o treinamento, apresente:**

- arquitetura utilizada;
- selecione 10 imagens do conjunto de teste e compare os erros de reconstrução e a qualidade das imagens reconstruídas;
- apresente visualmente as imagens originais e reconstruídas.

### 5b) Autoencoder para remoção de ruído (denoising)

Adicione ruído aleatório às imagens de entrada e treine um novo autoencoder com o objetivo de remover esse ruído. Após o treinamento, utilize as mesmas 10 imagens selecionadas anteriormente e apresente as imagens ruidosas e as reconstruções produzidas pelo modelo.

---
