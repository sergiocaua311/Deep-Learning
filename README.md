# Projetos de Aprendizado Profundo (Deep Learning)

Este repositório consolida os trabalhos práticos desenvolvidos durante a disciplina de Aprendizado Profundo da Universidade Federal da Paraíba (UFPB). Os projetos exploram a teoria e a prática por trás das redes neurais, aplicando arquiteturas variadas para resolver problemas complexos envolvendo dados tabulares, visão computacional e processamento de áudio.

## Estrutura do Repositório

O repositório está dividido em dois módulos principais, cada um focando em um domínio específico de dados e arquiteturas.

---

### 1. Práticas com Redes Neurais e Visão Computacional

Este módulo abrange a construção, o treinamento e a avaliação de Redes Perceptron de Múltiplas Camadas (MLP), Redes Neurais Convolucionais (CNN) e Autoencoders. A ênfase é na compreensão do fluxo completo de Machine Learning, desde a análise exploratória até a interpretabilidade do modelo.

**Principais Atividades e Conceitos:**
- **Regressão com MLPs:** 
  - *Aproximação de Funções:* Modelagem de funções matemáticas contínuas analisando as curvas de erro (MAE, MSE, RMSE).
  - *California Housing Dataset:* Previsão de valores imobiliários com base em dados socioeconômicos e geográficos.
- **Classificação de Dados Tabulares:** 
  - *Titanic Dataset:* Tratamento rigoroso de valores ausentes, normalização e engenharia de atributos para prever sobrevivência, validando o aprendizado com matriz de confusão, acurácia, *Precision*, *Recall* e *F1-Score*.
- **Visão Computacional com CNNs e MLPs:** 
  - *Fashion-MNIST:* Comparação arquitetural e de performance entre redes densas tradicionais e redes convolucionais especializadas para classificação de vestuário.
- **Interpretabilidade de Modelos (XAI):** 
  - Inspeção de redes neurais através da extração de filtros da primeira camada convolucional e geração de mapas de ativação (Feature Maps), visualizando qualitativamente as regiões da imagem que mais contribuem para as decisões da rede.
- **Autoencoders:** 
  - *Reconstrução de Imagens:* Implementação de um modelo focado em aprender representações latentes comprimidas das imagens.
  - *Denoising Autoencoders:* Treinamento de uma arquitetura especializada em remover ruídos inseridos artificialmente, restaurando a qualidade das imagens originais.

**Acesse o código e os resultados detalhados:** [Pratica-Redes-Neurais](./Pratica-Redes-Neurais/)

---

### 2. Classificação de Sons Ambientais (Audio Spectrogram Transformer)

Neste projeto, o foco muda para o processamento avançado de dados de áudio. O objetivo é classificar 2.000 amostras sonoras do dataset **ESC-50** em 50 categorias acústicas distintas (animais, sons urbanos, fenômenos naturais, etc.), alcançando **92,64% de F1-Score (Macro)**.

**Principais Atividades e Conceitos:**
- **Fundamentos de Áudio para Deep Learning:** 
  - Compreensão do sinal de áudio como série temporal (taxa de amostragem) e a extração de características pela transformação matemática para o domínio da frequência, gerando **Espectrogramas Log-Mel**. Essa abordagem permite tratar o áudio como uma "imagem" bidimensional (Tempo vs Frequência).
- **Arquitetura State-of-the-Art (ViT/AST):** 
  - Substituição das tradicionais CNNs pelo **Audio Spectrogram Transformer (AST)**. Essa arquitetura divide o espectrograma em pequenos recortes (*patches*) e utiliza complexos mecanismos de Autoatenção (*Self-Attention*) para correlacionar padrões visuais com assinaturas acústicas.
- **Transfer Learning e Fine-Tuning:** 
  - Adaptação de um modelo gigantesco, previamente treinado no vasto *AudioSet* do Google. A camada final original é substituída por uma nova, focada exclusivamente no aprendizado das 50 classes do projeto, o que economiza drásticos recursos computacionais e tempo de treinamento.
- **Ecossistema Hugging Face:** 
  - Utilização da biblioteca `transformers` para instanciar a arquitetura e os *feature extractors*, além de `datasets` para automação de download. Para análise acústica, integrou-se o `librosa` para plotagem de espectrogramas e depuração visual.

**Acesse o código, métricas e o notebook completo:** [Projeto-Sons-Ambientes](./Projeto-Sons-Ambientes/)
