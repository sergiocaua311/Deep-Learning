# Portfólio de Deep Learning

Este repositório documenta os projetos desenvolvidos durante a disciplina de Aprendizado Profundo da Universidade Federal da Paraíba (UFPB). O objetivo é aplicar conceitos fundamentais e avançados de redes neurais na resolução de problemas envolvendo dados tabulares, imagens e áudio.

## Projetos

### 1. Práticas com Redes Neurais e Visão Computacional

Este projeto engloba a implementação e análise de diferentes arquiteturas de redes neurais para tarefas variadas de aprendizado de máquina. 

**Tópicos Abordados:**
- **Redes Perceptron de Múltiplas Camadas (MLP) para Regressão:** Modelagem para aproximação de funções contínuas e previsão de valores imobiliários com o *California Housing Dataset*.
- **MLP para Classificação:** Análise exploratória, tratamento de dados e predição de sobrevivência utilizando a base do *Titanic*.
- **Redes Neurais Convolucionais (CNNs) e MLPs:** Treinamento, avaliação e comparação de desempenho de arquiteturas para classificação de imagens no dataset *Fashion-MNIST*.
- **Interpretabilidade em Deep Learning:** Extração e visualização qualitativa de filtros convolucionais e mapas de ativação para investigar como a rede toma decisões.
- **Autoencoders:** Construção de modelos para compressão/reconstrução de imagens do *Fashion-MNIST* e experimentação com autoencoders voltados para remoção de ruídos (*denoising*).

Para explorar o código e os resultados detalhados, acesse o diretório do projeto: [Pratica-Redes-Neurais](./Pratica-Redes-Neurais/)

---

### 2. Classificação de Sons Ambientais

O segundo projeto consiste no processamento de áudio focado em classificar os 2.000 sons do dataset *ESC-50* em 50 categorias de sons ambientais distintos, alcançando 92,75% de acurácia.

**Tópicos Abordados:**
- **Processamento de Áudio:** Fundamentos do áudio como vetor temporal (ondas e taxas de amostragem) e extração de características pela transformação em Espectrogramas Log-Mel.
- **Audio Spectrogram Transformer (AST):** Uso da arquitetura de ponta *Transformer*, que processa os espectrogramas como se fossem imagens divididas em pequenas partes (*patches*) com autoatenção.
- **Transfer Learning (Fine-Tuning):** Adaptação de um modelo AST previamente treinado no extenso *AudioSet* para a tarefa direcionada de 50 classes.
- **Treinamento e Inferência com Hugging Face:** Uso de ferramentas da Hugging Face (`transformers`, `datasets`) para baixar os dados, processá-los automaticamente com *feature extractors* e realizar treinamento em GPU, além da visualização espectral dos resultados com `librosa`.

Para explorar o código e os resultados detalhados, acesse o diretório do projeto: [Projeto-Sons-Ambientes](./Projeto-Sons-Ambientes/)
