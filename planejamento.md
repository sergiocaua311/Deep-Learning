## Projeto de Deep Learning: Classificação de Sons Ambientais
Objetivo: Treinar e avaliar um modelo de inteligência artificial capaz de classificar 50 categorias de sons ambientais (animais, natureza, sons urbanos) em um prazo de 3 dias.
Dataset: ESC-50 (Environmental Sound Classification)
Ambiente Recomendado: Google Colab (com GPU ativada)

#### Parte 1: Fundamentos Teóricos (O que precisamos saber para a apresentação)
Para defendermos o projeto, não basta mostrar o código; precisamos explicar como a rede neural "ouve" o som. Aqui estão os três pilares teóricos do nosso projeto:

1. O Som como um Vetor (Forma de Onda e Taxa de Amostragem)
Para o computador, o som é um vetor unidimensional (uma sequência longa de números). A Taxa de Amostragem (Sample Rate) define quantos números são capturados por segundo. Se o nosso áudio tem 16.000 Hz (16 kHz), significa que 1 segundo de som é representado por um vetor de 16.000 posições. Redes neurais exigem que todos os áudios de entrada tenham exatamente a mesma taxa de amostragem.

2. A Transformação Matemática (Do Tempo para a Frequência)
Redes neurais modernas têm dificuldade em extrair padrões diretamente do áudio bruto. Por isso, convertemos esse vetor temporal em uma "imagem" através da Transformada de Fourier. Ao aplicarmos a Escala Mel (que distorce as frequências para imitar a audição humana), geramos o Espectrograma Log-Mel. Essa representação em 2D possui o tempo no eixo X, a frequência no eixo Y e a intensidade (volume) representada pelas cores.

3. A Arquitetura do Modelo: Audio Spectrogram Transformer (AST)
Em vez de usar redes convolucionais antigas, utilizaremos o estado da arte: a arquitetura Transformer. O modelo AST pega o nosso espectrograma (a "imagem" do som) e o recorta em vários quadrados pequenos (patches). Ele então analisa esses recortes comparando-os entre si através de um mecanismo de Autoatenção (Self-Attention), aprendendo a associar padrões visuais do espectrograma a sons específicos (como um trovão ou um alarme).

#### Parte 2: Passo a Passo da Implementação (Notebook Python)
O projeto será desenvolvido em um notebook divido nestes 6 blocos principais. Utilizaremos o ecossistema da HuggingFace para abstrair a engenharia complexa e focar nos resultados.

- Passo 1 : Instalação e Importação de Bibliotecas
Ação: Instalar as ferramentas base.
Bibliotecas principais: transformers (para carregar o modelo AST), datasets (para baixar o ESC-50 nativamente), torchaudio e librosa (para manipulação e visualização acústica).
- Passo 2. : Carregamento do Conjunto de Dados (ESC-50)
Ação: Usar a biblioteca datasets para importar o ESC-50.
Detalhe: O conjunto já virá com os 2.000 áudios de 5 segundos mapeados para suas 50 categorias categóricas (ex: 0 = cachorro, 14 = chuva).
- Passo 3: Pré-processamento com o Feature Extractor
Ação: Instanciar o AutoFeatureExtractor específico do modelo AST.
Detalhe: O extrator fará a reamostragem dos áudios para 16 kHz e gerará os Espectrogramas Mel automaticamente. Vamos criar uma função map para processar todo o dataset de uma só vez e transformá-lo nos tensores exigidos pela rede.
- Passo 4: Configuração do Modelo (Transfer Learning)
Ação: Instanciar o modelo ASTForAudioClassification.
Detalhe: Como o modelo pré-treinado conhece milhares de classes do banco de dados do Google, vamos configurá-lo com num_labels=50. A rede descartará sua última camada original e criará uma nova, focada exclusivamente em aprender as 50 classes do nosso projeto (Transfer Learning).
- Passo 5: Treinamento (Fine-Tuning)
Ação: Utilizar a classe Trainer da HuggingFace.
Parâmetros Críticos:
Batch Size (Tamanho do Lote): Manter baixo (4 ou 8) para evitar erros de falta de memória (Out-of-Memory) na GPU do Colab.
Epochs (Épocas): Treinar por 3 a 5 épocas será suficiente e deve demorar menos de 1 hora.
Métricas: Integrar a biblioteca evaluate para calcular e exibir a Acurácia no fim de cada época.
- Passo 6: Inferência e Visualização (O Gran Finale)
Ação: Testar o modelo e exibir os resultados graficamente para a apresentação.
Detalhe:
Passaremos um áudio de teste (ou um áudio gravado por nós) pelo modelo treinado.
Imprimiremos a previsão: "O modelo previu que este som é: [Classe]".
Usaremos librosa.display.specshow para plotar o Espectrograma Mel colorido na tela, provando visualmente à banca que dominamos o processo de transformação do sinal descrito na teoria.