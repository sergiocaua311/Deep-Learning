# Classificação de sons ambientais — ESC-50

Este projeto compara quatro formas de representar áudio para classificação:

| Família | Entrada | Pré-treinamento | Papel no experimento |
|---|---|---|---|
| AST | Log-Mel em patches | AudioSet | baseline de domínio |
| Wav2Vec2 Base | forma de onda | fala auto-supervisionada | encoder temporal |
| Whisper Tiny | Log-Mel | encoder-decoder de fala | somente o encoder é usado |
| CNN compacta | Log-Mel | nenhum | baseline leve |

O notebook original foi preservado. Seu AST atingiu **92,75% de acurácia** e
**92,64% de F1 macro** no fold 5, mas esse fold também escolheu o checkpoint.
Por isso, esse número é mantido como referência histórica e não é misturado com
o novo teste cego.

## Instalação

Requer Python 3.11 e FFmpeg. No macOS:

```bash
brew install ffmpeg
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Os checkpoints públicos são baixados do Hugging Face na primeira execução. As
execuções escrevem em `artifacts/`, que não é versionado.

## Protocolo recomendado

### 1. Testes rápidos

```bash
python -m unittest discover -s tests -v
```

### 2. Triagem

Treina nos folds 1–3, escolhe a época pelo F1 macro do fold 4 e só então avalia
o fold 5:

```bash
python -m esc50 screen
```

Para conferir o fluxo com poucas amostras antes de iniciar os treinos longos:

```bash
python -m esc50 screen --max-train-samples 8 --max-eval-samples 4 --epochs 1 --no-roundtrip
```

Cada run é retomado do checkpoint mais recente. Use `--no-resume` para impedir
a retomada e `--force` apenas quando quiser executar novamente uma run concluída.

### 3. Validação cruzada

Use os dois modelos registrados em `artifacts/screening/selected_models.json` e
as melhores épocas da triagem. Por exemplo:

```bash
python -m esc50 cross-validate \
  --models ast,whisper \
  --model-epochs ast=3,whisper=5
```

Cada fold é avaliado somente depois do treino nos outros quatro. O resumo contém
média e desvio-padrão de F1 macro e acurácia, além de latência e tamanho.

### 4. Robustez e ablação

```bash
python -m esc50 robustness \
  --checkpoint artifacts/cross_validation/ast-cv-fold5-s42/model \
  --test-fold 5 \
  --output-dir artifacts/robustness/ast-fold5

python -m esc50 ablation --model ast --epochs 3
```

A robustez cobre deslocamento de ±0,5 s, ganho de ±6 dB e ruído gaussiano em
20, 10 e 0 dB de SNR, sempre com sementes determinísticas.

### 5. Modelo final e inferência

Depois de escolher o vencedor pela validação cruzada:

```bash
python -m esc50 train-final --model ast --epochs 3

python -m esc50 predict \
  --checkpoint artifacts/final/ast-final-all-folds-s42/model \
  --audio meu_audio.wav --top-k 5

python -m esc50 explain \
  --checkpoint artifacts/final/ast-final-all-folds-s42/model \
  --audio meu_audio.wav \
  --output-dir artifacts/explanations/meu_audio
```

O modelo final usa os cinco folds e serve para inferência; ele não recebe uma
métrica de teste independente. O comando `predict` aceita WAV, MP3 e FLAC.

## Artefatos

Cada execução salva:

- pesos, processador, configuração e model card;
- métricas, duração, parâmetros, tamanho e latência;
- previsões, erros de alta confiança e relatório por classe;
- matriz de confusão e diagrama de confiabilidade;
- diferença máxima dos logits antes/depois de salvar e recarregar.

Consolide várias execuções com:

```bash
python -m esc50 summarize \
  --registry artifacts/screening/registry.csv \
  --output-dir artifacts/report
```

O notebook `notebooks/comparacao_modelos.ipynb` lê esses artefatos e produz as
tabelas e gráficos finais sem repetir treinamento.
