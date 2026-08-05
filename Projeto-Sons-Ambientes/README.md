# Classificação de Sons Ambientais com Deep Learning

Projeto da disciplina **Aprendizado Profundo**, orientado pelo professor [Tiago Maritan Ugulino de Araujo](http://www.ufpb.br/docente/tiagomaritan), na Universidade Federal da Paraíba.

## Equipe

- Matheus Bruno da Silva Oliveira
- Micael Oliveira de Lima Toscano
- Sergio Caua dos Santos

## Objetivo

Realizar a classificação dos 2.000 áudios do conjunto ESC-50 em 50 categorias de sons ambientais. A solução utiliza o **Audio Spectrogram Transformer (AST)** pré-treinado no AudioSet e ajustado com os folds 1 a 4 do ESC-50. O fold 5 foi utilizado para acompanhamento e avaliação do experimento.

## Resultado registrado

| Métrica | Fold 5 |
|---|---:|
| Acurácia | 92,75% |
| F1 macro | 92,64% |
| Loss | 0,2183 |
| Tempo de treinamento | 64,8 min |

Esses valores correspondem à terceira época. Como o fold 5 também orientou a escolha do melhor checkpoint, o resultado é apresentado como avaliação do experimento, e não como estimativa obtida em um teste cego independente.

## Estrutura

```text
Projeto-Sons-Ambientes/
├── data/
│   └── esc50.csv
├── models/
│   └── ast_esc50/               
├── notebooks/
│   └── classificacao_sons_ambientais_esc50.ipynb
├── outputs/
│   └── ast_esc50/                 
├── .gitignore
├── README.md
└── requirements.txt
```

## Execução

Recomenda-se Python 3.11. No macOS, instale o FFmpeg e depois as dependências do projeto:

```bash
brew install ffmpeg
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Abra [o notebook principal](notebooks/classificacao_sons_ambientais_esc50.ipynb) e execute as células na ordem. O conjunto ESC-50 é obtido pelo Hugging Face Datasets e, na primeira execução, requer acesso à internet.

O notebook inicia com `EXECUTAR_TREINAMENTO = False`, utilizando o checkpoint local quando ele está disponível. Altere a opção para `True` somente se desejar repetir o treinamento completo; essa etapa é demorada e substitui os artefatos locais.
