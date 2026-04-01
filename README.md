# Laboratório 6 — BPE e WordPiece em Python

Este repositório contém uma implementação completa e pronta para execução dos requisitos do laboratório:

- implementação de `get_stats(vocab)`
- validação de que o par `('e', 's')` tem frequência **9**
- implementação de `merge_vocab(pair, v_in)`
- execução de **5 iterações** de merge, imprimindo o par fundido e o vocabulário atualizado
- tokenização com `AutoTokenizer.from_pretrained("bert-base-multilingual-cased")`

## Estrutura

```text
.
├── .gitignore
├── README.md
├── requirements.txt
└── src
    └── lab6.py
```

## Como executar

1. (Opcional) Crie e ative um ambiente virtual.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute o laboratório:

```bash
python src/lab6.py
```

## Vocab inicial (obrigatório)

O código usa exatamente este vocabulário inicial:

```python
{
  'l o w </w>': 5,
  'l o w e r </w>': 2,
  'n e w e s t </w>': 6,
  'w i d e s t </w>': 3
}
```

## O que significa `##` em tokens WordPiece?

No WordPiece (como no BERT), o prefixo `##` indica que aquele token é uma **continuação** de uma palavra, e não o início.

Exemplo conceitual:

- `in`
- `##constitu`
- `##cional`
- `##mente`

Isso ajuda muito com palavras desconhecidas (out-of-vocabulary), porque o modelo não precisa conhecer a palavra completa no vocabulário. Em vez disso, ele a decompõe em subpalavras conhecidas, permitindo representar melhor palavras raras, longas ou recém-criadas.

## Observações

- O script está comentado e organizado em funções para facilitar leitura e reutilização.
- A parte do tokenizer pode baixar arquivos do modelo na primeira execução (internet necessária nesse momento).
- Este projeto contou com apoio parcial de inteligência artificial na estruturação e revisão do código, sendo todo o conteúdo analisado e compreendido pelo autor.
