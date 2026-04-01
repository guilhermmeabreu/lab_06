"""Laboratório 6: BPE + WordPiece com tokenizer do BERT multilingual.

Este script:
1) Implementa get_stats(vocab) para contar pares de símbolos.
2) Valida que ('e', 's') possui frequência 9 no vocab inicial.
3) Implementa merge_vocab(pair, v_in).
4) Executa 5 iterações de merge, imprimindo o par e o vocab atualizado.
5) Usa AutoTokenizer.from_pretrained('bert-base-multilingual-cased').
6) Tokeniza uma frase de exemplo em português.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import DefaultDict

from transformers import AutoTokenizer


def get_stats(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    """Conta a frequência de cada par adjacente de símbolos no vocabulário.

    Args:
        vocab: Dicionário no formato {'s i m b o l o s </w>': frequencia}.

    Returns:
        Dicionário {(simbolo1, simbolo2): frequencia_total}.
    """
    pairs: DefaultDict[tuple[str, str], int] = defaultdict(int)

    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq

    return dict(pairs)


def merge_vocab(pair: tuple[str, str], v_in: dict[str, int]) -> dict[str, int]:
    """Funde um par de símbolos em todo o vocabulário.

    Ex.: pair=('e','s') transforma "n e w e s t </w>" em "n e w es t </w>".
    """
    v_out: dict[str, int] = {}
    bigram = re.escape(" ".join(pair))
    # casa somente o par inteiro entre fronteiras de token
    pattern = re.compile(rf"(?<!\S){bigram}(?!\S)")

    for word, freq in v_in.items():
        w_out = pattern.sub("".join(pair), word)
        v_out[w_out] = freq

    return v_out


def print_vocab(vocab: dict[str, int]) -> None:
    """Imprime o vocabulário de forma estável para facilitar leitura."""
    for k, v in sorted(vocab.items()):
        print(f"  {k}: {v}")


def run_bpe_demo() -> dict[str, int]:
    """Executa a demonstração de 5 merges com o vocab obrigatório."""
    vocab: dict[str, int] = {
        "l o w </w>": 5,
        "l o w e r </w>": 2,
        "n e w e s t </w>": 6,
        "w i d e s t </w>": 3,
    }

    print("Vocab inicial:")
    print_vocab(vocab)

    # validação obrigatória
    stats = get_stats(vocab)
    es_count = stats.get(("e", "s"), 0)
    print(f"\nValidação ('e', 's'): {es_count}")
    assert es_count == 9, "Falha: ('e', 's') deveria ser 9."

    print("\nIniciando 5 iterações de merge:")
    for i in range(1, 6):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)

        print(f"\nIteração {i} -> par fundido: {best} (freq={pairs[best]})")
        print("Vocab atualizado:")
        print_vocab(vocab)

    return vocab


def run_wordpiece_demo() -> None:
    """Demonstra tokenização WordPiece com BERT multilingual."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    sentence = (
        "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
    )
    tokens = tokenizer.tokenize(sentence)

    print("\nTokenização com bert-base-multilingual-cased:")
    print(f"Frase: {sentence}")
    print("Tokens:")
    print(tokens)


if __name__ == "__main__":
    run_bpe_demo()
    run_wordpiece_demo()
