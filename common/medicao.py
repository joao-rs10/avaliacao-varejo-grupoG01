"""Medição de tempo (time.perf_counter) e memória de pico (tracemalloc)."""
from __future__ import annotations

import time
import tracemalloc


def medir(func, *args, **kwargs):
    """Executa `func` duas vezes: uma só com perf_counter (tempo limpo, sem a
    sobrecarga do tracemalloc) e outra com tracemalloc (pico de memória).

    As funções medidas devem ser determinísticas por seed (todas são).
    Retorna (resultado, tempo_s, pico_mem_mib).
    """
    t0 = time.perf_counter()
    resultado = func(*args, **kwargs)
    tempo = time.perf_counter() - t0

    tracemalloc.start()
    func(*args, **kwargs)
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return resultado, tempo, pico / (1024 * 1024)
