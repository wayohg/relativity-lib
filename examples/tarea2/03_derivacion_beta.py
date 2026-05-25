"""Ejercicio 3: derivación simbólica de beta desde energía cinética."""

from __future__ import annotations

import sympy as sp

from _tarea2_common import print_header


def main() -> None:
    print_header('Ejercicio 3 - Derivación de beta')

    beta, E0, K = sp.symbols('beta E0 K', positive=True)
    gamma = 1 / sp.sqrt(1 - beta**2)

    print('Energía total: E = gamma E0')
    print('Energía cinética: K = E - E0 = (gamma - 1) E0')
    print('Entonces gamma = (E0 + K)/E0')

    gamma_expr = (E0 + K) / E0
    equation = sp.Eq(1 / sp.sqrt(1 - beta**2), gamma_expr)
    solution = sp.solve(equation, beta)[0]

    print('\nSolución positiva para beta:')
    print(sp.simplify(solution))
    print('\nForma esperada:')
    expected = sp.sqrt(1 - (E0 / (E0 + K))**2)
    print(expected)
    print('\nVerificación simbólica solution - expected =')
    print(sp.simplify(solution - expected))


if __name__ == '__main__':
    main()
