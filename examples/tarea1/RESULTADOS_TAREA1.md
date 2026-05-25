# Resultados rápidos de Tarea 1

Estos resultados son los que imprimen los scripts. Las simulaciones generan figuras en `examples/tarea1/output/`.

## Ejercicio 1

Los eventos son simultáneos en un marco `K'` que se mueve respecto a `K` con:

```text
β = v/c = -0.5
v = -0.5c
```

La separación espacial en ese marco es:

```text
Δx' = 0.866025 a
```

## Ejercicio 2

Para la recepción de la señal en `(x,y,z)=(3,5,10) m`:

```text
r = sqrt(134) = 11.575836903 m
t = 3.861283562641e-08 s
```

Para `K'` moviéndose con `β=0.8` sobre `+x`:

```text
x' = -10.434449204 m
y' = 5.000000000 m
z' = 10.000000000 m
t' = 5.101216223608e-08 s
```

La verificación da:

```text
|r'|/t' = c
```

## Ejercicio 3

La velocidad mínima de la nave es:

```text
β = 1/sqrt(2) = 0.707106781
v = 0.707106781 c
```

El tiempo medido desde la Tierra es:

```text
56.568542495 años
```

## Ejercicio 4

Para `β=0.94`:

```text
γ = 2.931051909
Distancia medida por la pelota = 12214.045030 km
Tiempo en Tierra = 0.127038 s
Tiempo en la pelota = 0.043342 s
```

## Ejercicio 5

La velocidad relativa protón-antiprotón es:

```text
0.975609756 c
```

## Ejercicio 6

```text
s² = -0.75 a²
```

Por tanto la separación es tipo espacio. El marco donde son simultáneos tiene:

```text
β = 0.5
v = 0.5c
Δx' = 0.866025 a
```

## Ejercicio 7

Para `f=300 Hz` y `β=0.4`:

```text
Observador moviéndose en la misma dirección que la luz: f' = 196.396101 Hz, redshift
Observador viniendo hacia la luz: f' = 458.257569 Hz, blueshift
```

## Ejercicio 8

Si la partícula va a `0.95c` y tú a `0.60c`, ambos sobre `x`:

```text
u' = 0.813953488 c
```

Para un fotón, no existe marco inercial que se mueva a `c`; no se puede transformar al “marco del fotón”. Todo observador inercial mide la luz con velocidad `c`.

## Ejercicio 9

ARA llega a Berius en la Tierra en:

```text
37.5 años
```

En el marco de tu nave, moviéndose a `0.6c`:

```text
t' = 24.375 años
x' = 9.375 ly
```

## Ejercicio 10

La rapidez se suma linealmente:

```text
v = c tanh(u)
w = c tanh(U)
w' = c tanh(u+U)
```

Para velocidades sucesivas de `0.9c`:

```text
β_N = tanh(N artanh(0.9))
β_N = (1 - (1/19)^N)/(1 + (1/19)^N)
β_N ≈ 1 - 2/19^N, para N grande
```

Valores:

```text
N=1  β_N=0.900000000000
N=2  β_N=0.994475138122
N=3  β_N=0.999708454810
N=5  β_N=0.999999192278
N=10 β_N≈1.000000000000
```
