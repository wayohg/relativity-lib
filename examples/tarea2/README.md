# Scripts para Tarea 2 / ADA 2: Dinámica relativista

Estos scripts resuelven y simulan los ejercicios de la Tarea 2.
Están pensados para colocarse dentro de tu proyecto en:

```text
examples/tarea2/
```

## Uso

Desde la raíz del proyecto:

```bash
python examples/tarea2/run_all_tarea2.py
```

O ejecuta un ejercicio individual:

```bash
python examples/tarea2/02_tevatr_protones.py
python examples/tarea2/06_decaimiento_kaon.py
```

Las figuras se guardan en:

```text
examples/tarea2/output/
```

## Archivos

```text
01_fuerza_perpendicular.py
02_tevatr_protones.py
03_derivacion_beta.py
04_agujero_negro_energia.py
05_aniquilacion_electron_positron.py
06_decaimiento_kaon.py
run_all_tarea2.py
RESULTADOS_TAREA2.md
```

## Notas importantes

- En ejercicios de partículas se usan energías en MeV y unidades naturales con `c = 1`.
- En el ejercicio 4 se usan unidades SI.
- En el ejercicio 6, las velocidades de los piones en el laboratorio dependen del ángulo de emisión en el marco de reposo del kaón. El script calcula explícitamente el caso colineal y además grafica la dependencia angular.
