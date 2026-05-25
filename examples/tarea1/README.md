# Tarea 1: Cinemática relativista

Scripts para resolver y simular los ejercicios de **Tarea 1: Cinemática relativista**.

Estos archivos están pensados para colocarse dentro de tu proyecto, en:

```text
examples/tarea1/
```

## Requisitos

Usan solamente:

```text
numpy
matplotlib
```

Si ya instalaste los requerimientos de tu proyecto, no deberías necesitar nada más.

## Ejecución

Desde la raíz del proyecto:

```bash
python examples/tarea1/run_all_tarea1.py
```

O ejecuta un ejercicio individual:

```bash
python examples/tarea1/03_viaje_astronautas.py
```

Las imágenes se guardan en:

```text
examples/tarea1/output/
```

## Contenido

```text
01_eventos_simultaneos.py
02_senal_luminosa.py
03_viaje_astronautas.py
04_pelota_golf_satelite.py
05_proton_antiproton.py
06_intervalo_spacelike.py
07_doppler.py
08_perseguir_particula_foton.py
09_mision_berius.py
10_rapidez_estrellas.py
run_all_tarea1.py
_tarea1_common.py
```

## Nota de unidades

En la mayoría de ejercicios se usan unidades naturales `c=1`. Para el ejercicio 2 y 4, donde aparecen metros, kilómetros y segundos, se usa:

```python
C_SI = 299_792_458.0  # m/s
```

