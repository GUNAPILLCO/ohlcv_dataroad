# Hardware del proyecto

## 1. Propósito

Este documento registra el hardware disponible para el proyecto `ohlcv_dataroad`.

Su objetivo es servir como referencia técnica para decidir cómo ejecutar cargas de trabajo del proyecto, incluyendo análisis estadístico, procesamiento de datos, simulaciones, optimización, machine learning y tareas que puedan beneficiarse de paralelización por CPU o aceleración por GPU.

Este documento describe únicamente la capacidad de cómputo disponible. No define decisiones estadísticas, metodológicas ni de modelado.

---

## 2. Equipo

- **Notebook:** Lenovo Legion Pro 7 16IAX10H
- **Sistema operativo:** Windows
- **Pantalla:** 2560 × 1600, 240 Hz

---

## 3. CPU

- **Procesador:** Intel Core Ultra 9 275HX
- **Núcleos físicos:** 24
- **Procesadores lógicos:** 24
- **Frecuencia base:** 2,70 GHz
- **Frecuencia máxima observada:** aproximadamente 4,74 GHz

### Uso recomendado

La CPU debe ser la opción por defecto para tareas como:

- procesamiento y transformación de datos;
- análisis estadístico;
- estimaciones independientes;
- simulaciones;
- bootstrap;
- validaciones;
- generación de reportes;
- ejecución de pipelines;
- cargas que no tengan una implementación GPU claramente ventajosa.

Cuando existan múltiples trabajos independientes, se puede utilizar paralelización por CPU.

Como configuración inicial de referencia:

- **máximo aproximado de workers:** 20;
- **núcleos reservados para el sistema:** aproximadamente 4.

Este valor es una referencia operativa y puede ajustarse según consumo de memoria, temperatura, librerías utilizadas y características de cada tarea.

Debe evitarse el **oversubscription**, es decir, lanzar más procesos o hilos de los que el hardware puede ejecutar eficientemente.

Si cada proceso utiliza internamente librerías multihilo, debe reducirse el número de procesos paralelos.

---

## 4. GPU

- **GPU:** NVIDIA GeForce RTX 5070 Ti Laptop GPU
- **VRAM dedicada:** 12 GB
- **Memoria GPU compartida disponible:** aproximadamente 17,9 GB
- **Memoria GPU total reportada por Windows:** aproximadamente 29,9 GB

### Uso recomendado

La GPU debe utilizarse cuando la carga de trabajo tenga una implementación compatible y exista una ventaja real frente a CPU.

Ejemplos:

- entrenamiento de redes neuronales;
- inferencia con modelos acelerados por CUDA;
- operaciones matriciales o tensoriales de gran tamaño;
- procesamiento masivo por lotes;
- algoritmos de machine learning con soporte GPU;
- simulaciones o cálculos numéricos diseñados para paralelismo GPU.

La disponibilidad de GPU no implica que todas las tareas deban ejecutarse en ella.

Antes de trasladar una carga a GPU debe verificarse que:

1. la librería utilizada tenga soporte adecuado;
2. el tamaño del problema justifique la transferencia de datos;
3. el uso de GPU reduzca realmente el tiempo total;
4. el resultado siga siendo reproducible.

---

## 5. Memoria RAM

- **Fabricante reportado:** Samsung
- **Capacidad instalada:** pendiente de verificación explícita.

La capacidad total de RAM deberá añadirse cuando se confirme directamente desde Windows u otra fuente confiable del sistema.

---

## 6. Almacenamiento

- **Unidad principal:** WD PC SN800S

### Uso recomendado

Para el proyecto:

- utilizar el SSD local para datasets, archivos temporales y resultados intermedios;
- evitar copias innecesarias de archivos grandes;
- conservar únicamente artefactos reproducibles o necesarios;
- mantener los resultados bajo la estructura definida del repositorio.

---

## 7. Política general de uso de hardware

Para cualquier etapa del proyecto, la estrategia de cómputo debe seguir estas reglas:

1. Usar primero el algoritmo correcto y más simple para la tarea.
2. No aumentar complejidad computacional sin necesidad.
3. Utilizar paralelización por CPU cuando existan trabajos independientes.
4. Evitar paralelización anidada y oversubscription.
5. Utilizar GPU cuando exista una ventaja técnica comprobable.
6. No modificar la metodología de un análisis únicamente para aprovechar el hardware.
7. Mantener reproducibilidad entre ejecuciones.
8. Registrar cualquier configuración especial de CPU, GPU, workers, seeds o librerías cuando pueda afectar los resultados.
9. Controlar uso de memoria y temperatura en ejecuciones prolongadas.
10. Priorizar estabilidad y reproducibilidad frente a utilizar el 100 % del hardware de forma permanente.

---

## 8. Configuración de referencia

| Componente | Especificación |
|---|---|
| Notebook | Lenovo Legion Pro 7 16IAX10H |
| CPU | Intel Core Ultra 9 275HX |
| Núcleos físicos | 24 |
| Procesadores lógicos | 24 |
| Frecuencia base CPU | 2,70 GHz |
| Frecuencia máxima observada | ~4,74 GHz |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| VRAM dedicada | 12 GB |
| Memoria GPU compartida | ~17,9 GB |
| SSD | WD PC SN800S |
| Pantalla | 2560 × 1600 @ 240 Hz |
| RAM | Pendiente de verificación |

---

## 9. Mantenimiento del documento

Actualizar este archivo cuando:

- cambie el equipo principal del proyecto;
- cambie CPU, GPU, RAM o almacenamiento;
- se confirme la capacidad total de RAM;
- una verificación posterior corrija algún dato técnico;
- cambie de forma relevante la política de ejecución del proyecto.

No incluir en este documento:

- números de serie;
- identificadores personales;
- credenciales;
- claves;
- información de cuentas;
- cualquier dato sensible que no sea necesario para ejecutar el proyecto.
