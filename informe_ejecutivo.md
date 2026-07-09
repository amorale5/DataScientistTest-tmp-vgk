# Informe ejecutivo — Score de probabilidad de default
### Andina Crédito · Gerencia de Riesgo

---

## 1. Resumen Ejecutivo

Andina Crédito aprueba hoy sus créditos de consumo con reglas simples, y la mora se ha convertido en un problema relevante para el negocio. Se desarrolló el **primer modelo de probabilidad de default** de la empresa, usando un enfoque de machine learning entrenado sobre 45.300 solicitudes históricas, para estimar la probabilidad de que cada solicitud caiga en mora de 90+ días dentro de los 12 meses posteriores al desembolso.

Se evaluaron **seis iteraciones de modelamiento** ("rondas"), cada una probando una hipótesis distinta sobre los datos o el entrenamiento, comparando en cada una tres familias de modelos (regresión logística, LightGBM y XGBoost). La selección no se basó solo en métricas estadísticas (AUC, Brier score), sino en el **impacto económico** definido en el desafío: cuánta plata gana o pierde la empresa al aprobar o rechazar cada solicitud.

El modelo elegido (Ronda 4, regresión logística) logra un AUC de 0,835 sobre datos que nunca vio al entrenar. Aplicando la política de aprobación por umbral de probabilidad (diferenciada por plazo del crédito), genera **$406.006.500** de ganancia validada — **5,3 veces más que aprobar el 100% de las solicitudes** (práctica implícita actual, que genera solo $77.239.300 en el mismo grupo). Aplicado a las 12.000 solicitudes nuevas de test.csv, la ganancia esperada es de **$1.234.680.251**.

Se recomienda implementar este modelo junto con la política de aprobación por umbral, y establecer un monitoreo mensual de la tasa de default real vs. la predicha para detectar necesidad de recalibración.

---

## 2. Problema de negocio

Andina Crédito aprueba hoy sus créditos de consumo con reglas simples, y la mora se ha vuelto un problema. Este proyecto construye el **primer modelo de probabilidad de default** de la empresa: para cada solicitud, estima qué tan probable es que el crédito caiga en mora de 90+ días dentro de los 12 meses posteriores al desembolso. El objetivo final no es solo "tener un modelo", sino traducirlo en una **política de aprobación concreta** que la Gerencia de Riesgo pueda usar para decidir a quién prestarle, y cuantificar cuánta plata le ahorra a la empresa frente a la práctica actual.

---

## 3. Datos

Se recibieron dos archivos desde el data warehouse de la empresa: **45.300 solicitudes históricas** de créditos, desembolsadas entre enero 2024 y febrero 2025, junto **con el resultado observado** (si cayeron en default o no); y **12.000 solicitudes nuevas**, entre febrero y junio 2025, **sin resultado conocido** — son las que hay que scorear con el modelo final.

Cada solicitud trae la siguiente información:

| Grupo | Campos |
|---|---|
| Identificación y fecha | `id_solicitud`, `fecha_solicitud`, `dia_semana_solicitud` |
| Perfil del solicitante | `edad`, `tipo_empleo`, `ingreso_declarado`, `antiguedad_laboral_meses`, `region`, `antiguedad_cliente_meses` |
| Historial crediticio (buró) | `score_buro`, `deuda_sistema`, `num_creditos_vigentes`, `peor_morosidad_12m`, `num_consultas_buro_3m`, `uso_linea_credito_pct` |
| Comportamiento reciente | `num_contactos_ult_trimestre` |
| Condiciones del crédito solicitado | `canal`, `monto_solicitado`, `plazo_meses`, `tasa_interes_anual` |
| Resultado (solo en train) | `default_12m` — variable objetivo |

**Herramientas y forma de trabajo**: clonamos el repositorio que la empresa envió por correo y lo descargamos en un ambiente de trabajo local. Sobre ese repositorio, trabajamos con **Claude Code**, un agente de inteligencia artificial para ingeniería de software, **anclado directamente al repositorio local** — es decir, con acceso de lectura/escritura a los datos, al código y a la capacidad de ejecutar Python en el mismo ambiente. Esto permitió iterar de forma conversacional: explorar los datos, tomar decisiones de limpieza, entrenar y comparar modelos, generar reportes visuales, y validar cada paso antes de avanzar al siguiente, todo dejando registro en el repositorio (código, reportes HTML, modelos entrenados).

El stack técnico usado fue Python, con `pandas`/`numpy` para manejo de datos, `scikit-learn` para la regresión logística y el pipeline de preprocesamiento, `LightGBM` y `XGBoost` para los modelos de boosting, `statsmodels` para los diagnósticos estadísticos (VIF, heterocedasticidad), e `imbalanced-learn` para las pruebas de balanceo de clases.

---

## 4. Metodología

El trabajo se hizo en tres grandes etapas: primero un **exploratorio de datos** (para entender qué tan confiables son los datos antes de modelar), después una **preparación** de esos datos (corrección de inconsistencias, definición del pipeline de limpieza), y finalmente un **proceso iterativo de modelamiento** en 6 rondas, cada una probando una hipótesis concreta.

### 4.1 Exploración

**Univariada** — revisamos cada campo por separado: nulos, valores fuera de rango, distribución. Encontramos:

- **Inconsistencias de captura**: 10% de los ingresos declarados están muy por debajo del sueldo mínimo (error de unidades), 70 solicitudes con edad fuera del rango válido [18, 100] años (hasta 133 años), y 9 solicitudes con una tasa de interés que supera la Tasa Máxima Convencional que publica mensualmente la CMF (verificado cruzando la fecha, el monto y el plazo de cada solicitud contra los certificados oficiales).
- **Nulos que son señal, no ruido**: los nulos de `ingreso_declarado` (18%) y `antiguedad_laboral_meses` (16%) no son aleatorios — los segundos coinciden al 100% con `tipo_empleo` (informal/jubilado no tiene "antigüedad en el empleo actual" que reportar, es un nulo estructural, no un error).
- **Una fuga de información**: `num_contactos_ult_trimestre` tenía una correlación de 0,73 con el resultado — casi con certeza porque registra gestiones de cobranza que ocurren *después* del default, no antes. Se excluyó del modelo.
- **Un campo no confiable**: `dia_semana_solicitud` no coincide con la fecha real en el 86% de los casos.

**Multivariada** — con los campos ya entendidos, medimos cómo se relacionan entre sí y con el resultado:

- **Correlación** (Pearson y Spearman): `score_buro` correlaciona moderadamente con `tasa_interes_anual` (-0,56), `num_consultas_buro_3m` (-0,47) y `uso_linea_credito_pct` (-0,47) — señales relacionadas pero no redundantes.
- **Multicolinealidad** (VIF): sin problema — el VIF máximo entre todas las variables numéricas fue 1,88 (la regla práctica de alerta es 5). Se pueden usar todas las variables juntas en la regresión logística sin que se desestabilicen los coeficientes.
- **Heterocedasticidad** (test de Levene y de Breusch-Pagan): confirmada en la mayoría de las variables clave. Esto es evidencia adicional de que un modelo lineal simple (regresión lineal) no es apropiado para este problema, y refuerza usar regresión **logística** o **boosting**.
- **Poder predictivo por variable** (Information Value): confirmó cuantitativamente la sospecha de fuga en `num_contactos_ult_trimestre` (IV=4,52, muy por sobre el umbral de sospecha de 0,5) y validó que `score_buro` (IV=1,50), `uso_linea_credito_pct` (IV=0,80) y `num_consultas_buro_3m` (IV=0,52) son los predictores legítimos más fuertes.
- **ANOVA / Chi-cuadrado**: `tipo_empleo` y `canal` están significativamente asociados al riesgo; `region` y `dia_semana_solicitud` no aportan señal.

### 4.2 Preparación

Con los hallazgos del exploratorio, definimos un pipeline de limpieza y preprocesamiento aplicado de forma consistente a train y test:

- **Corrección de inconsistencias** (clave de la Ronda 4, ver 4.3): la edad imposible se reemplazó por la mediana de edades válidas, el ingreso declarado con error de unidades se corrigió, y la tasa de interés se ajustó a la Tasa Máxima Convencional vigente en los 9 casos que la superaban. Cada corrección quedó marcada con una variable *flag* (`flag_edad_inconsistente`, `flag_ingreso_inconsistente`, `flag_tasa_corregida`) para que el modelo supiera qué registros fueron ajustados.
- **Imputación de nulos**: los nulos estructurales de `antiguedad_laboral_meses` (informal/jubilado) se rellenaron con 0; los nulos de `ingreso_declarado` se imputaron con la mediana calculada dentro de cada grupo de `tipo_empleo`, para no mezclar escalas de ingreso entre perfiles distintos.
- **Transformación de variables**: las variables monetarias con distribución sesgada (`monto_solicitado`, `ingreso_declarado`, `deuda_sistema`) se transformaron con logaritmo (`log1p`) para estabilizar su varianza; las variables categóricas se codificaron con *one-hot encoding*; para la regresión logística se estandarizaron las variables numéricas.
- **Exclusión de variables**: se excluyó `num_contactos_ult_trimestre` por fuga de información, y `dia_semana_solicitud` por no ser confiable.

### 4.3 Modelamiento

En cada ronda entrenamos y comparamos **tres familias de modelos** — regresión logística, boosting (LightGBM), y boosting alternativo (XGBoost) — usando siempre el mismo principio de validación: **entrenar solo con el pasado y medir qué tan bien predice el futuro** (partición temporal), y elegir hiperparámetros exigiendo que el modelo no solo discrimine bien (AUC), sino que sus probabilidades sean confiables (Brier score) — un modelo que "adivina bien el orden" pero da probabilidades mal calibradas no sirve para una política basada en un umbral de probabilidad.

| Ronda | Qué se probó | Conclusión |
|---|---|---|
| 1 | Base: datos tal cual, sin corregir inconsistencias | Punto de partida |
| 2 | + mes de solicitud como variable, entrenado solo con 2024 | Sin mejora real; empeoró la confiabilidad de las probabilidades → **descartada** |
| 3 | Balanceo sintético de clases (SMOTE) + corrección de calibración, sin `tasa_interes_anual` | Ayudó levemente solo en la logística; dañó la calibración del boosting → sin ganancia neta |
| **4** | **Corrección de las 3 inconsistencias de datos (edad, ingreso, tasa vs. TMC)** | **Mejora consistente en todos los modelos, sin efectos negativos → elegida** |
| 5 | Entrenar con enero-junio 2024, validar con julio-diciembre 2024 | Validación contra un examen más fácil (no comparable) → **descartada** |
| 6 | Entrenar solo con los 6 meses más recientes (jul-dic 2024) | Sin mejora frente a usar todo el historial disponible → **descartada** |

---

## 5. Resultados

### 5.1 Comparación de modelos — todas las rondas comparables

Validado sobre solicitudes de enero-febrero 2025 que ningún modelo vio al entrenar (n=4.874, salvo la Ronda 5, marcada aparte por usar una ventana de validación distinta y no comparable en magnitud).

| Ronda | Modelo | AUC | Gini | KS | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| 1 | Logística | 0,834 | 0,667 | 0,521 | 0,484 | 0,090 |
| 1 | LightGBM | 0,831 | 0,661 | 0,517 | 0,468 | 0,090 |
| 1 | XGBoost | 0,835 | 0,670 | 0,524 | 0,484 | 0,089 |
| 2 (descartada) | Logística | 0,834 | 0,668 | 0,525 | 0,484 | 0,092 |
| 2 (descartada) | LightGBM | 0,832 | 0,664 | 0,527 | 0,473 | 0,092 |
| 2 (descartada) | XGBoost | 0,834 | 0,668 | 0,526 | 0,487 | 0,091 |
| 3 | Logística | 0,834 | 0,668 | 0,520 | 0,484 | 0,089 |
| 3 | LightGBM | 0,830 | 0,660 | 0,516 | 0,471 | 0,090 |
| 3 | XGBoost | 0,835 | 0,669 | 0,522 | 0,483 | 0,089 |
| **4** | **Logística (elegida)** | **0,835** | **0,670** | **0,529** | **0,485** | **0,089** |
| 4 | LightGBM | 0,832 | 0,663 | 0,523 | 0,470 | 0,090 |
| 4 | XGBoost | 0,835 | 0,670 | 0,530 | 0,483 | 0,089 |
| 5 (no comparable) | Logística | 0,850 | 0,701 | 0,532 | 0,446 | 0,077 |
| 5 (no comparable) | LightGBM | 0,841 | 0,683 | 0,524 | 0,420 | 0,079 |
| 5 (no comparable) | XGBoost | 0,848 | 0,695 | 0,530 | 0,438 | 0,078 |
| 6 (descartada) | Logística | 0,834 | 0,669 | 0,526 | 0,485 | 0,089 |
| 6 (descartada) | LightGBM | 0,827 | 0,654 | 0,513 | 0,455 | 0,091 |
| 6 (descartada) | XGBoost | 0,832 | 0,665 | 0,522 | 0,467 | 0,090 |

**Lectura**: las tres familias de modelos rinden prácticamente igual dentro de cada ronda (diferencias de 0,1%-0,5% en AUC) — el tipo de algoritmo no es lo que marca la diferencia en este problema, es la calidad de los datos de entrada. La Ronda 4 tiene el mejor AUC/KS entre las rondas comparables, en las tres familias de modelos.

### 5.2 Economía de la política de aprobación, por ronda

Ganancia real (con resultados observados) al aplicar la política de umbrales de la sección 6 sobre las mismas 4.874 solicitudes de validación.

| Ronda | Modelo | % aprobado | Ganancia con política | Pérdida evitada |
|---|---|---|---|---|
| 1 | Logística | 77,1% | $404.680.600 | $444.081.000 |
| 1 | XGBoost | 76,8% | $405.026.700 | $442.079.000 |
| 2 (descartada) | Logística | 82,3% | $382.467.200 | $390.247.000 |
| 3 | Logística | 77,0% | $407.584.300 | $451.060.500 |
| **4** | **Logística (elegida)** | **77,4%** | **$406.006.500** | **$445.692.500** |
| 4 | XGBoost | 76,8% | $405.750.900 | $443.250.500 |
| 6 (descartada) | XGBoost | 74,2% | $422.311.800 | $479.369.000 |

Como referencia, **aprobar el 100% de las solicitudes** (la práctica implícita actual) genera solo **$77.239.300** en el mismo grupo de validación — cualquiera de las políticas de la tabla multiplica esa ganancia por 5 veces o más, porque el 55% de pérdida de un default pesa mucho más que el 6% de ganancia de un crédito bueno, y la política evita selectivamente a los más riesgosos sin sacrificar demasiados clientes buenos.

*Nota: LightGBM no se evaluó en la política porque su performance fue sistemáticamente igual o peor que la logística en la sección 5.1, y la logística ya se había fijado como modelo preferido por interpretabilidad.*

### 5.3 Elección final: Ronda 4, Regresión Logística

Se recomienda la **regresión logística de la Ronda 4** como modelo de producción, por tres razones:

1. **Mejor performance entre lo comparable**: mejor AUC, KS y Brier que las Rondas 1, 2, 3 y 6, de forma consistente en las tres familias de modelos probadas (no es una casualidad de un solo algoritmo).
2. **Viene de datos limpios**: es la única ronda cuya mejora frente al baseline (Ronda 1) es limpia y sin contrapartida negativa — las otras alternativas (agregar variables, SMOTE, ventanas más cortas) mostraron efectos marginales, mixtos, o mejoras engañosas por una validación menos exigente.
3. **Interpretabilidad**: rinde prácticamente igual que XGBoost y LightGBM, pero cada coeficiente de la regresión logística tiene una lectura directa (qué variable sube o baja el riesgo y cuánto) — clave para justificar una decisión de aprobación o rechazo ante un cliente o un regulador, algo que un modelo de árboles no ofrece con la misma claridad.

---

## 6. Política de aprobación

Un modelo que solo entrega una probabilidad no decide nada por sí solo — hay que traducir esa probabilidad en un "sí" o un "no". La forma correcta de hacerlo no es un corte arbitrario (como 50%), sino comparar el **valor esperado de aprobar** contra la alternativa segura de **rechazar**.

**La economía del producto** (simplificada): si el cliente paga, la empresa gana aproximadamente el 6% del monto solicitado, prorrateado por el plazo (`monto × 12% anual × plazo/12 × 0,5`, donde el factor 0,5 corrige porque el crédito se amortiza y el saldo pendiente promedio es la mitad del monto original). Si el cliente cae en default, la empresa pierde el 55% del monto (LGD — *Loss Given Default*, el porcentaje que no se logra recuperar).

**La lógica del umbral**: rechazar una solicitud siempre vale $0 — no se gana ni se pierde. Aprobar es una apuesta: con probabilidad `(1-p)` se gana la ganancia, con probabilidad `p` se pierde la pérdida. El valor esperado de esa apuesta es:

> **EV(aprobar) = (1-p) × ganancia − p × pérdida**

La regla de decisión es simple: se aprueba solo si esa apuesta vale más que el $0 seguro de rechazar, es decir, si `EV(aprobar) > 0`. El **umbral óptimo (p*)** es exactamente el punto donde esa expresión es igual a cero — el punto de indiferencia entre aprobar y rechazar. Despejando:

> **p* = ganancia / (ganancia + pérdida)**

Como la ganancia depende del plazo pero la pérdida es un % fijo del monto, **el monto se cancela en la fórmula: el umbral no depende de cuánto pide el cliente, solo de a cuántos meses lo pide**. Créditos más largos generan más interés acumulado y toleran algo más de riesgo antes de que convenga rechazar:

| Plazo | Rechazar si prob. de default ≥ |
|---|---|
| 6 meses | 5,2% |
| 12 meses | 9,8% |
| 18 meses | 14,1% |
| 24 meses | 17,9% |
| 36 meses | 24,7% |
| 48 meses | 30,4% |

**Por qué no usar simplemente 50%**: un corte de 0,5 solo tiene sentido si equivocarse en ambas direcciones cuesta lo mismo. Acá no es así — perder por un default (55% del monto) pesa mucho más que dejar de ganar por rechazar a un buen pagador (~6% del monto). Con la tasa de default real (~10-13%), casi ninguna solicitud supera 50% de probabilidad predicha, así que un corte de 0,5 equivaldría en la práctica a **aprobar casi todo** — exactamente el escenario que menos plata genera.

**Por qué la calibración importa tanto como la discriminación**: el AUC mide si el modelo *ordena* bien a los clientes de más a menos riesgoso, pero no si el número que entrega es confiable. El **Brier score** sí mide eso — es el promedio de `(probabilidad predicha − resultado real)²`; más bajo es mejor. Como la política de aprobación compara la probabilidad exacta contra el umbral de la tabla de arriba (no solo el orden relativo entre clientes), un modelo mal calibrado toma decisiones equivocadas en el punto de corte aunque "ordene" perfecto. Por eso, en cada ronda, el criterio de selección de hiperparámetros exigió buena calibración además de buen AUC.

**Resultado de aplicar esta política**, con el modelo de la Ronda 4, **validado con resultados reales de 2025** (4.874 solicitudes nunca vistas al entrenar):

- **Aprueba el 77,4%** de las solicitudes.
- Genera **$406.006.500** de ganancia, evitando **$445.692.500** en pérdidas por default.
- Esto es **5,3 veces más ganancia que aprobar todo** ($77.239.300).

---

## 7. Conclusiones

- El primer modelo de probabilidad de default de Andina Crédito (regresión logística, Ronda 4) discrimina bien (AUC 0,835) y está bien calibrado (Brier 0,089), condición necesaria para que la política de umbrales funcione correctamente.
- La mejora decisiva no vino de probar algoritmos más sofisticados (LightGBM y XGBoost rinden prácticamente igual), sino de **corregir la calidad de los datos de entrada** — edad, ingreso y tasa de interés inconsistentes.
- Aplicada a las 12.000 solicitudes nuevas de **test.csv**, la política recomendada genera una ganancia esperada de **$1.234.680.251**, casi 5 veces más que la práctica implícita de aprobar todo.
- Se recomienda implementar el modelo y la política de umbrales por plazo como el nuevo estándar de aprobación, con revisión y recalibración periódica.

---

## 8. Limitaciones

- El modelo final se reentrenó con el 100% de train.csv (incluyendo enero-febrero 2025) para maximizar los datos disponibles al scorear test.csv; las métricas de la sección 5 provienen de la versión entrenada solo hasta diciembre 2024, para que la validación fuera honesta.
- `tasa_interes_anual` se mantuvo como variable, pero es fijada por la propia empresa según su política de pricing — existe riesgo de que el modelo esté parcialmente re-aprendiendo esa política en vez de aportar señal 100% independiente. Recomendamos evaluar el modelo con y sin esa variable en la próxima iteración.
- La tasa de default real subió de ~8-10% en 2024 a ~13% en enero-febrero de 2025. El modelo sigue discriminando bien pese a ese cambio, pero recomendamos **monitoreo mensual** de la tasa de default real vs. la predicha, y recalibrar si la tendencia continúa.
- El dataset cubre solo 14 meses — no permite distinguir con certeza estacionalidad de una tendencia sostenida. Reevaluar variables temporales (como el mes de solicitud) cuando haya más historia disponible.

**Archivos de respaldo:** `predictions.csv` (predicciones sobre test.csv con el modelo elegido), `reports/modelamiento_ronda1.html` a `ronda6.html` (detalle técnico de cada ronda), `reports/politica_aprobacion_ronda1.html`, `ronda3.html` y `ronda4.html` (detalle de la política por ronda), `reports/exploracion_variables.html` y `reports/analisis_multivariado.html` (exploración de datos).
