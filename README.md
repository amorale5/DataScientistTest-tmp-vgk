# Desafío Técnico — Data Scientist Senior

> **Cómo empezar:** haz un **fork** de este repositorio a tu cuenta de GitHub (botón *Fork*, arriba a la derecha). Trabaja en tu fork y, al terminar, **envíanos el link a tu repositorio público** según las instrucciones de [Entrega](#entrega). No abras Pull Requests sobre este repo.

## Contexto de negocio

**Andina Crédito** es una fintech de créditos de consumo. Hoy las solicitudes se aprueban con reglas simples y la mora se ha vuelto un problema: el equipo de Riesgo te pide construir el primer modelo de **probabilidad de default** (mora de 90+ días dentro de los 12 meses posteriores al desembolso) para apoyar la decisión de aprobación.

Recibes dos archivos:

- `data/train.csv`: 45.300 solicitudes históricas desembolsadas entre **2024-01 y 2025-02**, con el resultado observado (`default_12m`).
- `data/test.csv`: 12.000 solicitudes entre **2025-02 y 2025-06** que debes scorear (sin target).

La data proviene del data warehouse tal como está.

## Tu tarea

1. **Explora y prepara la data.** Documenta los problemas que encuentres y las decisiones que tomes.
2. **Construye un modelo** que estime la probabilidad de default para cada solicitud de `test.csv`.
3. **Define una estrategia de validación** que entregue una estimación honesta de cómo rendirá el modelo en producción. Justifícala.
4. **Recomienda una política de aprobación.** Usando la economía del producto (abajo), define el umbral de aprobación y estima la ganancia esperada de tu política versus aprobar todo.
5. **Comunica.** El gerente de Riesgo no es técnico: tu informe debe poder leerlo él.

### Economía del producto (simplificada)

- Si el cliente **paga**: ganancia ≈ `monto_solicitado × 12% anual × (plazo_meses / 12) × 0.5` (el factor 0.5 aproxima la amortización del capital).
- Si el cliente **cae en default**: pérdida ≈ `monto_solicitado × 55%` (LGD).
- Una solicitud rechazada no genera ganancia ni pérdida.

## Uso de IA

**Puedes y debes usar herramientas de IA** (Copilot, Claude, ChatGPT, agentes, etc.). Nos interesa cómo las usas, no que no las uses. Incluye en tu entrega un archivo `AI_USAGE.md` con:

- Qué herramientas usaste y para qué partes del trabajo.
- Al menos **2 ejemplos concretos** donde el output de la IA fue incorrecto, subóptimo o requirió tu corrección, y qué hiciste al respecto.
- Qué validaste manualmente antes de confiar en código o análisis generado.

Una entrega donde todo fue aceptado tal como lo produjo la IA, sin evidencia de criterio propio, será evaluada negativamente aunque las métricas sean buenas.

## Entregables

Dentro de tu repositorio público debes incluir:

1. `predictions.csv` en la raíz, con columnas `id_solicitud,prob_default` y las 12.000 filas de test. Mira `predictions_example.csv` para el formato exacto.
2. **Código reproducible**: tu solución más un `README` con los pasos para correrla (instalar dependencias y regenerar `predictions.csv`). Notebooks están bien si están ordenados y se ejecutan de principio a fin.
3. **Informe ejecutivo** (máximo 2 páginas, en el repo como `.md` o `.pdf`): problemas encontrados en la data, enfoque, performance esperada con su justificación, política de aprobación recomendada y ganancia estimada, limitaciones y próximos pasos.
4. `AI_USAGE.md` según lo descrito arriba.

## Entrega

1. **Forkea** este repositorio a tu cuenta de GitHub y trabaja sobre tu fork (mantenlo **público**).
2. Haz commits con frecuencia: el historial nos ayuda a entender tu proceso, así que evita un único commit gigante al final.
3. Cuando termines, responde el correo del desafío con **el link a tu repositorio** (ej. `https://github.com/tu-usuario/desafio-ds-senior`).
4. Asegúrate de que el repo sea público y de que `git clone` + los pasos de tu `README` reproduzcan tu `predictions.csv`.

## Condiciones

- Plazo: **48 horas** desde la recepción del correo. Esperamos una dedicación efectiva de 4 a 8 horas: prioriza, no es necesario explorar todo.
- Lenguaje libre (Python recomendado). Librerías libres.
- Reportaremos tu performance real sobre el target oculto de test; una brecha grande entre la performance que declaras y la real será parte de la conversación de evaluación.

## Qué valoramos (en orden)

1. Criterio para trabajar la data y decisiones metodológicas bien justificadas.
2. Honestidad en la estimación de performance (validación bien diseñada).
3. Traducción del modelo a una decisión de negocio con impacto cuantificado.
4. Uso transparente e inteligente de IA.
5. Performance del modelo — importa, pero menos que lo anterior.

Buena suerte.
