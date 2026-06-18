# Diccionario de datos

Fuente: data warehouse de Andina Crédito, snapshot a la fecha de extracción.

| Columna | Descripción |
|---|---|
| `id_solicitud` | Identificador único de la solicitud. |
| `fecha_solicitud` | Fecha de la solicitud del crédito. |
| `edad` | Edad del solicitante en años. |
| `tipo_empleo` | Situación laboral declarada: `dependiente`, `independiente`, `informal`, `jubilado`. |
| `ingreso_declarado` | Ingreso mensual declarado (CLP). |
| `antiguedad_laboral_meses` | Meses en el empleo actual. |
| `region` | Región de residencia. |
| `canal` | Canal de originación: `digital`, `sucursal`, `partner`. |
| `antiguedad_cliente_meses` | Meses desde que la persona es cliente de Andina Crédito. |
| `score_buro` | Score del buró de crédito externo (300–850, mayor = mejor). |
| `deuda_sistema` | Deuda total vigente en el sistema financiero (CLP). |
| `num_creditos_vigentes` | Número de créditos vigentes en el sistema. |
| `peor_morosidad_12m` | Peor mora histórica (días) en los últimos 12 meses previos a la solicitud. |
| `num_contactos_ult_trimestre` | Número de contactos/interacciones del cliente con la plataforma registrados en el sistema en el último trimestre. |
| `num_consultas_buro_3m` | Consultas al buró en los últimos 3 meses. |
| `uso_linea_credito_pct` | % de utilización de líneas de crédito. |
| `monto_solicitado` | Monto del crédito solicitado (CLP). |
| `plazo_meses` | Plazo del crédito en meses. |
| `tasa_interes_anual` | Tasa de interés anual asignada por pricing (%). |
| `dia_semana_solicitud` | Día de la semana de la solicitud. |
| `default_12m` | **Target** (solo en train): 1 si el crédito alcanzó mora de 90+ días dentro de los 12 meses posteriores al desembolso. |
