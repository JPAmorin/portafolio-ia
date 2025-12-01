# Modelos de Lenguaje, Prompting, Plantillas y Salida Estructurada

| Title | Date |
| --- | --- |
| Modelos de Lenguaje, Prompting, Plantillas y Salida Estructurada | 29/11/25 |

# Modelos de Lenguaje, Prompting, Plantillas y Salida Estructurada

## **Contexto**

En esta actividad del exploraremos cómo usar LangChain junto con OpenAI para construir aplicaciones que utilicen modelos de lenguaje (LLMs). Veremos conceptos clave como el ajuste de parámetros de generación (por ejemplo temperature, max_tokens, top_p), la creación de plantillas reutilizables de prompts, y la generación de salidas estructuradas en formato JSON mediante esquemas definidos. También aprenderemos a medir métricas relevantes como el uso de tokens y la latencia, lo que ayudará a evaluar la eficiencia de nuestras interacciones con el modelo. 

## **Objetivo**

- Instanciar un modelo de chat de OpenAI desde LangChain (ChatOpenAI) y realizar invocaciones básicas. ([docs.langchain.com](https://docs.langchain.com/oss/python/integrations/chat/openai?utm_source=chatgpt.com))
- Controlar parámetros de decodificación comunes (temperature, max_tokens, top_p) y razonar sobre su efecto.
- Diseñar prompts reutilizables con ChatPromptTemplate y el LCEL (operador | o pipe) para encadenar componentes. ([api.python.langchain.com](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html?utm_source=chatgpt.com))
- Obtener salidas estructuradas (JSON/Pydantic) de forma fiable con with_structured_output(…). ([docs.langchain.com](https://docs.langchain.com/oss/python/langchain/structured-output?utm_source=chatgpt.com))
- Medir tokens/latencia con tracing en Lang Smith (o callbacks) como base de observabilidad. ([docs.langchain.com](https://docs.langchain.com/langsmith/trace-with-langchain?utm_source=chatgpt.com))

## **Actividades**

- **Parte 0: Setup y *Hello LLM***
- **Parte 1: Parámetros clave (temperature, max_tokens, top_p)**
- **Parte 2: De texto suelto a plantillas con ChatPromptTemplate + LCEL**
- **Parte 3: Salida estructurada (JSON) sin *post-processing* frágil**
- **Parte 4: Métricas mínimas — tokens y latencia (LangSmith / callbacks)**
- **Parte 5: Mini-tareas guiadas (aún sin RAG)**
- **Parte 6: Zero-shot vs Few-shot — "Playground" guiado**
- **Parte 7: Resúmenes — single-doc y multi-doc (map-reduce)**
- **Parte 8: Extracción de información — entidades y campos clave**
- **Parte 9: RAG básico con textos locales (sin base de datos externa)**

## **Desarrollo**

Antes de empezar la actividad, si el lector quiere, puede utilizar el archivo de Google Colab de las referencias para probar los modelos en su propia máquina. Para hacer esto deberá tener su llave API de [LangSmith](https://smith.langchain.com/) y [OpenAI](https://platform.openai.com/api-keys). De igual manera, podrá explorar este informe y ver los resultados obtenidos.

Para empezar esta actividad, debemos aclarar qué es un modelo de lenguaje (LLM). Los LLMs son sistemas de inteligencia artificial entrenados con grandes cantidades de texto para aprender patrones del lenguaje y generar respuestas coherentes y útiles. Pueden comprender instrucciones, producir textos, resumir información, traducir y más tareas complejas basadas en el lenguaje natural. 

En este caso estaremos usando un modelo pre-entrenado de OpenAI, el GPT-5-mini. Este modelo nos da resultados con alta velocidad, a un bajo costo, con una capacidad suficiente para comprender instrucciones, generar texto coherente y realizar tareas comunes de procesamiento de lenguaje natural. Gracias a que GPT-5-mini es un modelo pre-entrenado, solo debemos instanciarlo y configurar su "temperatura" (o creatividad). A continuación vemos una respuesta generada con temperatura 0.0 (baja, más determinista). ([Evidencia 1](#evidencia-1)). Por otro lado, podemos ver cómo varía la respuesta cuando le damos más "libertades creativas" al modelo. ([Evidencia 2](#evidencia-2)).

Esto nos da la posibilidad de darle un enfoque más a nuestro gusto del modelo según las necesidades que tengamos del mismo, ajustando su comportamiento para que sea más preciso y consistente cuando lo necesitemos, o más creativo y flexible cuando la tarea lo requiera. Esta adaptabilidad es un gran atractivo que tienen estos modelos de lenguaje natural. 

También podemos darles múltiples prompts en una sola vez, lo cual nos devuelve una respuesta para cada prompt, de manera ordenada. Una vez más podemos ver la diferencia en las respuestas según la temperatura que usemos. ([Evidencia 3](#evidencia-3)). Además, según el prompt utilizado, podemos pedirles a los modelos que nos devuelvan respuestas en formatos específicos. En este caso le pedimos que nos responda con Haikus, poemas japoneses de tres versos de 5, 7 y 5 sílabas. ([Evidencia 4](#evidencia-4)).

Podemos usar ChatPromptTemplate para estructurar mensajes dirigidos al modelo, separando claramente el rol del sistema y del usuario. ([Evidencia 5](#evidencia-5)). En el mensaje del sistema definimos el comportamiento deseado del asistente, mientras que en el mensaje del usuario tenemos un parámetro dinámico (tema) que permite reutilizar el mismo prompt para distintos contenidos.

Las salidas estructuradas son una opción que tenemos para interactuar con los modelos de lenguaje. En lugar de recibir texto libre, que puede variar en formato, estilo o consistencia, el modelo produce una respuesta que sigue un esquema predefinido, lo que facilita su validación y posterior uso dentro de una aplicación. Esto es especialmente valioso cuando se necesitan datos confiables para automatizar procesos, integrar resultados en otros sistemas o evitar errores derivados de interpretaciones ambiguas del texto generado. ([Evidencia 6](#evidencia-6)).

Otra tarea importante que podemos realizar es la traducción de texto, siendo esta una de las tareas más usadas en Internet. ([Evidencia 7](#evidencia-7)). El modelo interpreta, por el contexto, el idioma del texto base y el idioma objetivo al que tendrá que traducir, sin necesidad de que se ajuste manualmente nada. Este comportamiento puede ser determinado, usando mensajes de rol del sistema, aclarando cómo debe responder el modelo frente al contexto presentado. ([Evidencia 8](#evidencia-8)).

La clasificación de sentimiento de texto es una tarea que se basa en analizar las palabras que componen un texto y determinar la emoción que emite quiere emitir el escritor. Esta tarea es útil para comprender opiniones de usuarios, monitorear la satisfacción de clientes o detectar reacciones positivas y negativas en redes sociales. A continuación vemos la diferencia a la hora de realizar esta tarea haciendo uso de Zero-shot y Few-shot. Zero-shot consiste en pedirle al modelo que clasifique el sentimiento sin darle ningún ejemplo previo, confiando únicamente en su conocimiento general. En cambio, Few-shot implica proporcionarle algunos ejemplos concretos antes de la tarea, permitiendo que el modelo entienda mejor el estilo, el criterio o el formato deseado, y produciendo resultados generalmente más precisos y consistentes. ([Evidencia 9](#evidencia-9)).

También podemos resumir textos extensos, dividiéndolo en fragmentos manejables y extrayendo de cada uno los conceptos más importantes. Logramos esto partiendo el texto en chunks para que el modelo pueda procesarlos sin superar sus límites contextuales; luego, cada fragmento se resume en bullets que capturan sus ideas clave de forma clara y factual. Finalmente, esos bullets individuales se consolidan en un único resumen sintético, eliminando redundancias y manteniendo los puntos esenciales. Este enfoque permite obtener un resumen coherente y enfocado incluso cuando se trabaja con documentos largos o altamente detallados. ([Evidencia 10](#evidencia-10)).

La extracción de información es otra tarea que podemos realizar con modelos de lenguaje. En lugar de devolver un texto libre con posibles variaciones en formato, el modelo devuelve un objeto que sigue un esquema definido, lo que facilita procesar y reutilizar esos datos dentro de una aplicación. Esta técnica resulta especialmente útil para tareas de análisis de documentos, clasificación de contenido o construcción de pipelines que necesitan información precisa y bien organizada a partir de textos no estructurados. ([Evidencia 11](#evidencia-11)).

Por último podemos usar RAG (Retrieval-Augmented Generation), una técnica que permite que un modelo de lenguaje responda preguntas basándose únicamente en información recuperada de un conjunto de documentos. Primero, los textos se dividen en fragmentos y se indexan mediante embeddings para permitir una búsqueda semántica eficiente. Luego, ante una consulta del usuario, el sistema recupera los fragmentos más relevantes y los pasa al modelo junto con un prompt que exige responder solo usando ese contexto. De esta manera, la respuesta queda fundamentada en los datos disponibles, reduciendo al mínimo la alucinación del modelo y permitiendo obtener respuestas más precisas, confiables y trazables. ([Evidencia 12](#evidencia-12)).

## **Evidencias**

### **Evidencia 1** {#evidencia-1}

```
prompt: "Definí 'Transformer' en una sola oración."
respuesta: "Transformer es una arquitectura de red neuronal que, mediante mecanismos de atención (self-attention) y codificaciones posicionales, modela dependencias en secuencias y permite procesarlas en paralelo para tareas como traducción automática y procesamiento del lenguaje natural."
```

### **Evidencia 2** {#evidencia-2}

```
prompt: "Definí 'Transformer' en una sola oración."
respuesta: "Transformer: arquitectura de red neuronal introducida por Vaswani et al. (2017) que utiliza mecanismos de auto‑atención para procesar secuencias en paralelo y modelar dependencias a largo plazo en tareas de lenguaje y otras modalidades."
```

### **Evidencia 3** {#evidencia-3}

```
prompts: [
    "Escribí un tuit (<=20 palabras) celebrando un paper de IA.",
    "Dame 3 bullets concisos sobre ventajas de los Transformers."
]

Respuestas:

--- Temperature=0.0 ---
[1] ¡Enhorabuena por el paper! Un avance de IA que inspira innovación responsable y colaboración global.
[2] - Permiten paralelizar el entrenamiento (más rápidos y eficientes que RNNs/seq2seq tradicionales).  
- Capturan dependencias a largo plazo mediante self-attention, manejando contexto amplio de forma efectiva.  
- Escalan bien y facilitan el preentrenamiento/fine-tuning, logrando alto rendimiento en muchas tareas.

--- Temperature=0.5 ---
[1] ¡Celebrando este nuevo paper de IA! Innovación, rigor y potencial transformador. Enhorabuena, equipo.
[2] - Paralelizable: la atención permite procesar secuencias en paralelo, acelerando el entrenamiento frente a RNNs.  
- Captura dependencias a larga distancia de forma eficiente, modelando contexto global.  
- Altamente escalable y favorable para preentrenamiento + fine-tuning, logrando transferencia y SOTA en muchas tareas.

--- Temperature=0.9 ---
[1] Brillante paper de IA: rigor, innovación y impacto. ¡Felicidades al equipo! #IA #Investigación
[2] - Capturan dependencias a largo alcance mediante self-attention sin procesamiento secuencial.  
- Permiten paralelización durante el entrenamiento, acelerando el cómputo frente a RNNs.  
- Escalan bien y facilitan preentrenamiento/transferencia, logrando alto rendimiento en muchas tareas.
```

### **Evidencia 4** {#evidencia-4}

```
prompt: "Escribí un haiku sobre evaluación de modelos."
Respuestas:

--- Temperature=0.0 ---
Métricas claras  
Pérdida y precisión  
Corta la duda

--- Temperature=0.5 ---
Datos en la red
mide sesgos, errores
veraz decisión

--- Temperature=0.9 ---
Pone a prueba  
métricas, validación  
mejora sin fin
```

### **Evidencia 5** {#evidencia-5}

```
prompts:
  ("system", "Sos un asistente conciso, exacto y profesional."),
  ("human",  "Explicá {tema} en <= 3 oraciones, con un ejemplo real.")
Respuestas:
  La atención multi-cabeza calcula varias atenciones en paralelo sobre distintas proyecciones de consultas, claves y valores para que cada cabeza capture diferentes tipos de relación (sintácticas, semánticas o posicionales), y luego concatena y proyecta esos resultados en una representación final.  
	Ejemplo real: al traducir "Ella le dio la mano al presidente", una cabeza puede alinear "la mano" con "hand" (objeto), otra mantener la concordancia y dependencia sujeto-verbo, y otra identificar "presidente" como entidad para traducirla correctamente según el contexto.
```

### **Evidencia 6** {#evidencia-6}

```
prompt: "Resumí en 3 bullets los riesgos de la 'prompt injection' en LLM apps."
Respuesta:
Resumen(title="Riesgos de la 'prompt injection' en apps con LLM", bullets=['Exfiltración de datos sensibles: entradas maliciosas pueden inducir al modelo a revelar prompts del sistema, secretos, datos de usuarios o información interna.', 'Manipulación del comportamiento y acciones no autorizadas: instrucciones inyectadas pueden anular controles, llevar al modelo a dar consejos peligrosos o ejecutar operaciones en sistemas conectados (APIs, bases de datos).', 'Erosión de confianza y cumplimiento: produce respuestas incorrectas o peligrosas, incumplimiento normativo, daño reputacional y posible propagación de instrucciones maliciosas a terceros.'])
```

### **Evidencia 7** {#evidencia-7}

```
prompt: "Traducí al portugués: 'Excelente trabajo del equipo'."
Respuesta: text='Excelente trabalho da equipe' lang='pt'
```

### **Evidencia 8** {#evidencia-8}

```
prompts:
    ("system", "Respondé SOLO usando el contexto. Si no alcanza, decí 'No suficiente contexto'."),
    ("human",  "Contexto:\n{contexto}\n\nPregunta: {pregunta}\nRespuesta breve:")
Respuesta:
content='No suficiente contexto' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 204, 'prompt_tokens': 54, 'total_tokens': 258, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 192, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-5-mini-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-ChYJGpFgfyMNkTNdpgzOCa2OMxzMq', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--39f4e3a4-4c1b-4ac5-9a7f-05fba9ee06a9-0' usage_metadata={'input_tokens': 54, 'output_tokens': 204, 'total_tokens': 258, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 192}}
```

### **Evidencia 9** {#evidencia-9}

```
Zero-shot

prompts:
    ("system", "Sos un asistente conciso y exacto."),
    ("human",  "Clasificá el sentimiento de este texto como POS, NEG o NEU:\n\n{texto}")
Respuesta:
	POS
	NEG
	NEU
	
Few-shot

prompts:
    ("system", "Sos un asistente conciso y exacto."),
    ("human",  "Ejemplo:\nTexto: 'El producto superó mis expectativas'\nEtiqueta: POS"),
    ("human",  "Ejemplo:\nTexto: 'La entrega fue tarde y vino roto'\nEtiqueta: NEG"),
    ("human",  "Texto: {texto}\nEtiqueta:")
Respuesta:
	Etiqueta: POS
	Etiqueta: NEG
	NEU
```

### **Evidencia 10** {#evidencia-10}

```
Texto a resumir

"""Max Emilian Verstappen (pronunciación en neerlandés: /ˈmɑks vɛrˈstɑ.pə(n)/; Hasselt, 30 de septiembre de 1997) es un piloto de automovilismo neerlandés nacido en Bélgica. Ganó el Campeonato Mundial de Karting en 2013 y finalizó tercero en el Campeonato Europeo de Fórmula 3 de la FIA en su debut en monoplazas. Debutó en Fórmula 1 con la escudería Toro Rosso en 2015. Desde 2016 es piloto del equipo Red Bull Racing, con el que se consagró tetracampeón del Campeonato Mundial de Fórmula 1 tras los títulos obtenidos en 2021, 2022, 2023 y 2024, y logró dos terceros puestos en 2019 y 2020.

Max Verstappen es el tercer piloto con mayor número de victorias en la historia de la categoría con 69 grandes premios ganados, el tercero con más podios: 125, el quinto con más pole position: 47, el sexto con más vueltas rápidas: 36, y el segundo con más Grand Chelem, igualado con Lewis Hamilton: 6.[5]

El 3 de octubre de 2014, siendo piloto de reserva de la escudería Toro Rosso, formó parte en la primera sesión oficial de entrenamientos libres del Gran Premio de Japón de 2014,[6] con 17 años y tres días, de esa forma fue la persona más joven en la historia de la Fórmula 1 en participar en un fin de semana de Gran Premio. Entre otros récords, es el primer piloto de nacionalidad neerlandesa en subir al primer lugar del podio de la categoría, lo hizo en el Gran Premio de España de 2016 en su primera carrera con Red Bull.[7]

Posee varios récords como el piloto más joven de la Fórmula 1: (1) Competir en un Gran Premio, hizo su debut con 17 años y 166 días en el Gran Premio de Australia de 2015 con la Scuderia Toro Rosso.[8] (2) Sumar puntos en el Gran Premio de Malasia de 2015 con 17 años y 180 días. (3) Ganar un Gran Premio. (4) Subir al podio. Los récords mencionados en los puntos 3 y 4 fueron logrados teniendo 18 años, 7 meses y 15 días en el Gran Premio de España de 2016.[7] (5) Récord de vuelta en una sesión, esta fue en la tercera práctica libre en el Autódromo Hermanos Rodríguez de México el 28 de octubre de 2017. (6) En el Gran Premio de Austria de 2021 después de 128 grandes premios, 50 podios y 15 victorias, consiguió su primer Grand Chelem con 23 años, 9 meses y 4 días.[9] (7) Obtener 100 podios con 26 años, 5 meses y 9 días en el Gran Premio de Arabia Saudita de 2024.[10][11]

Fue el primer ganador en la historia de una carrera sprint, clasificatoria para el Gran Premio de Gran Bretaña de 2021.[12] Ese año se proclamó campeón de Fórmula 1 tras ganar en el Gran Premio de Abu Dabi; última carrera de la temporada; derrotando al defensor del título Lewis Hamilton en la vuelta final, con quien había llegado empatado en puntos. De esta forma, también se convirtió en el primer neerlandés en obtener el título.[13] Dicha temporada fue catalogada casi en forma unánime, como una de las mejores en la historia de la categoría.[14]

En 2023 logró dos récords con respecto a las vueltas lideradas, tanto en su cantidad como en el kilometraje recorrido, 1003/1325 (75,70 %) y 4914/6700 (73,34 %) respectivamente.[15][16]

El «efecto Verstappen» en las normativas de la Federación Internacional del Automóvil (FIA). La llegada de Max Verstappen y otros pilotos de corta edad e inexperiencia a la Fórmula 1, provocó la desconfianza en las altas autoridades de la FIA. El 6 de enero de 2015 en una reunión mantenida en Doha la FIA dio los detalles del nuevo sistema de licencias que regiría a partir de 2016. Con respecto a la Superlicencia de la FIA es, como su nombre indica, una autorización, la cual ratifica la capacidad de un piloto y le abre la posibilidad a competir en la Fórmula 1. Entre los nuevos requisitos el más relevante es la edad del piloto, fijada en 18 años. Por ello, mientras esta normativa siga vigente, lo conseguido por él jamás podrá ser superado.[17][18]"""

prompts para resumir un chunk:

    ("system", "Resumí el siguiente fragmento en 2–3 bullets, claros y factuales."),
    ("human", "{input}")
    
prompts para combinar resultados:

    ("system", "Consolidá bullets redundantes y producí un resumen único y breve."),
    ("human", "Bullets:\n{bullets}\n\nResumen final (<=120 tokens):")

Resultado:

	Max Emilian Verstappen (Hasselt, 30/09/1997) es piloto neerlandés. Campeón mundial de F1 2021–2024; debutó en 2015 (Toro Rosso) y corre con Red Bull desde 2016. Acumula 69 victorias, 125 podios, 47 poles, 36 vueltas rápidas y 6 Grand Chelems; fue el primer neerlandés en ganar (España 2016) y el primer vencedor de una sprint (GBR 2021). Su precocidad motivó endurecer la Superlicencia (edad mínima 18).

```

### **Evidencia 11** {#evidencia-11}

```
Texto

"OpenAI anunció una colaboración con la Universidad Catolica del Uruguay en Montevideo el 05/11/2025."

Prompt:

"Extraé titulo, fecha y entidades (ORG/PER/LOC) del siguiente texto:\n\n{texto}"

Resultado
ExtractInfo(titulo=None, fecha='05/11/2025', entidades=[Entidad(tipo='ORG', valor='OpenAI'), Entidad(tipo='ORG', valor='Universidad Catolica del Uruguay'), Entidad(tipo='LOC', valor='Montevideo')])

```

### **Evidencia 12** {#evidencia-12}

```
Textos
    "LangChain soporta structured output con Pydantic.",
    "RAG combina recuperación + generación para mejor grounding.",
    "OpenAIEmbeddings facilita embeddings para indexar textos."
Prompts:
		("system", "Respondé SOLO con el contexto. Si no alcanza, decí 'No suficiente contexto'."),
    ("human",  "Contexto:\n{context}\n\nPregunta: {input}")
Input:
		"¿Qué ventaja clave aporta RAG?"
Respuesta:
		'RAG combina recuperación + generación para mejor grounding.'
```

## **Reflexión**

En este trabajo se exploraron técnicas fundamentales para construir aplicaciones prácticas basadas en modelos de lenguaje, desde las más simples hasta otras más estructuradas y avanzadas. Empezamos por el uso directo de un LLM con prompts básicos, entendiendo cómo parámetros como temperature influyen en el comportamiento del modelo. Vimos prompt templates, mecanismos para generar salidas estructuradas y estrategias Zero-shot y Few-shot, lo que nos permitió mejorar la precisión y consistencia según la tarea. Luego tratamos con la síntesis de información mediante resúmenes por fragmentos y reducción, así como la extracción precisa de datos a través de esquemas validados. Finalmente, integramos recuperación semántica con RAG para reforzar las respuestas del modelo con contexto real y evitar alucinaciones. En conjunto, estas técnicas muestran cómo un LLM deja de ser solo un generador de texto para convertirse en un componente confiable dentro de pipelines más amplios, capaces de procesar, estructurar, recuperar y razonar sobre información de manera mucho más robusta y controlada.

## **Referencias**

*Assignment UT4-14: LLMs con LangChain (OpenAI) — Prompting, Plantillas y Salida Estructurada - Fundamentos del Aprendizaje Automático - Universidad Católica del Uruguay*. (n.d.). [https://juanfkurucz.com/ucu-ia/ut4/14-langchain-openai-intro/](https://juanfkurucz.com/ucu-ia/ut4/14-langchain-openai-intro/)

*Guía de Ingeniería de Prompt – Nextra*. (n.d.). [https://www.promptingguide.ai/es](https://www.promptingguide.ai/es)

*Prompting best practices*. (n.d.). Claude Docs. [https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices)

*Google Colab*. (n.d.-l). [https://colab.research.google.com/drive/1EiHZyuJx8acrpR4-S077iwPJa5GMaAvr?usp=sharing](https://colab.research.google.com/drive/1EiHZyuJx8acrpR4-S077iwPJa5GMaAvr?usp=sharing)
