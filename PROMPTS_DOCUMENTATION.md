# Документация Промптов Prompt-Ops

Этот документ содержит полное описание всех AI промптов, используемых в системе Prompt-Ops. Промпты организованы по категориям для удобства навигации.

## Содержание

1. [PDO Meta-Prompts (Prompt Duel Optimizer)](#1-pdo-meta-prompts-prompt-duel-optimizer)
   - [Dataset Descriptor Prompt](#11-dataset-descriptor-prompt)
   - [Instruction Proposer Template](#12-instruction-proposer-template)
   - [Reason Prompt](#13-reason-prompt)
   - [Evaluate Prompt](#14-evaluate-prompt)
   - [Answer Prompt Open](#15-answer-prompt-open)
   - [Requirement Proposer Template](#16-requirement-proposer-template)
   - [Evaluate Open Prompt](#17-evaluate-open-prompt)
2. [Evaluation and Metrics Prompts](#2-evaluation-and-metrics-prompts)
3. [Use-Case Specific Prompts](#3-use-case-specific-prompts)
4. [Mutation and Optimization Tips](#4-mutation-and-optimization-tips)

---

## 1. PDO Meta-Prompts (Prompt Duel Optimizer)

Расположение: `/home/user/prompt-ops/src/prompt_ops/core/pdo/meta_prompt.py`

### 1.1. Dataset Descriptor Prompt

**Назначение:** Анализирует примеры из неразмеченного датасета и создает структурированное описание характеристик входных данных. Используется на начальном этапе PDO для понимания структуры данных перед генерацией инструкций.

**Когда используется:** В начале процесса оптимизации PDO для создания описания датасета, которое затем используется при генерации инструкций.

**Входные параметры:**
- `{examples}` - примеры неразмеченных входных данных из датасета

**Выходные данные:** Краткое описание структуры датасета (3-5 предложений)

```python
DATASET_DESCRIPTOR_PROMPT = """
You are a data analyst specialized in preparing instructions for language models.

Your task is to write a **clear, structured summary** of the unlabeled dataset based on the sample inputs provided below.

## Sample Unlabeled Inputs ##
{examples}

## Summary Guidelines ##
- Focus on input structure: Is it plain text, question-answer pairs, JSON dictionaries, CSV rows, lists, etc.?
- Identify common data fields, phrasing patterns, or key terminology present across examples.
- Describe typical input length (short phrases, single sentences, multi-step reasoning questions, etc.).
- Mention any domain-specific jargon, numbers, or named entities if present.
- Do **not** attempt to infer answers, labels, or outputs.
- Keep the summary concise: 3 to 5 sentences total.

## Your Output ##
Write a concise natural-language summary of the dataset's input characteristics, suitable for guiding an instruction proposal system.
"""
```

---

### 1.2. Instruction Proposer Template

**Назначение:** Генерирует новую высококачественную системную инструкцию для целевой задачи рассуждения. Это ключевой промпт для создания кандидатов инструкций в процессе оптимизации PDO.

**Когда используется:** При генерации новых инструкций-кандидатов на основе анализа датасета и prompt engineering советов.

**Входные параметры:**
- `{dataset_summary}` - LLM-сгенерированное описание датасета
- `{questions}` - примеры входных данных (вопросов)
- `{tip}` - совет по prompt engineering из словаря INITIAL_INSTRUCTION_TIPS
- `{base_instruction_block}` - (опционально) референсная инструкция для ориентира

**Выходные данные:** JSON массив с одной инструкцией

```python
INSTRUCTION_PROPOSER_TEMPLATE = """
You are an expert prompt‑engineer.
Your task is to generate **exactly 1** *high‑quality* **system‑level instruction** for the target reasoning task.

# Dataset Snapshot
Below is an *LM‑written summary* of the unlabeled question pool:
{dataset_summary}

# Sample Inputs (do NOT answer them)
{questions}

# Prompt‑Engineering Tip
{tip}

# Reference Instruction (Optional)
{base_instruction_block}

# Length & Style Constraints
- Avoid dataset‑specific jargon unless it appears in the sample inputs.
- Vary phrasing and level of explicit reasoning to foster exploration.

# Output Format (STRICT)
Return **exactly** 1 item in a JSON array, *and nothing else*:

[
"Your single high-quality instruction here"
]
"""
```

---

### 1.3. Reason Prompt

**Назначение:** Промпт для выполнения задачи с структурированным JSON-ответом. Заставляет модель предоставить обоснование (reasoning) и выбрать один из предложенных вариантов ответа. Используется для задач с закрытыми вопросами (close-ended tasks).

**Когда используется:** При выполнении задач типа multiple choice или binary classification, где требуется structured output с обоснованием.

**Входные параметры:**
- `{instruction}` - системная инструкция, которая тестируется
- `{question}` - вопрос или задача для решения
- `{answer_choices_str}` - строка с возможными вариантами ответа (например, "Yes, No" или "A, B, C, D")

**Выходные данные:** JSON объект с полями `reasoning` и `answer`

```python
REASON_PROMPT = """
## Instruction ##
{instruction}

## Question ##
{question}

## Output format ##
You must follow the instruction exactly as given above. Provide a brief reasoning for your answer, then select one of the following choices: {answer_choices_str}. Your response must be returned strictly in the following JSON format, with no additional text:

{{
  "reasoning": "Brief explanation of your reasoning (1-2 sentences)",
  "answer": "{answer_choices_str}"
}}

Do not include any explanation or extra output outside the JSON.
"""
```

---

### 1.4. Evaluate Prompt

**Назначение:** Промпт для беспристрастной оценки двух конкурирующих ответов на одну задачу. Судья (judge model) сравнивает ответы X и Y по критериям корректности и качества рассуждений. Используется в механизме "дуэли" PDO для определения победителя.

**Когда используется:** В процессе PDO dueling при сравнении результатов работы двух разных инструкций на одной и той же задаче.

**Входные параметры:**
- `{question}` - задача/вопрос
- `{reasoning_X}` - обоснование ответа промпта X
- `{answer_X}` - ответ промпта X
- `{reasoning_Y}` - обоснование ответа промпта Y
- `{answer_Y}` - ответ промпта Y

**Выходные данные:** JSON объект с полями `reasoning` (детальное обоснование, ~100 слов) и `winner` ("X" или "Y")

```python
EVALUATE_PROMPT = """
## Role ##
You are a meticulous, impartial referee evaluating two competing responses to determine which better satisfies the given task.

## Task ##
{question}

## Response from Prompt X ##
**Reasoning:** {reasoning_X}
**Answer:** {answer_X}

## Response from Prompt Y ##
**Reasoning:** {reasoning_Y}
**Answer:** {answer_Y}

## Evaluation Criteria ##
Evaluate both responses using the following weighted criteria:

1. **Correctness (50%)** - Does the final answer match factual reality and task requirements?
2. **Reasoning Quality (50%)** - Is the logic coherent, complete, and free of hallucinations?

## Output Instructions ##
- Provide detailed justification for your decision (~100 words)
- Select the winner: either "X" or "Y"
- Output **only** the JSON object below with **no additional text**

## Output Format ##
{{
  "reasoning": "Your detailed justification explaining why prompt X or Y produced the better response (~100 words).",
  "winner": "X or Y"
}}
"""
```

---

### 1.5. Answer Prompt Open

**Назначение:** Упрощенный промпт для open-ended задач, где не требуется структурированный JSON-ответ. Модель просто генерирует ответ в свободной форме, следуя инструкции.

**Когда используется:** Для задач открытого типа (open-ended), где ответ не может быть выражен в виде выбора из фиксированного списка вариантов.

**Входные параметры:**
- `{instruction}` - системная инструкция для задачи
- `{question}` - вопрос или задача

**Выходные данные:** Свободный текстовый ответ

```python
ANSWER_PROMPT_OPEN = """
## Instruction ##
{instruction}

## Question ##
{question}
"""
```

---

### 1.6. Requirement Proposer Template

**Назначение:** Создает критерии оценки (evaluation rubric) для сравнения двух ответов на открытые вопросы. Генерирует 5-7 критериев с весами для использования судьей при оценке open-ended задач.

**Когда используется:** В начале процесса PDO для open-ended задач, чтобы создать объективные критерии оценки качества ответов.

**Входные параметры:**
- `{dataset_summary}` - описание датасета
- `{questions}` - примеры вопросов из датасета

**Выходные данные:** Список из 5-7 bullet points, каждый в формате "- CriterionName (Weight%): actionable rule"

```python
REQUIREMENT_PROPOSER_TEMPLATE = """
You are an expert evaluation designer. Write a concise, objective rubric to compare two answers to the same question.

# Dataset Snapshot
{dataset_summary}

# Sample Inputs (do NOT answer)
{questions}

# Output (STRICT)
Return 5–7 bullet points ONLY, one per line, each in the form:
- CriterionName (Weight%): short, actionable rule.

Guidelines:
- Emphasize factual correctness and completeness relative to the question.
- Require faithfulness to provided information; penalize hallucinations.
- Prefer clarity, specificity, and relevance; penalize verbosity/off-topic content.
- Include scope adherence (no unsupported claims).
- Add a Tie-Breaker (5–10%) preferring more precise, directly stated answers when otherwise comparable.

Do not include any preamble or closing text.
"""
```

---

### 1.7. Evaluate Open Prompt

**Назначение:** Промпт для оценки двух конкурирующих ответов на open-ended задачи. Похож на EVALUATE_PROMPT, но работает с произвольными текстовыми ответами и использует custom критерии оценки.

**Когда используется:** При сравнении результатов работы двух инструкций на open-ended задачах в процессе PDO dueling.

**Входные параметры:**
- `{question}` - задача/вопрос
- `{answer_X}` - ответ от промпта X (текст)
- `{answer_Y}` - ответ от промпта Y (текст)
- `{criteria_text}` - критерии оценки, сгенерированные REQUIREMENT_PROPOSER_TEMPLATE

**Выходные данные:** JSON объект с полями `reasoning` и `winner` ("X" или "Y")

```python
EVALUATE_OPEN_PROMPT = """
## Role ##
You are a meticulous, impartial referee evaluating two competing answers to determine which better satisfies the given task.

## Task ##
{question}

## Answer from Prompt X ##
{answer_X}

## Answer from Prompt Y ##
{answer_Y}

## Evaluation Criteria ##
{criteria_text}

## Output Instructions ##
- Provide detailed justification for your decision (~100 words)
- Select the winner: either "X" or "Y"
- Output **only** the JSON object below with **no additional text**

## Output Format ##
{{
  "reasoning": "Your detailed justification explaining why prompt X or Y produced the better response (~100 words).",
  "winner": "X or Y"
}}
"""
```

---

## 2. Evaluation and Metrics Prompts

Расположение: `/home/user/prompt-ops/src/prompt_ops/core/metrics.py`

Эти промпты используются в метриках DSPyMetricAdapter для автоматической оценки качества ответов модели с помощью LLM-as-a-judge подхода.

### 2.1. Similarity Evaluation Prompt

**Назначение:** Оценивает семантическое сходство между предсказанным ответом модели и ground truth ответом. Используется для проверки, насколько ответ модели близок по смыслу к эталонному ответу.

**Когда используется:** В метриках для оценки semantic similarity, когда нужно понять степень совпадения смысла, а не точного текста.

**Входные параметры:**
- `{output}` - предсказанный ответ модели
- `{ground_truth}` - эталонный (ground truth) ответ

**Выходные данные:** Целочисленная оценка от 1 до 10

**Встроен в:** DSPyMetricAdapter с signature_name="similarity"

```python
SIMILARITY_PROMPT = """You are a smart language model that evaluates the similarity between a predicted text and the expected ground truth answer. You do not propose changes to the answer and only critically evaluate the existing answer and provide feedback following the instructions given.

The following is the response provided by a language model to a prompt:
{output}

The expected answer to this prompt is:
{ground_truth}

Answer only with an integer from 1 to 10 based on how semantically similar the responses are to the expected answer. where 1 is no semantic similarity at all and 10 is perfect agreement between the responses and the expected answer. On a NEW LINE, give the integer score and nothing more."""
```

---

### 2.2. Correctness Evaluation Prompt

**Назначение:** Оценивает корректность предсказанного ответа по сравнению с ground truth. Фокусируется на фактической правильности, а не только на семантическом сходстве.

**Когда используется:** Для оценки correctness метрики, когда важна фактическая точность ответа.

**Входные параметры:**
- `{output}` - предсказанный ответ модели
- `{ground_truth}` - эталонный (ground truth) ответ

**Выходные данные:** Целочисленная оценка от 1 до 10

**Встроен в:** DSPyMetricAdapter с signature_name="correctness"

```python
CORRECTNESS_PROMPT = """You are a smart language model that evaluates the correctness of a predicted answer compared to the expected ground truth. You do not propose changes to the answer and only critically evaluate the existing answer.

The following is the response provided by a language model to a prompt:
{output}

The expected answer to this prompt is:
{ground_truth}

Answer only with an integer from 1 to 10 based on how correct the response is compared to the expected answer, where 1 means completely incorrect and 10 means perfectly correct. On a NEW LINE, give the integer score and nothing more."""
```

---

### 2.3. DSPyMetricAdapter Configuration

**Описание:** DSPyMetricAdapter - это универсальный адаптер для создания LLM-based метрик. Он поддерживает:

- **Встроенные сигнатуры:** "similarity", "correctness"
- **Custom сигнатуры:** можно создать свои через параметры
- **Нормализация оценок:** автоматическое преобразование из одного диапазона (например, 1-10) в другой (например, 0-1)
- **Гибкий маппинг:** настраиваемое соответствие между входными данными и полями промпта

**Параметры конфигурации:**
```python
DSPyMetricAdapter(
    model=model,                    # DSPy-совместимая модель
    signature_name="similarity",    # Или "correctness"
    score_range=(1, 10),           # Ожидаемый диапазон от LLM
    normalize_to=(0, 1)            # Целевой диапазон для нормализации
)
```

---
## 3. Use-Case Specific Prompts

Эти промпты используются в конкретных примерах использования (use-cases) системы Prompt-Ops для решения различных задач.

### 3.1. HotpotQA - Multi-hop Reasoning

**Расположение:** `/home/user/prompt-ops/use-cases/hotpotqa/hotpot_qa_sys_prompt.txt`

**Назначение:** Системный промпт для задачи multi-hop question answering, где требуется рассуждение с использованием нескольких фрагментов контекста для получения ответа.

**Тип задачи:** Multi-hop QA (вопросы, требующие объединения информации из нескольких источников)

**Входные данные:**
- `question` - вопрос, требующий multi-hop рассуждения
- `context` - набор контекстных фрагментов (passages)

**Выходные данные:** Короткий фактический ответ (short factoid answer)

**Конфигурация:** `use-cases/hotpotqa/hotpotqa.yaml`, `configs/hotpotqa.yaml`

```
You are an expert at answering complex questions that require multi-hop reasoning. Give a short factoid answer.
```

---

### 3.2. MS-Marco PDO - Open-ended Question Answering

**Расположение:** `/home/user/prompt-ops/use-cases/ms-marco-pdo/prompts/prompt.txt`

**Назначение:** Промпт для open-ended QA задачи с использованием PDO оптимизации. Модель должна генерировать точный и краткий ответ своими словами.

**Тип задачи:** Open-ended question answering с PDO оптимизацией (20 раундов)

**Входные данные:**
- `question` - вопрос пользователя
- (опционально) `context` - контекст для ответа

**Выходные данные:** Краткий, точный ответ в свободной форме

**Конфигурация:** `use-cases/ms-marco-pdo/config.yaml`

**Особенности:**
- Используется PDO стратегия с 20 total rounds
- Open-ended формат (без структурированного JSON)
- Фокус на генерации ответа своими словами

```
You are an expert answerer. Read the question and the provided context (if any). Write a concise, accurate answer in your own words.
```

---

### 3.3. Web of Lies PDO - Logical Reasoning

**Расположение:** `/home/user/prompt-ops/use-cases/web-of-lies-pdo/prompts/prompt.txt`

**Назначение:** Минималистичный промпт для задач логического рассуждения в формате Web of Lies. Используется с PDO оптимизацией для улучшения accuracy.

**Тип задачи:** Close-ended logical reasoning (binary classification)

**Входные данные:**
- `question` - логическая задача или утверждение

**Выходные данные:** "Yes" или "No"

**Конфигурация:** `use-cases/web-of-lies-pdo/config.yaml`

**Особенности:**
- PDO стратегия с 30 total rounds
- Dueling bandits оптимизация
- Close-ended задача с фиксированными вариантами ответа
- Изначально минималистичный промпт, который PDO улучшает

```
Answer the following question.
```

---

### 3.4. Facility Support Analyzer - Classification & Extraction

**Расположение:** `/home/user/prompt-ops/use-cases/facility-support-analyzer/facility_prompt_sys.txt`

**Назначение:** Комплексный промпт для классификации и извлечения структурированной информации из сообщений о проблемах в facility management. Модель должна извлечь urgency, sentiment и categories из входного сообщения.

**Тип задачи:** Multi-label classification + sentiment analysis + urgency detection

**Входные данные:**
- `input` - текст сообщения от пользователя о проблеме/запросе

**Выходные данные:** JSON объект с полями:
- `urgency` - "high" | "medium" | "low"
- `sentiment` - "negative" | "neutral" | "positive"  
- `categories` - словарь с 10 возможными категориями (boolean значения)

**Конфигурация:** `configs/facility.yaml`, `configs/facility-simple.yaml`

**Категории поддержки:**
- `emergency_repair_services` - экстренный ремонт
- `routine_maintenance_requests` - плановое обслуживание
- `quality_and_safety_concerns` - проблемы качества и безопасности
- `specialized_cleaning_services` - специализированная уборка
- `general_inquiries` - общие вопросы
- `sustainability_and_environmental_practices` - экология и устойчивость
- `training_and_support_requests` - обучение и поддержка
- `cleaning_services_scheduling` - планирование уборки
- `customer_feedback_and_complaints` - отзывы и жалобы
- `facility_management_issues` - вопросы управления объектом

**Особенности:**
- Strict JSON output (без markdown форматирования)
- Multi-task: одновременная классификация по трем измерениям
- Специализированная метрика FacilityMetric для оценки

```
You are a helpful assistant. Extract and return a json with the following keys and values:
- "urgency" as one of `high`, `medium`, `low`
- "sentiment" as one of `negative`, `neutral`, `positive`
- "categories" Create a dictionary with categories as keys and boolean values (True/False), where the value indicates whether the category is one of the best matching support category tags from: `emergency_repair_services`, `routine_maintenance_requests`, `quality_and_safety_concerns`, `specialized_cleaning_services`, `general_inquiries`, `sustainability_and_environmental_practices`, `training_and_support_requests`, `cleaning_services_scheduling`, `customer_feedback_and_complaints`, `facility_management_issues`
Your complete message should be a valid json string that can be read directly and only contain the keys mentioned in the list above. Never enclose it in ```json...```, no newlines, no unnessacary whitespaces.
```

---
