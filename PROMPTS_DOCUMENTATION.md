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
