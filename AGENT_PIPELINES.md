# Схемы Работы Агента Prompt-Ops (Agent Pipelines)

Этот документ описывает все возможные пайплайны (схемы работы) агента оптимизации промптов в системе Prompt-Ops. Для каждого пайплайна приведены ASCII-диаграммы, описание этапов, используемые промпты и передаваемые данные.

## Содержание

1. [Общая Архитектура](#1-общая-архитектура)
2. [Pipeline 1: BasicOptimization (DSPy MIPROv2)](#2-pipeline-1-basicoptimization-dspy-miprov2)
3. [Pipeline 2: PDO (Prompt Duel Optimizer)](#3-pipeline-2-pdo-prompt-duel-optimizer)
4. [Pipeline 3: Evaluation Pipeline](#4-pipeline-3-evaluation-pipeline)

---

## 1. Общая Архитектура

### 1.1. Обзор системы

```
┌─────────────────────────────────────────────────────────────────┐
│                      PROMPT-OPS SYSTEM                          │
│                                                                 │
│   ┌─────────────┐                                              │
│   │   User      │                                              │
│   │   Input     │                                              │
│   └──────┬──────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────────────────────────────┐                 │
│   │     PromptMigrator (Orchestrator)       │                 │
│   │  - Loads datasets                       │                 │
│   │  - Initializes models                   │                 │
│   │  - Selects strategy                     │                 │
│   └──────────────┬──────────────────────────┘                 │
│                  │                                              │
│                  ▼                                              │
│   ┌──────────────────────────────────────────────────┐        │
│   │         Strategy Selection                       │        │
│   │                                                  │        │
│   │    ┌──────────────┐      ┌──────────────┐      │        │
│   │    │ BasicOptim   │  OR  │     PDO      │      │        │
│   │    │  Strategy    │      │   Strategy   │      │        │
│   │    └──────┬───────┘      └──────┬───────┘      │        │
│   │           │                     │              │        │
│   └───────────┼─────────────────────┼──────────────┘        │
│               │                     │                         │
│               ▼                     ▼                         │
│   ┌──────────────────┐  ┌──────────────────┐                │
│   │  DSPy MIPROv2    │  │   PDO Engine     │                │
│   │   Compiler       │  │  (Dueling        │                │
│   │                  │  │   Bandits)       │                │
│   └────────┬─────────┘  └────────┬─────────┘                │
│            │                     │                            │
│            └──────────┬──────────┘                            │
│                       │                                       │
│                       ▼                                       │
│            ┌─────────────────────┐                           │
│            │  Optimized Program  │                           │
│            │  (New Instruction)  │                           │
│            └─────────┬───────────┘                           │
│                      │                                        │
│                      ▼                                        │
│            ┌─────────────────────┐                           │
│            │  Evaluation         │                           │
│            │  (Optional)         │                           │
│            └─────────┬───────────┘                           │
│                      │                                        │
│                      ▼                                        │
│            ┌─────────────────────┐                           │
│            │  Save Results       │                           │
│            └─────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2. Компоненты системы

**Модели:**
- `task_model` - выполняет задачи (генерирует ответы)
- `prompt_model` (он же `judge_model`) - генерирует/оценивает промпты

**Датасеты:**
- `trainset` - обучающие примеры
- `valset` - валидационные примеры для выбора лучшего промпта
- `testset` - тестовые примеры для финальной оценки

**Метрики:**
- Функции оценки качества: `metric(prediction, expected) → float [0,1]`

---

## 2. Pipeline 1: BasicOptimization (DSPy MIPROv2)

### 2.1. Описание

BasicOptimization использует алгоритм MIPROv2 из библиотеки DSPy для оптимизации промптов. Это "мягкая" оптимизация, фокусирующаяся на улучшении формата и стиля инструкции.

**Ключевые особенности:**
- Использует DSPy MIPROv2 Compiler
- Генерирует кандидаты инструкций на основе анализа данных
- Опционально добавляет few-shot примеры
- Быстрая оптимизация (несколько итераций)

### 2.2. Полная схема пайплайна

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  BASICOPTIMIZATION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────┘

ВХОДНЫЕ ДАННЫЕ
┌──────────────────────────────────────────────┐
│ prompt_data = {                              │
│   "text": "Original instruction",           │
│   "inputs": ["question", "context"],        │
│   "outputs": ["answer"]                     │
│ }                                            │
│                                              │
│ trainset = [Example(question="...", ...)]  │
│ valset = [Example(question="...", ...)]    │
│ testset = [Example(question="...", ...)]   │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: ИНИЦИАЛИЗАЦИЯ                                              │
│                                                                      │
│  1.1 Создание DSPy Signature                                        │
│      ┌────────────────────────────────────┐                        │
│      │ class DynamicSignature(dspy.Signature):                      │
│      │   question = dspy.InputField()                               │
│      │   context = dspy.InputField()                                │
│      │   answer = dspy.OutputField()                                │
│      │   __doc__ = "Original instruction"                           │
│      └────────────────────────────────────┘                        │
│                                                                      │
│  1.2 Создание DSPy Program                                          │
│      program = dspy.Predict(DynamicSignature)                       │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: BASELINE COMPUTATION (Опционально)                         │
│                                                                      │
│  ┌────────────────────────────────────────────────────────┐        │
│  │ FOR each example in testset:                           │        │
│  │   prediction = program(example)                        │        │
│  │   score = metric(prediction, example.expected)         │        │
│  │                                                         │        │
│  │ baseline_score = mean(scores)                          │        │
│  │ LOG: "Baseline score: 0.65"                            │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: MIPROV2 OPTIMIZATION                                        │
│                                                                      │
│  3.1 Инициализация оптимизатора                                     │
│      ┌──────────────────────────────────────────────────┐          │
│      │ optimizer = MIPROv2(                             │          │
│      │   metric=metric_function,                        │          │
│      │   prompt_model=prompt_model,  # для генерации   │          │
│      │   task_model=task_model,      # для выполнения  │          │
│      │   auto="light",                                  │          │
│      │   num_candidates=10,           # кандидатов     │          │
│      │   max_labeled_demos=5,         # примеров       │          │
│      │   num_threads=18               # параллелизм    │          │
│      │ )                                                │          │
│      └──────────────────────────────────────────────────┘          │
│                                                                      │
│  3.2 Запуск компиляции                                              │
│      optimized_program = optimizer.compile(                         │
│        program=program,                                             │
│        trainset=trainset,                                           │
│        valset=valset                                                │
│      )                                                              │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: INSTRUCTION PROPOSAL (Внутри MIPROv2)                      │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ FOR iteration in num_trials:                                │  │
│  │                                                              │  │
│  │   4.1 Анализ датасета                                       │  │
│  │       ┌───────────────────────────────────────────────┐     │  │
│  │       │ Prompt: DATASET_DESCRIPTOR_PROMPT             │     │  │
│  │       │ Input:  3 sample examples                     │     │  │
│  │       │ Output: dataset_summary                       │     │  │
│  │       │         "The dataset contains questions..."   │     │  │
│  │       └───────────────────────────────────────────────┘     │  │
│  │                                                              │  │
│  │   4.2 Генерация кандидатов (num_candidates раз)            │  │
│  │       ┌───────────────────────────────────────────────┐     │  │
│  │       │ Prompt: INSTRUCTION_PROPOSER_TEMPLATE         │     │  │
│  │       │                                               │     │  │
│  │       │ Входные данные:                               │     │  │
│  │       │   {dataset_summary}    <- из 4.1             │     │  │
│  │       │   {questions}          <- 3 examples         │     │  │
│  │       │   {tip}                <- "framing" tip      │     │  │
│  │       │   {base_instruction}   <- original prompt    │     │  │
│  │       │                                               │     │  │
│  │       │ Выходные данные:                              │     │  │
│  │       │   ["You are an expert. Given the context..."] │     │  │
│  │       └───────────────────────────────────────────────┘     │  │
│  │                                                              │  │
│  │       Результат: candidates = [instr1, instr2, ..., instr10]│  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 5: CANDIDATE EVALUATION & SELECTION                           │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ FOR each candidate_instruction in candidates:               │  │
│  │                                                              │  │
│  │   5.1 Создание программы с кандидатом                       │  │
│  │       program_i = dspy.Predict(                             │  │
│  │         Signature(__doc__=candidate_instruction)            │  │
│  │       )                                                      │  │
│  │                                                              │  │
│  │   5.2 Оценка на валидационном наборе                        │  │
│  │       ┌──────────────────────────────────────────┐          │  │
│  │       │ FOR example in valset:                   │          │  │
│  │       │   prediction = program_i(example)        │          │  │
│  │       │   score = metric(prediction, expected)   │          │  │
│  │       │                                           │          │  │
│  │       │ val_score_i = mean(scores)               │          │  │
│  │       └──────────────────────────────────────────┘          │  │
│  │                                                              │  │
│  │   5.3 Сохранение результата                                 │  │
│  │       scores.append((candidate_i, val_score_i))            │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  5.4 Выбор лучшего кандидата                                        │
│      best_instruction = max(scores, key=lambda x: x[1])[0]          │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 6: FEW-SHOT LEARNING (Опционально)                            │
│                                                                      │
│  IF max_labeled_demos > 0:                                           │
│                                                                      │
│    6.1 Выбор лучших примеров из trainset                            │
│        ┌────────────────────────────────────────────┐              │
│        │ FOR example in trainset:                   │              │
│        │   prediction = optimized_program(example)  │              │
│        │   score = metric(prediction, expected)     │              │
│        │                                             │              │
│        │ demos = top_k_examples(scores, k=5)        │              │
│        └────────────────────────────────────────────┘              │
│                                                                      │
│    6.2 Добавление демо к программе                                  │
│        optimized_program.demos = demos                              │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ВЫХОДНЫЕ ДАННЫЕ                                                      │
│                                                                      │
│  optimized_program = {                                              │
│    signature: {                                                     │
│      __doc__: "You are an expert. Given the question and context,  │
│                provide a clear, well-reasoned answer. Focus on...  │
│                [OPTIMIZED INSTRUCTION]",                            │
│      fields: [question, context, answer]                           │
│    },                                                               │
│    demos: [                                                         │
│      Example(question="...", context="...", answer="..."),         │
│      Example(question="...", context="...", answer="..."),         │
│      ...  # up to 5 examples                                       │
│    ]                                                                │
│  }                                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3. Передача данных между этапами

```
STAGE 1 → STAGE 2:
  program (DSPy.Predict)

STAGE 2 → STAGE 3:
  baseline_score (float)
  program (DSPy.Predict)

STAGE 3 → STAGE 4:
  optimizer (MIPROv2)
  trainset, valset

STAGE 4 → STAGE 5:
  candidates (List[str]) - список инструкций-кандидатов
  dataset_summary (str) - описание датасета

STAGE 5 → STAGE 6:
  best_instruction (str) - лучшая инструкция
  program with best_instruction

STAGE 6 → OUTPUT:
  optimized_program с demos
```

### 2.4. Используемые промпты

| Этап | Промпт | Вход | Выход |
|------|--------|------|-------|
| 4.1 | DATASET_DESCRIPTOR_PROMPT | 3 sample examples | dataset_summary (строка) |
| 4.2 | INSTRUCTION_PROPOSER_TEMPLATE | dataset_summary, questions, tip, base_instruction | ["новая инструкция"] |

### 2.5. Параметры конфигурации

```python
{
  "auto": "basic",              # интенсивность оптимизации
  "num_candidates": 10,         # кандидатов на итерацию
  "max_labeled_demos": 5,       # максимум примеров
  "max_bootstrapped_demos": 4,  # авто-генерированных примеров
  "num_threads": 18,            # параллельных потоков
  "compute_baseline": True,     # вычислять baseline
  "num_trials": 3               # итераций оптимизации
}
```

---
