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
## 3. Pipeline 2: PDO (Prompt Duel Optimizer)

### 3.1. Описание

PDO (Prompt Duel Optimizer) - это продвинутый алгоритм оптимизации промптов через механизм "дуэлей". Инструкции соревнуются друг с другом на одних и тех же задачах, и судья (judge model) определяет победителя. Система использует Thompson Sampling для выбора пар для дуэлей и различные ranking системы для итоговой оценки.

**Ключевые особенности:**
- Эволюционный подход с мутациями
- Thompson Sampling для балансировки exploration/exploitation
- Параллельное выполнение задач и оценки
- Множественные ranking системы (Copeland, Elo, TrueSkill и др.)
- Поддержка close-ended и open-ended задач

### 3.2. Общая схема пайплайна

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PDO PIPELINE                                     │
└─────────────────────────────────────────────────────────────────────────┘

                           ╔═══════════════════╗
                           ║  PHASE 1: INIT    ║
                           ╚═══════════════════╝
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │   Dataset    │ │  Generate    │ │  Initialize  │
            │  Descriptor  │ │  Initial     │ │  Win Matrix  │
            │              │ │  Instruction │ │  & Rankings  │
            │              │ │  Pool        │ │              │
            └──────────────┘ └──────────────┘ └──────────────┘
                                     │
                           ╔═══════════════════╗
                           ║  PHASE 2: LOOP    ║
                           ║  (100 rounds)     ║
                           ╚═══════════════════╝
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │   Thompson   │ │     Task     │ │    Judge     │
            │   Sampling   │ │  Execution   │ │  Evaluation  │
            │  (Select     │ │  (Parallel)  │ │  (Parallel)  │
            │   Pairs)     │ │              │ │              │
            └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                   │                │                │
                   └────────────────┼────────────────┘
                                    ▼
                            ┌──────────────┐
                            │ Win Matrix   │
                            │   Update     │
                            └──────┬───────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼                             ▼
            ┌──────────────┐              ┌──────────────┐
            │   Prune      │              │   Mutate     │
            │   Worst      │              │   Best       │
            │ Instructions │              │ Instructions │
            └──────────────┘              └──────────────┘
                                   │
                           ╔═══════════════════╗
                           ║  PHASE 3: FINAL   ║
                           ╚═══════════════════╝
                                   │
                            ┌──────────────┐
                            │    Rank      │
                            │     All      │
                            │ Instructions │
                            └──────┬───────┘
                                   │
                            ┌──────────────┐
                            │   Return     │
                            │   Champion   │
                            └──────────────┘
```

### 3.3. Фаза 1: Инициализация

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: INITIALIZATION                                                 │
└─────────────────────────────────────────────────────────────────────────┘

ВХОДНЫЕ ДАННЫЕ:
  base_instruction: "Answer the following question."
  examples: [q1, q2, q3, ..., qN]
  labels: [a1, a2, a3, ..., aN] (optional)

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1.1: Генерация описания датасета                                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ Prompt: DATASET_DESCRIPTOR_PROMPT                            │     │
│  │                                                               │     │
│  │ Входные данные:                                              │     │
│  │   {examples} = [                                             │     │
│  │     "What is the capital of France?",                        │     │
│  │     "Who wrote Romeo and Juliet?",                           │     │
│  │     "What is 2+2?"                                           │     │
│  │   ]                                                          │     │
│  │                                                               │     │
│  │ Model: prompt_model.generate(temp=0.7)                       │     │
│  │                                                               │     │
│  │ Выходные данные (dataset_summary):                           │     │
│  │   "The dataset contains short factual questions              │     │
│  │    requiring knowledge-based answers. Questions span         │     │
│  │    geography, literature, and mathematics domains."          │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1.2: Генерация начального пула инструкций                         │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ FOR i = 1 to num_initial_instructions (default: 2):         │     │
│  │                                                               │     │
│  │   1. Выбор tip из INITIAL_INSTRUCTION_TIPS                   │     │
│  │      tips = ["framing", "simple", "description",             │     │
│  │               "persona", "relevance", "completeness",        │     │
│  │               "clarity", "evidence"]                         │     │
│  │      selected_tip = tips[i % 8]                              │     │
│  │                                                               │     │
│  │   2. Создание промпта                                        │     │
│  │      ┌────────────────────────────────────────────────┐     │     │
│  │      │ Prompt: INSTRUCTION_PROPOSER_TEMPLATE          │     │     │
│  │      │                                                 │     │     │
│  │      │ Входные данные:                                │     │     │
│  │      │   {dataset_summary}    <- из Step 1.1         │     │     │
│  │      │   {questions}          <- 3 random examples   │     │     │
│  │      │   {tip}                <- selected_tip        │     │     │
│  │      │   {base_instruction}   <- base_instruction    │     │     │
│  │      │                                                 │     │     │
│  │      │ Model: prompt_model.generate(temp=0.7)         │     │     │
│  │      │                                                 │     │     │
│  │      │ Выходные данные:                               │     │     │
│  │      │   ["You are a knowledgeable assistant..."]    │     │     │
│  │      └────────────────────────────────────────────────┘     │     │
│  │                                                               │     │
│  │   3. Добавление в пул                                        │     │
│  │      instruction_pool.append(new_instruction)               │     │
│  │                                                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  Также добавляется base_instruction в пул                               │
│  instruction_pool.append(base_instruction)                             │
│                                                                         │
│  Результат:                                                             │
│    instruction_pool = [instr_0, instr_1, ..., base_instruction]       │
│    Размер пула: num_initial_instructions + 1                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1.3: Инициализация Win Matrix и Ranking                           │
│                                                                         │
│  win_matrix[i][j] = количество побед инструкции i над j                │
│                                                                         │
│  ┌────────────────────────────────────────────────┐                   │
│  │ Initial state:                                 │                   │
│  │                                                 │                   │
│  │      instr_0  instr_1  instr_2                 │                   │
│  │  ┌─────────────────────────────────┐           │                   │
│  │  │    0        0        0     │ instr_0       │                   │
│  │  │    0        0        0     │ instr_1       │                   │
│  │  │    0        0        0     │ instr_2       │                   │
│  │  └─────────────────────────────────┘           │                   │
│  │                                                 │                   │
│  │  Все значения = 0 (дуэлей еще не было)        │                   │
│  └────────────────────────────────────────────────┘                   │
│                                                                         │
│  Инициализация ranking систем:                                         │
│    - Copeland: scores = [0, 0, 0, ...]                                │
│    - Elo: ratings = [1500, 1500, 1500, ...]                           │
│    - TrueSkill: mu=[25, 25, ...], sigma=[8.33, 8.33, ...]            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

### 3.4. Фаза 2: Основной цикл оптимизации

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: MAIN OPTIMIZATION LOOP                                         │
│ FOR round = 1 to total_rounds (default: 100):                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2.1: Выбор пар для дуэлей (Thompson Sampling)                     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ Thompson Sampling Algorithm:                                 │     │
│  │                                                               │     │
│  │ FOR duel_num = 1 to num_duels_per_round (default: 3):       │     │
│  │                                                               │     │
│  │   A. Вычисление confidence bounds                            │     │
│  │      ┌────────────────────────────────────────────────┐     │     │
│  │      │ FOR each pair (i, j):                          │     │     │
│  │      │   wins_i = win_matrix[i][j]                    │     │     │
│  │      │   wins_j = win_matrix[j][i]                    │     │     │
│  │      │   total = wins_i + wins_j                      │     │     │
│  │      │                                                 │     │     │
│  │      │   IF total > 0:                                │     │     │
│  │      │     empirical_prob = wins_i / total            │     │     │
│  │      │   ELSE:                                         │     │     │
│  │      │     empirical_prob = 0.5                       │     │     │
│  │      │                                                 │     │     │
│  │      │   delta = sqrt(alpha * log(round) / max(total, 1))   │     │
│  │      │   upper_bound[i,j] = empirical_prob + delta   │     │     │
│  │      │   lower_bound[i,j] = empirical_prob - delta   │     │     │
│  │      └────────────────────────────────────────────────┘     │     │
│  │                                                               │     │
│  │   B. Thompson sampling для первой инструкции                 │     │
│  │      ┌────────────────────────────────────────────────┐     │     │
│  │      │ FOR each pair (i, j):                          │     │     │
│  │      │   theta[i,j] ~ Beta(wins_i + 1, wins_j + 1)   │     │     │
│  │      │                                                 │     │     │
│  │      │ Compute Copeland scores:                       │     │     │
│  │      │   FOR i in instruction_pool:                   │     │     │
│  │      │     copeland[i] = sum(theta[i,j] > 0.5        │     │     │
│  │      │                      for all j != i)           │     │     │
│  │      │                                                 │     │     │
│  │      │ first = argmax(copeland[i]                     │     │     │
│  │      │               for i in candidate_set)          │     │     │
│  │      └────────────────────────────────────────────────┘     │     │
│  │                                                               │     │
│  │   C. Thompson sampling для второй инструкции                 │     │
│  │      ┌────────────────────────────────────────────────┐     │     │
│  │      │ FOR each opponent k != first:                  │     │     │
│  │      │   wins_k = win_matrix[k][first]                │     │     │
│  │      │   wins_f = win_matrix[first][k]                │     │     │
│  │      │   theta2[k] ~ Beta(wins_k + 1, wins_f + 1)    │     │     │
│  │      │                                                 │     │     │
│  │      │ Filter: keep only k where lower_bound[k,first] <= 0.5     │
│  │      │ second = argmax(theta2[k])                     │     │     │
│  │      └────────────────────────────────────────────────┘     │     │
│  │                                                               │     │
│  │   Result: duel_pair = (first, second)                       │     │
│  │                                                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  Результат Step 2.1:                                                    │
│    duel_pairs = [(i1, j1), (i2, j2), (i3, j3)]                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2.2: Выполнение задач (Task Execution) - ПАРАЛЛЕЛЬНО              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ FOR each duel (instr_i, instr_j) in duel_pairs:             │     │
│  │                                                               │     │
│  │   Sample num_eval_examples_per_duel examples (default: 50)  │     │
│  │   selected_examples = random.sample(examples, 50)           │     │
│  │                                                               │     │
│  │   FOR each example_k in selected_examples:                   │     │
│  │                                                               │     │
│  │     A. Создание промпта для инструкции i                    │     │
│  │        ┌────────────────────────────────────────────────┐   │     │
│  │        │ Prompt: REASON_PROMPT (для close-ended)        │   │     │
│  │        │     OR: ANSWER_PROMPT_OPEN (для open-ended)   │   │     │
│  │        │                                                 │   │     │
│  │        │ Входные данные:                                │   │     │
│  │        │   {instruction} = instruction_pool[i]          │   │     │
│  │        │   {question}    = example_k                    │   │     │
│  │        │   {answer_choices_str} = "Yes, No"  (если close) │   │     │
│  │        │                                                 │   │     │
│  │        │ Пример заполненного промпта:                   │   │     │
│  │        │   ## Instruction ##                            │   │     │
│  │        │   You are a knowledgeable assistant...         │   │     │
│  │        │                                                 │   │     │
│  │        │   ## Question ##                               │   │     │
│  │        │   What is the capital of France?               │   │     │
│  │        │                                                 │   │     │
│  │        │   ## Output format ##                          │   │     │
│  │        │   You must return JSON: {"reasoning": "...", "answer": "..."}   │
│  │        └────────────────────────────────────────────────┘   │     │
│  │                                                               │     │
│  │     B. Создание промпта для инструкции j (аналогично)       │     │
│  │                                                               │     │
│  │     C. Параллельное выполнение                               │     │
│  │        task_model.generate_batch(                            │     │
│  │          prompts=[prompt_i, prompt_j],                       │     │
│  │          max_threads=8                                       │     │
│  │        )                                                      │     │
│  │                                                               │     │
│  │        Для close-ended с JSON schema enforcement             │     │
│  │                                                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  Выходные данные (для каждого примера):                                │
│    response_i = {                                                       │
│      "reasoning": "Paris is the capital and largest city of France.",  │
│      "answer": "Paris"                                                  │
│    }                                                                    │
│    response_j = {                                                       │
│      "reasoning": "The capital city of France is Paris.",              │
│      "answer": "Paris"                                                  │
│    }                                                                    │
│                                                                         │
│  Результат Step 2.2:                                                    │
│    all_responses = [                                                    │
│      (instr_i, instr_j, example_k, response_i, response_j),           │
│      ...  # 50 примеров на дуэль × 3 дуэли = 150 записей              │
│    ]                                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2.3: Оценка судьей (Judge Evaluation) - ПАРАЛЛЕЛЬНО               │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ FOR each (instr_i, instr_j, example_k, resp_i, resp_j):     │     │
│  │                                                               │     │
│  │   A. Случайная перестановка (предотвращение позиционной bias)     │
│  │      IF random() < 0.5:                                      │     │
│  │        swap(resp_i, resp_j)                                  │     │
│  │        swapped = True                                        │     │
│  │      ELSE:                                                    │     │
│  │        swapped = False                                       │     │
│  │                                                               │     │
│  │   B. Создание промпта для судьи                              │     │
│  │      ┌────────────────────────────────────────────────┐     │     │
│  │      │ Prompt: EVALUATE_PROMPT (для close-ended)      │     │     │
│  │      │     OR: EVALUATE_OPEN_PROMPT (для open-ended) │     │     │
│  │      │                                                 │     │     │
│  │      │ Входные данные:                                │     │     │
│  │      │   {question}     = example_k                   │     │     │
│  │      │   {reasoning_X}  = resp_i["reasoning"]         │     │     │
│  │      │   {answer_X}     = resp_i["answer"]            │     │     │
│  │      │   {reasoning_Y}  = resp_j["reasoning"]         │     │     │
│  │      │   {answer_Y}     = resp_j["answer"]            │     │     │
│  │      │                                                 │     │     │
│  │      │ Пример заполненного промпта:                   │     │     │
│  │      │   ## Role ##                                    │     │     │
│  │      │   You are a meticulous, impartial referee...   │     │     │
│  │      │                                                 │     │     │
│  │      │   ## Task ##                                    │     │     │
│  │      │   What is the capital of France?               │     │     │
│  │      │                                                 │     │     │
│  │      │   ## Response from Prompt X ##                 │     │     │
│  │      │   **Reasoning:** Paris is the capital...       │     │     │
│  │      │   **Answer:** Paris                            │     │     │
│  │      │                                                 │     │     │
│  │      │   ## Response from Prompt Y ##                 │     │     │
│  │      │   **Reasoning:** The capital city is Paris...  │     │     │
│  │      │   **Answer:** Paris                            │     │     │
│  │      │                                                 │     │     │
│  │      │   ## Evaluation Criteria ##                    │     │     │
│  │      │   1. Correctness (50%)                         │     │     │
│  │      │   2. Reasoning Quality (50%)                   │     │     │
│  │      │                                                 │     │     │
│  │      │   ## Output Format ##                          │     │     │
│  │      │   {"reasoning": "...", "winner": "X or Y"}    │     │     │
│  │      └────────────────────────────────────────────────┘     │     │
│  │                                                               │     │
│  │   C. Выполнение оценки                                       │     │
│  │      judge_model.generate(prompt, temp=0.0)                 │     │
│  │      С JSON schema enforcement                               │     │
│  │                                                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  Выходные данные (для каждого сравнения):                              │
│    judgment = {                                                         │
│      "reasoning": "Both responses are correct and provide the same...",│
│      "winner": "X"  # или "Y"                                          │
│    }                                                                    │
│                                                                         │
│  Результат Step 2.3:                                                    │
│    all_judgments = [                                                    │
│      (instr_i, instr_j, winner, swapped),                             │
│      ...  # 150 judgments                                              │
│    ]                                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2.4: Обновление Win Matrix                                        │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ FOR each judgment (instr_i, instr_j, winner, swapped):      │     │
│  │                                                               │     │
│  │   IF swapped:                                                │     │
│  │     # X соответствует j, Y соответствует i                  │     │
│  │     IF winner == "X":                                        │     │
│  │       win_matrix[j][i] += 1                                  │     │
│  │     ELSE:                                                     │     │
│  │       win_matrix[i][j] += 1                                  │     │
│  │   ELSE:                                                       │     │
│  │     # X соответствует i, Y соответствует j                  │     │
│  │     IF winner == "X":                                        │     │
│  │       win_matrix[i][j] += 1                                  │     │
│  │     ELSE:                                                     │     │
│  │       win_matrix[j][i] += 1                                  │     │
│  │                                                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  Пример состояния после нескольких раундов:                            │
│  ┌────────────────────────────────────────────────┐                   │
│  │      instr_0  instr_1  instr_2                 │                   │
│  │  ┌─────────────────────────────────┐           │                   │
│  │  │    0       35       42     │ instr_0       │                   │
│  │  │   15        0       28     │ instr_1       │                   │
│  │  │    8       22        0     │ instr_2       │                   │
│  │  └─────────────────────────────────┘           │                   │
│  │                                                 │                   │
│  │  Интерпретация:                                │                   │
│  │  - instr_0 победил instr_1: 35 раз            │                   │
│  │  - instr_1 победил instr_0: 15 раз            │                   │
│  │  - instr_0 выглядит сильнее                    │                   │
│  └────────────────────────────────────────────────┘                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2.5: Обновление пула (каждые N раундов)                           │
│                                                                         │
│  IF round % gen_new_prompt_round_frequency == 0:                       │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ A. PRUNING (Удаление худших)                                │     │
│  │                                                               │     │
│  │    1. Вычислить текущий ranking (используя ranking_method)   │     │
│  │       copeland_scores = compute_copeland(win_matrix)         │     │
│  │       ranked_indices = argsort(copeland_scores, reverse=True)│     │
│  │                                                               │     │
│  │    2. Удалить num_to_prune_each_round худших (default: 1)   │     │
│  │       worst_indices = ranked_indices[-1:]                    │     │
│  │       instruction_pool = remove(instruction_pool, worst_indices)    │
│  │       win_matrix = remove_rows_cols(win_matrix, worst_indices)      │
│  │                                                               │     │
│  │    Пример:                                                    │     │
│  │    Before: [instr_0, instr_1, instr_2] (3 instructions)     │     │
│  │    After:  [instr_0, instr_1]          (2 instructions)     │     │
│  │                                                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ B. MUTATION (Генерация новых из лучших)                     │     │
│  │                                                               │     │
│  │    1. Выбрать num_top_prompts_to_combine лучших (default: 3)│     │
│  │       top_indices = ranked_indices[:3]                       │     │
│  │       champions = [instruction_pool[i] for i in top_indices] │     │
│  │                                                               │     │
│  │    2. Для каждого чемпиона сгенерировать мутацию            │     │
│  │       FOR champion in champions[:num_new_prompts_to_generate]:     │
│  │                                                               │     │
│  │         ┌────────────────────────────────────────────────┐   │     │
│  │         │ Prompt: MUTATE_PROMPT_TEMPLATE                 │   │     │
│  │         │     OR: MUTATE_PROMPT_TEMPLATE_WITH_LABELS    │   │     │
│  │         │                                                 │   │     │
│  │         │ Входные данные:                                │   │     │
│  │         │   {instructions}  = champion                   │   │     │
│  │         │   {tip}           = random.choice(             │   │     │
│  │         │                       MUTATION_TIPS.values())  │   │     │
│  │         │                     # "expansion", "minimal",  │   │     │
│  │         │                     #  "few_shot", "emphasis"  │   │     │
│  │         │   {sample_pairs}  = [                          │   │     │
│  │         │     (q1, a1), (q2, a2), ...                    │   │     │
│  │         │   ] (if use_labels)                            │   │     │
│  │         │                                                 │   │     │
│  │         │ Model: prompt_model.generate(temp=0.7)         │   │     │
│  │         │                                                 │   │     │
│  │         │ Выходные данные:                               │   │     │
│  │         │   {"mutated_prompt": "Enhanced instruction..."}│   │     │
│  │         └────────────────────────────────────────────────┘   │     │
│  │                                                               │     │
│  │         instruction_pool.append(mutated_instruction)         │     │
│  │                                                               │     │
│  │    3. Расширить win_matrix нулями для новых инструкций      │     │
│  │       win_matrix = expand_with_zeros(win_matrix, new_count) │     │
│  │                                                               │     │
│  │    Пример:                                                    │     │
│  │    Before: [instr_0, instr_1]          (2 instructions)     │     │
│  │    After:  [instr_0, instr_1, mutant1] (3 instructions)     │     │
│  │                                                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

END OF ROUND
(повторяется total_rounds раз, default: 100)

### 3.5. Фаза 3: Финальный выбор

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: FINAL SELECTION                                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3.1: Финальное ранжирование                                       │
│                                                                         │
│  Используется ranking_method (default: "copeland")                     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ МЕТОД 1: Copeland Ranking                                    │     │
│  │                                                               │     │
│  │  FOR each instruction i:                                     │     │
│  │    wins = 0                                                   │     │
│  │    losses = 0                                                 │     │
│  │    winrate_sum = 0                                            │     │
│  │                                                               │     │
│  │    FOR each opponent j (j != i):                             │     │
│  │      total_matches = win_matrix[i][j] + win_matrix[j][i]    │     │
│  │      IF total_matches > 0:                                   │     │
│  │        IF win_matrix[i][j] > win_matrix[j][i]:              │     │
│  │          wins += 1                                            │     │
│  │        ELSE:                                                  │     │
│  │          losses += 1                                          │     │
│  │        winrate_sum += win_matrix[i][j] / total_matches      │     │
│  │                                                               │     │
│  │    copeland_score[i] = wins - losses                        │     │
│  │    avg_winrate[i] = winrate_sum / num_opponents             │     │
│  │                                                               │     │
│  │  Сортировка:                                                 │     │
│  │    primary: по copeland_score (desc)                         │     │
│  │    tie-breaker: по avg_winrate (desc)                        │     │
│  │                                                               │     │
│  │  Пример:                                                      │     │
│  │    instr_0: copeland=+2, winrate=0.72 → Rank 1              │     │
│  │    instr_1: copeland=+1, winrate=0.65 → Rank 2              │     │
│  │    instr_2: copeland=-1, winrate=0.48 → Rank 3              │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ МЕТОД 2: Elo Rating                                          │     │
│  │                                                               │     │
│  │  Начальный рейтинг: 1500 для всех                            │     │
│  │  K-factor: 32                                                 │     │
│  │                                                               │     │
│  │  Обновляется после каждого матча:                            │     │
│  │    expected_i = 1 / (1 + 10^((elo_j - elo_i) / 400))       │     │
│  │    elo_i_new = elo_i + K * (actual_score - expected_i)      │     │
│  │                                                               │     │
│  │  Финальная сортировка по elo (desc)                          │     │
│  │                                                               │     │
│  │  Пример:                                                      │     │
│  │    instr_0: elo=1687 → Rank 1                               │     │
│  │    instr_1: elo=1542 → Rank 2                               │     │
│  │    instr_2: elo=1371 → Rank 3                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ МЕТОД 3: TrueSkill (Байесовский)                            │     │
│  │                                                               │     │
│  │  Начальные параметры:                                         │     │
│  │    mu = 25.0 (skill estimate)                                │     │
│  │    sigma = 8.333 (uncertainty)                               │     │
│  │                                                               │     │
│  │  Conservative skill estimate:                                │     │
│  │    conservative_skill[i] = mu[i] - 3 * sigma[i]             │     │
│  │                                                               │     │
│  │  Сортировка по conservative_skill (desc)                     │     │
│  │                                                               │     │
│  │  Пример:                                                      │     │
│  │    instr_0: mu=28.5, sigma=2.1, score=22.2 → Rank 1         │     │
│  │    instr_1: mu=26.0, sigma=2.5, score=18.5 → Rank 2         │     │
│  │    instr_2: mu=24.0, sigma=3.0, score=15.0 → Rank 3         │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ МЕТОД 4: Aggregate Ranking (Комбинированный)                │     │
│  │                                                               │     │
│  │  Использует 5 ранкеров:                                      │     │
│  │    1. Copeland                                                │     │
│  │    2. Borda (sum of winrates)                                │     │
│  │    3. Average winrate                                         │     │
│  │    4. Elo                                                     │     │
│  │    5. TrueSkill                                               │     │
│  │                                                               │     │
│  │  FOR each ranker:                                             │     │
│  │    получить позиции [pos_0, pos_1, pos_2, ...]              │     │
│  │    Например: [1, 3, 2] означает instr_0 на 1 месте и т.д.  │     │
│  │                                                               │     │
│  │  Применить Borda count к позициям:                           │     │
│  │    FOR each instruction i:                                   │     │
│  │      borda_score[i] = sum(                                   │     │
│  │        n - position_in_ranker[i]                             │     │
│  │        for each ranker                                       │     │
│  │      )                                                        │     │
│  │                                                               │     │
│  │  Сортировка по borda_score (desc)                            │     │
│  │                                                               │     │
│  │  Пример:                                                      │     │
│  │    instr_0: всегда в топе → высокий borda → Rank 1          │     │
│  │    instr_1: средние позиции → средний borda → Rank 2        │     │
│  │    instr_2: часто внизу → низкий borda → Rank 3             │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3.2: Выбор чемпиона                                                │
│                                                                         │
│  best_index = ranked_indices[0]                                         │
│  best_instruction = instruction_pool[best_index]                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ ВЫХОДНЫЕ ДАННЫЕ                                                         │
│                                                                         │
│  {                                                                      │
│    "best_instruction": "You are an expert answerer. When given a       │
│                         question, carefully analyze the context and    │
│                         provide a precise, evidence-based answer.      │
│                         Focus on clarity and completeness.",           │
│                                                                         │
│    "metadata": {                                                        │
│      "best_instruction_index": 5,                                      │
│      "total_instructions_generated": 47,                               │
│      "total_duels_conducted": 300,                                     │
│      "final_pool_size": 8,                                             │
│      "ranking_method": "copeland",                                     │
│      "final_win_matrix": [[...], [...], ...],                         │
│      "instruction_pool": [                                             │
│        "Instruction 0...",                                             │
│        "Instruction 1...",                                             │
│        ...                                                             │
│      ]                                                                 │
│    }                                                                   │
│  }                                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.6. Передача данных между фазами

```
PHASE 1 → PHASE 2:
  instruction_pool (List[str]) - пул инструкций (размер: 3-10)
  win_matrix (2D array) - матрица побед (инициализирована нулями)
  dataset_summary (str) - описание датасета
  examples (List) - примеры для оценки

PHASE 2 (каждая итерация):
  Step 2.1 → Step 2.2:
    duel_pairs (List[(int, int)]) - пары индексов для дуэлей

  Step 2.2 → Step 2.3:
    all_responses (List) - все ответы task_model
    Format: [(instr_i, instr_j, example_k, response_i, response_j), ...]

  Step 2.3 → Step 2.4:
    all_judgments (List) - все вердикты judge_model
    Format: [(instr_i, instr_j, winner, swapped), ...]

  Step 2.4 → Step 2.5:
    win_matrix (обновленная матрица побед)

  Step 2.5 → Next Round:
    instruction_pool (обновленный пул: худшие удалены, новые добавлены)
    win_matrix (обновленная матрица с новыми размерами)

PHASE 2 → PHASE 3:
  instruction_pool (финальный пул после всех раундов)
  win_matrix (финальная матрица побед)

PHASE 3 → OUTPUT:
  best_instruction (str) - лучшая инструкция
  metadata (dict) - статистика оптимизации
```

### 3.7. Используемые промпты и их порядок

| Фаза | Этап | Промпт | Входные данные | Выходные данные |
|------|------|--------|----------------|-----------------|
| 1 | 1.1 | DATASET_DESCRIPTOR_PROMPT | 3 sample examples | dataset_summary |
| 1 | 1.2 | INSTRUCTION_PROPOSER_TEMPLATE | dataset_summary, questions, tip, base_instruction | ["new instruction"] |
| 2 | 2.2 | REASON_PROMPT | instruction, question, answer_choices | {"reasoning": "...", "answer": "..."} |
| 2 | 2.2 | ANSWER_PROMPT_OPEN | instruction, question | free-form answer text |
| 2 | 2.3 | EVALUATE_PROMPT | question, reasoning_X, answer_X, reasoning_Y, answer_Y | {"winner": "X/Y", "reasoning": "..."} |
| 2 | 2.3 | EVALUATE_OPEN_PROMPT | question, answer_X, answer_Y, criteria | {"winner": "X/Y", "reasoning": "..."} |
| 2 | 2.5 | MUTATE_PROMPT_TEMPLATE | instructions (champion), tip | {"mutated_prompt": "..."} |
| 2 | 2.5 | MUTATE_PROMPT_TEMPLATE_WITH_LABELS | instructions, tip, sample_pairs | {"mutated_prompt": "..."} |
| 3 | 3.1 | REQUIREMENT_PROPOSER_TEMPLATE (для open-ended, в начале) | dataset_summary, questions | evaluation criteria |

### 3.8. Параметры конфигурации

```python
{
  # Основные параметры
  "total_rounds": 100,                      # раундов оптимизации
  "num_duels_per_round": 3,                 # дуэлей на раунд
  "num_eval_examples_per_duel": 50,        # примеров на дуэль
  
  # Инициализация
  "num_initial_instructions": 2,            # начальный размер пула
  
  # Thompson Sampling
  "thompson_alpha": 2.0,                    # множитель для confidence bounds
  
  # Обновление пула
  "gen_new_prompt_round_frequency": 1,      # как часто обновлять пул
  "num_top_prompts_to_combine": 3,         # топ инструкций для мутации
  "num_new_prompts_to_generate": 1,        # новых мутаций за раз
  "num_to_prune_each_round": 1,            # удалять худших
  
  # Параллелизм
  "max_concurrent_threads": 8,              # параллельных потоков
  
  # Ранжирование
  "ranking_method": "copeland",             # "copeland", "elo", "trueskill", "aggregate"
  
  # Тип задачи
  "task_type": "close_ended",               # "close_ended" или "open_ended"
  "use_labels": False                       # использовать ли labels в мутации
}
```

### 3.9. Варианты задач

#### Close-ended Tasks
- Используют REASON_PROMPT с заданными вариантами ответа
- JSON schema enforcement для структурированного вывода
- Судья использует EVALUATE_PROMPT
- Критерии: Correctness (50%) + Reasoning Quality (50%)
- Примеры: Yes/No вопросы, Multiple choice

#### Open-ended Tasks
- Используют ANSWER_PROMPT_OPEN без ограничений на формат
- Свободная генерация текста
- Судья использует EVALUATE_OPEN_PROMPT с динамическими критериями
- Критерии генерируются REQUIREMENT_PROPOSER_TEMPLATE в начале
- Примеры: Генерация текста, открытые вопросы

---
## 4. Pipeline 3: Evaluation Pipeline

### 4.1. Описание

Evaluation Pipeline используется для оценки качества оптимизированных промптов на тестовых данных. Система поддерживает базовую оценку и статистическую оценку с доверительными интервалами.

**Ключевые особенности:**
- Метрики для оценки quality predictions
- Параллельное выполнение на тестовом наборе
- Статистическая оценка с confidence intervals
- Поддержка custom метрик

### 4.2. Полная схема пайплайна

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     EVALUATION PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────┘

ВХОДНЫЕ ДАННЫЕ
┌──────────────────────────────────────────────┐
│ optimized_program: DSPy Program              │
│ testset: List[Example]                       │
│ metric: Callable[[pred, expected], float]   │
│ num_threads: int (default: 10)              │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: SETUP                                                          │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ Create Evaluator:                                            │     │
│  │                                                               │     │
│  │   evaluator = Evaluator(                                     │     │
│  │     program=optimized_program,                               │     │
│  │     metric=metric_function,                                  │     │
│  │     devset=testset,                                          │     │
│  │     num_threads=10                                           │     │
│  │   )                                                          │     │
│  │                                                               │     │
│  │ Optional Statistical Setup:                                  │     │
│  │                                                               │     │
│  │   statistical_evaluator = StatisticalEvaluator(             │     │
│  │     evaluator=evaluator,                                     │     │
│  │     n_runs=5,              # повторов оценки                │     │
│  │     confidence_level=0.95  # 95% доверительный интервал     │     │
│  │   )                                                          │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: PROGRAM EXECUTION                                              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ FOR each example in testset (ПАРАЛЛЕЛЬНО):                  │     │
│  │                                                               │     │
│  │   2.1 Извлечение входных данных                             │     │
│  │       inputs = {                                             │     │
│  │         field: example[field]                                │     │
│  │         for field in program.signature.inputs                │     │
│  │       }                                                      │     │
│  │                                                               │     │
│  │       Пример:                                                │     │
│  │       example = Example(                                     │     │
│  │         question="What is the capital of France?",          │     │
│  │         context="France is a country in Europe...",         │     │
│  │         answer="Paris"                                       │     │
│  │       )                                                      │     │
│  │                                                               │     │
│  │       inputs = {                                             │     │
│  │         "question": "What is the capital of France?",       │     │
│  │         "context": "France is a country in Europe..."       │     │
│  │       }                                                      │     │
│  │                                                               │     │
│  │   2.2 Выполнение программы                                  │     │
│  │       ┌────────────────────────────────────────────────┐   │     │
│  │       │ Program состоит из:                            │   │     │
│  │       │                                                 │   │     │
│  │       │ 1. Instruction (system prompt):                │   │     │
│  │       │    "You are an expert. Given the question..." │   │     │
│  │       │                                                 │   │     │
│  │       │ 2. Few-shot demos (if any):                    │   │     │
│  │       │    Example 1: Q: "...", A: "..."              │   │     │
│  │       │    Example 2: Q: "...", A: "..."              │   │     │
│  │       │                                                 │   │     │
│  │       │ 3. Current query:                              │   │     │
│  │       │    Q: "What is the capital of France?"        │   │     │
│  │       │    Context: "..."                              │   │     │
│  │       │                                                 │   │     │
│  │       │ task_model generates:                          │   │     │
│  │       │    "Paris"                                     │   │     │
│  │       └────────────────────────────────────────────────┘   │     │
│  │                                                               │     │
│  │       prediction = program(**inputs)                         │     │
│  │                                                               │     │
│  │   2.3 Извлечение предсказания                               │     │
│  │       pred_value = prediction[output_field]                 │     │
│  │       # Например: pred_value = "Paris"                      │     │
│  │                                                               │     │
│  │   2.4 Извлечение ground truth                               │     │
│  │       expected_value = example[output_field]                │     │
│  │       # Например: expected_value = "Paris"                  │     │
│  │                                                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  Результат Stage 2:                                                     │
│    predictions = [                                                      │
│      (pred1, expected1),                                               │
│      (pred2, expected2),                                               │
│      ...                                                               │
│    ]                                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: METRIC EVALUATION                                              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ FOR each (prediction, expected) pair:                       │     │
│  │                                                               │     │
│  │   score = metric(prediction, expected)                       │     │
│  │                                                               │     │
│  │   ┌─────────────────────────────────────────────────────┐  │     │
│  │   │ Примеры метрик:                                     │  │     │
│  │   │                                                      │  │     │
│  │   │ 1. ExactMatchMetric:                                │  │     │
│  │   │    return 1.0 if pred == expected else 0.0         │  │     │
│  │   │                                                      │  │     │
│  │   │ 2. DSPyMetricAdapter (LLM-based):                  │  │     │
│  │   │    ┌────────────────────────────────────────┐      │  │     │
│  │   │    │ Prompt: SIMILARITY_PROMPT              │      │  │     │
│  │   │    │                                         │      │  │     │
│  │   │    │ Входные данные:                        │      │  │     │
│  │   │    │   {output} = prediction                │      │  │     │
│  │   │    │   {ground_truth} = expected            │      │  │     │
│  │   │    │                                         │      │  │     │
│  │   │    │ judge_model оценивает similarity       │      │  │     │
│  │   │    │                                         │      │  │     │
│  │   │    │ Выходные данные:                       │      │  │     │
│  │   │    │   score: 8 (из 10)                     │      │  │     │
│  │   │    │   normalized: 0.8 (из 1.0)            │      │  │     │
│  │   │    └────────────────────────────────────────┘      │  │     │
│  │   │                                                      │  │     │
│  │   │ 3. FacilityMetric (JSON evaluation):               │  │     │
│  │   │    Compare JSON fields:                            │  │     │
│  │   │      - urgency match: 1.0 or 0.0                   │  │     │
│  │   │      - sentiment match: 1.0 or 0.0                 │  │     │
│  │   │      - categories accuracy: 0.0-1.0                │  │     │
│  │   │    total = average(all_field_scores)               │  │     │
│  │   │                                                      │  │     │
│  │   └─────────────────────────────────────────────────────┘  │     │
│  │                                                               │     │
│  │   scores.append(score)                                       │     │
│  │                                                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: AGGREGATION                                                    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ БАЗОВАЯ ОЦЕНКА:                                              │     │
│  │                                                               │     │
│  │   mean_score = sum(scores) / len(scores)                    │     │
│  │                                                               │     │
│  │   Пример:                                                     │     │
│  │     scores = [0.8, 0.9, 0.85, 0.7, 0.95, ...]              │     │
│  │     mean_score = 0.84                                        │     │
│  │                                                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ СТАТИСТИЧЕСКАЯ ОЦЕНКА:                                       │     │
│  │                                                               │     │
│  │   Повторить оценку n_runs раз (default: 5)                  │     │
│   FOR run in range(n_runs):                                    │     │
│  │     run_scores[run] = evaluate_once()                        │     │
│  │                                                               │     │
│  │   Вычислить статистику:                                      │     │
│  │     mean = np.mean(run_scores)                               │     │
│  │     std_dev = np.std(run_scores, ddof=1)                    │     │
│  │     std_error = std_dev / sqrt(n_runs)                       │     │
│  │                                                               │     │
│  │   Доверительный интервал (t-distribution):                   │     │
│  │     df = n_runs - 1                                          │     │
│  │     t_value = t.ppf((1 + confidence_level) / 2, df)         │     │
│  │     margin = t_value * std_error                             │     │
│  │     ci_lower = mean - margin                                 │     │
│  │     ci_upper = mean + margin                                 │     │
│  │                                                               │     │
│  │   Пример:                                                     │     │
│  │     run_scores = [0.84, 0.86, 0.83, 0.85, 0.84]            │     │
│  │     mean = 0.844                                             │     │
│  │     std_dev = 0.011                                          │     │
│  │     95% CI = [0.830, 0.858]                                  │     │
│  │                                                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ВЫХОДНЫЕ ДАННЫЕ                                                         │
│                                                                         │
│  Базовая оценка:                                                        │
│    score: 0.84                                                          │
│                                                                         │
│  Статистическая оценка:                                                 │
│    {                                                                    │
│      "mean": 0.844,                                                     │
│      "std_dev": 0.011,                                                  │
│      "std_error": 0.005,                                                │
│      "confidence_level": 0.95,                                          │
│      "confidence_interval": (0.830, 0.858),                            │
│      "n_runs": 5,                                                       │
│      "individual_scores": [0.84, 0.86, 0.83, 0.85, 0.84]              │
│    }                                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3. Типы метрик

#### 1. ExactMatchMetric (Точное совпадение)

```python
def exact_match(prediction, expected):
    return 1.0 if prediction.strip() == expected.strip() else 0.0
```

**Особенности:**
- Простая binary метрика
- Не учитывает частичное совпадение
- Быстрая оценка

#### 2. DSPyMetricAdapter (LLM-as-a-judge)

```
Prompt: SIMILARITY_PROMPT или CORRECTNESS_PROMPT
Input: prediction, ground_truth
Process: judge_model оценивает от 1 до 10
Output: normalized score 0.0-1.0
```

**Особенности:**
- Использует LLM для оценки
- Учитывает семантическое сходство
- Более гибкая оценка
- Медленнее, чем exact match

#### 3. FacilityMetric (JSON структурированная оценка)

```python
def facility_metric(prediction_json, expected_json):
    scores = []
    
    # Urgency match
    scores.append(1.0 if pred["urgency"] == exp["urgency"] else 0.0)
    
    # Sentiment match
    scores.append(1.0 if pred["sentiment"] == exp["sentiment"] else 0.0)
    
    # Categories accuracy
    cat_score = sum(
        pred["categories"][k] == exp["categories"][k]
        for k in exp["categories"]
    ) / len(exp["categories"])
    scores.append(cat_score)
    
    return sum(scores) / len(scores)
```

**Особенности:**
- Специализированная для JSON задач
- Оценивает несколько полей
- Взвешенное усреднение

### 4.4. Передача данных

```
INPUT → STAGE 1:
  optimized_program (DSPy.Predict)
  testset (List[Example])
  metric (Callable)

STAGE 1 → STAGE 2:
  evaluator (Evaluator object)
  testset

STAGE 2 → STAGE 3:
  predictions (List[Any])
  expected_values (List[Any])

STAGE 3 → STAGE 4:
  scores (List[float])

STAGE 4 → OUTPUT:
  mean_score (float)
  OR statistical_results (dict)
```

### 4.5. Параметры конфигурации

```python
{
  # Базовая оценка
  "num_threads": 10,              # параллельных потоков
  
  # Статистическая оценка
  "n_runs": 5,                    # повторов оценки
  "confidence_level": 0.95,       # доверительный интервал (95%)
  
  # Метрика
  "metric_type": "exact_match",   # или "dspy_adapter" или "custom"
  "metric_threshold": None        # порог для early stopping (опционально)
}
```

---

## 5. Заключение

### 5.1. Сравнение пайплайнов

| Характеристика | BasicOptimization | PDO | Evaluation |
|----------------|-------------------|-----|------------|
| **Цель** | Улучшение формата и стиля | Эволюция через соревнование | Оценка качества |
| **Алгоритм** | DSPy MIPROv2 | Dueling Bandits + Thompson Sampling | Метрики |
| **Сложность** | Низкая | Высокая | Низкая |
| **Время выполнения** | Быстро (минуты) | Медленно (часы) | Быстро (минуты) |
| **Количество промптов** | 2 | 8 | 0-2 |
| **Параллелизм** | Да | Да | Да |
| **Best for** | Quick improvements | Maximum performance | Quality assessment |

### 5.2. Общий workflow

```
User Request
     ↓
PromptMigrator.optimize()
     ↓
Choose Strategy
     ├─ BasicOptimization (for quick wins)
     └─ PDO (for best results)
     ↓
Generate optimized instruction
     ↓
(Optional) Evaluation Pipeline
     ↓
Save results
```

### 5.3. Ключевые промпты в системе

**Всего промптов: 10**

1. **DATASET_DESCRIPTOR_PROMPT** - анализ данных
2. **INSTRUCTION_PROPOSER_TEMPLATE** - генерация инструкций
3. **REASON_PROMPT** - выполнение close-ended задач
4. **ANSWER_PROMPT_OPEN** - выполнение open-ended задач
5. **EVALUATE_PROMPT** - судейство close-ended
6. **EVALUATE_OPEN_PROMPT** - судейство open-ended
7. **REQUIREMENT_PROPOSER_TEMPLATE** - создание критериев
8. **MUTATE_PROMPT_TEMPLATE** - мутация без labels
9. **MUTATE_PROMPT_TEMPLATE_WITH_LABELS** - мутация с labels
10. **SIMILARITY_PROMPT / CORRECTNESS_PROMPT** - LLM-метрики

### 5.4. Потоки данных в системе

```
Raw Data (questions, contexts, answers)
     ↓
Dataset Analysis (DATASET_DESCRIPTOR_PROMPT)
     ↓
Dataset Summary (string)
     ↓
Instruction Generation (INSTRUCTION_PROPOSER_TEMPLATE)
     ↓
Instruction Pool (List[str])
     ↓
Task Execution (REASON_PROMPT / ANSWER_PROMPT_OPEN)
     ↓
Responses (List[dict])
     ↓
Judge Evaluation (EVALUATE_PROMPT / EVALUATE_OPEN_PROMPT)
     ↓
Win Matrix (2D array)
     ↓
Ranking & Mutation (MUTATE_PROMPT_TEMPLATE)
     ↓
Best Instruction (string)
     ↓
Final Program (DSPy.Predict with optimized instruction)
     ↓
Evaluation (Metrics)
     ↓
Score (float 0-1)
```

### 5.5. Советы по использованию

**Когда использовать BasicOptimization:**
- Нужны быстрые результаты
- Ограниченный бюджет на API calls
- Задача не требует максимальной точности
- Первоначальное тестирование

**Когда использовать PDO:**
- Нужна максимальная производительность
- Есть бюджет на множество API calls
- Критически важная задача
- Финальная оптимизация для production

**Настройка параметров:**
- `num_candidates` ↑ → больше разнообразие, но дольше
- `total_rounds` ↑ → лучше результаты PDO, но дороже
- `num_duels_per_round` ↑ → более точный ranking
- `thompson_alpha` ↑ → больше exploration

---

## Документация завершена

Этот документ описывает все возможные пайплайны работы агента Prompt-Ops:
- **BasicOptimization** - быстрая оптимизация через DSPy MIPROv2
- **PDO** - продвинутая оптимизация через dueling bandits
- **Evaluation** - оценка качества промптов

Каждый пайплайн включает:
- Детальные ASCII-диаграммы
- Пошаговое описание всех этапов
- Потоки данных между этапами
- Используемые промпты с входными/выходными данными
- Параметры конфигурации

Для полного описания промптов см. `PROMPTS_DOCUMENTATION.md`.
