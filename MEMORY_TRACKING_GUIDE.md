# Руководство по отслеживанию памяти

## 📊 **Какую память мы измеряем?**

### **1. CPU RAM (Оперативная память)**

**Что измеряется:**
```python
cpu_ram_used_mb = process.memory_info().rss / (1024 * 1024)
```

**RSS (Resident Set Size)** - это:
- ✅ Физическая оперативная память, используемая процессом Python
- ✅ Включает все загруженные данные:
  - Датасет в памяти
  - Веса модели (если на CPU)
  - Промежуточные вычисления
  - Библиотеки Python
  - Кэш операционной системы для процесса

**НЕ включает:**
- ❌ Swap память (подкачка на диск)
- ❌ Shared memory между процессами
- ❌ GPU память

---

### **2. GPU RAM (Видеопамять)**

#### **a) `gpu_ram_used_mb` - Выделенная память**
```python
gpu_ram_used = torch.cuda.memory_allocated() / (1024 * 1024)
```

**Что это:**
- ✅ Память, **реально используемая** PyTorch для тензоров
- ✅ Веса модели на GPU
- ✅ Промежуточные активации
- ✅ Градиенты (если есть)

**Пример:**
```
Модель 1.7B параметров × 4 байта (float32) = ~6.8 GB
+ Активации и кэш KV = еще ~2-4 GB
Итого: ~10 GB реально используется
```

---

#### **b) `gpu_ram_peak_mb` - Пиковое использование**
```python
gpu_ram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
```

**Что это:**
- ✅ **Максимальная** память, которая была выделена за все время
- ✅ Полезно для оценки минимальных требований к GPU

**Пример:**
```
Обычное использование: 8 GB
Во время генерации длинного ответа: 12 GB ← это и будет peak
```

---

#### **c) `reserved_gpu_ram_mb` - Зарезервированная память**
```python
reserved_gpu_ram = torch.cuda.memory_reserved() / (1024 * 1024)
```

**Что это:**
- ✅ Память, **зарезервированная** PyTorch у GPU
- ✅ Может быть больше, чем реально используется
- ✅ PyTorch не возвращает память GPU сразу (для оптимизации)

**Пример:**
```
Используется: 8 GB
Зарезервировано: 10 GB ← PyTorch держит +2 GB "про запас"
```

**Зачем PyTorch резервирует больше?**
- Чтобы не выделять/освобождать память каждый раз
- Ускоряет последующие операции
- Можно очистить: `torch.cuda.empty_cache()`

---

## 📈 **Когда измеряется память?**

Смотрим в коде:

```python
# 1. В начале эксперимента
self.memory_tracker.log_memory("system", "experiment_start")

# 2. Перед/после ретривера (если используется)
self.memory_tracker.log_memory("retriever", "before_retrieve")
# ... retriever.retrieve() ...
self.memory_tracker.log_memory("retriever", "after_retrieve")

# 3. Перед/после генерации модели
self.memory_tracker.log_memory("model", "before_generate")
# ... model.generate() ...
self.memory_tracker.log_memory("model", "after_generate")

# 4. Периодически каждые 100 примеров
if processed % 100 == 0:
    self.memory_tracker.clear_memory()  # Очищаем кэш

# 5. В конце эксперимента
self.memory_tracker.log_memory("system", "experiment_end")
self.memory_tracker.save_log()  # Сохраняем в файл
```

---

## 💾 **Где найти результаты?**

### **1. Локальный файл:**
```bash
outputs/<experiment_name>/memory_usage.json
```

**Формат:**
```json
{
  "detailed_log": [
    {
      "component": "model",
      "operation": "before_generate",
      "timestamp": 1234567890.123,
      "cpu_ram_used_mb": 1024.5,
      "gpu_ram_used_mb": 8192.3,
      "gpu_ram_peak_mb": 8500.1,
      "reserved_gpu_ram_mb": 10240.0
    },
    ...
  ],
  "peak_usage": {
    "cpu_ram_used_mb": 1200.0,
    "gpu_ram_used_mb": 10240.0,
    "gpu_ram_peak_mb": 12000.0,
    "reserved_gpu_ram_mb": 14000.0
  }
}
```

---

### **2. ClearML Web UI:**

**SCALARS → memory/**
```
memory/model/
  ├─ cpu_ram_used_mb
  ├─ gpu_ram_used_mb
  ├─ gpu_ram_peak_mb
  └─ reserved_gpu_ram_mb

memory/retriever/
  ├─ cpu_ram_used_mb
  └─ ...

memory/system/
  ├─ cpu_ram_used_mb
  └─ ...
```

**Графики показывают:**
- X-axis = iteration (каждое измерение)
- Y-axis = память в MB
- Можно увидеть динамику использования памяти

---

### **3. MinIO S3:**
```
s3://clearml-artifacts/.../artifacts/memory_usage/memory_usage.json
```

---

## 🎯 **Практические примеры:**

### **Пример 1: SmolLM-1.7B на GPU**
```json
{
  "peak_usage": {
    "cpu_ram_used_mb": 2048,      // ~2 GB CPU (датасет + код)
    "gpu_ram_used_mb": 8192,       // ~8 GB GPU (модель + активации)
    "gpu_ram_peak_mb": 10240,      // Пик ~10 GB
    "reserved_gpu_ram_mb": 12288   // PyTorch зарезервировал ~12 GB
  }
}
```

**Вывод:**
- ✅ Модель работает на GPU
- ✅ Нужна GPU с минимум 12 GB памяти
- ✅ CPU использует ~2 GB для данных

---

### **Пример 2: Модель на CPU (нет GPU)**
```json
{
  "peak_usage": {
    "cpu_ram_used_mb": 15360,     // ~15 GB CPU (модель + данные)
    "gpu_ram_used_mb": 0,          // GPU не используется
    "gpu_ram_peak_mb": 0,
    "reserved_gpu_ram_mb": 0
  }
}
```

**Вывод:**
- ✅ Модель работает на CPU
- ✅ Нужно минимум 16 GB RAM
- ⚠️  Будет работать медленно

---

### **Пример 3: С квантизацией (int8)**
```json
{
  "peak_usage": {
    "cpu_ram_used_mb": 2048,
    "gpu_ram_used_mb": 4096,       // Меньше! 8-bit вместо 32-bit
    "gpu_ram_peak_mb": 5120,       // Пик тоже меньше
    "reserved_gpu_ram_mb": 6144    // Можно на меньшей GPU!
  }
}
```

**Вывод:**
- ✅ Квантизация уменьшила память в ~2 раза
- ✅ Можно использовать GPU с 8 GB вместо 12 GB

---

## 🔍 **Как анализировать память?**

### **1. Проверить пиковое использование:**
```python
import json

with open('outputs/.../memory_usage.json') as f:
    data = json.load(f)
    peak = data['peak_usage']
    
    print(f"Peak GPU: {peak['gpu_ram_peak_mb']} MB")
    print(f"Peak CPU: {peak['cpu_ram_used_mb']} MB")
```

### **2. Найти утечки памяти:**
Смотрите на график в ClearML:
- Если память растет постоянно → утечка
- Если память стабильна → всё ОК

### **3. Оптимизировать:**
Если памяти не хватает:
- ✅ Уменьшить `batch_size`
- ✅ Включить `gradient_checkpointing`
- ✅ Использовать квантизацию (`load_in_8bit`)
- ✅ Очищать кэш: `torch.cuda.empty_cache()`

---

## 📊 **Сравнение разных конфигураций:**

| Модель | Precision | CPU RAM | GPU RAM | GPU Peak | Комментарий |
|--------|-----------|---------|---------|----------|-------------|
| SmolLM-135M | FP32 | 1 GB | 2 GB | 2.5 GB | Легкая модель |
| SmolLM-1.7B | FP32 | 2 GB | 8 GB | 10 GB | Средняя модель |
| SmolLM-1.7B | INT8 | 2 GB | 4 GB | 5 GB | С квантизацией |
| Qwen-4B | FP32 | 3 GB | 18 GB | 22 GB | Большая модель |
| Qwen-4B | INT8 | 3 GB | 9 GB | 11 GB | Оптимизация |

---

## ✅ **Итого:**

### **Мы измеряем:**
1. **CPU RAM** - оперативная память процесса (датасет + код)
2. **GPU RAM Used** - реально используемая видеопамять (модель + активации)
3. **GPU RAM Peak** - максимальное использование (для оценки требований)
4. **GPU RAM Reserved** - зарезервированная PyTorch (может быть больше)

### **Данные сохраняются:**
- ✅ Локально: `outputs/.../memory_usage.json`
- ✅ ClearML: графики в SCALARS → memory/
- ✅ MinIO S3: артефакт `memory_usage`

### **Используйте для:**
- 🎯 Оценки минимальных требований к железу
- 🎯 Поиска утечек памяти
- 🎯 Сравнения разных конфигураций
- 🎯 Оптимизации использования ресурсов

