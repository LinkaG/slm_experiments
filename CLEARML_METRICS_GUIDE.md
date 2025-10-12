# Руководство по метрикам в ClearML

## 📊 **Как теперь отображаются результаты**

### **✅ Что изменилось:**

После обновления метрики отображаются правильно:

#### **1. SCALARS (вкладка)**
Содержит:
- ✅ **Single values** - итоговые значения метрик (БЕЗ графиков)
  - `token_recall` - финальное значение
  - `num_examples` - количество примеров
  - `duration_seconds` - время выполнения
  - `model_size_bytes` - размер модели
  - `dataset_num_eval` - размер датасета
  
- ✅ **Training Progress** - график прогресса (если нужен)
  - `token_recall` по оси X = номер примера
  - Показывает, как менялось качество по ходу обработки

#### **2. PLOTS (вкладка)**
Содержит:
- ✅ **Final Metrics** - таблица с итоговыми результатами
  ```
  | Metric         | Value  |
  |----------------|--------|
  | token_recall   | 0.6667 |
  | num_examples   | 10.0000|
  ```

#### **3. CONSOLE (вкладка)**
- Все логи эксперимента
- Примеры предсказаний
- Детальная информация

---

## 🔍 **Что такое "итерации" в графиках?**

### **До изменений:**
```
iteration=0 для всех метрик → все точки в одном месте → бессмысленный график
```

### **После изменений:**

#### **Single Values (SCALARS):**
```
Метрика отображается как ОДНО ЧИСЛО без графика
token_recall: 0.6667 ✅
```

#### **Progress Graph (SCALARS > Training Progress):**
```
iteration = номер обработанного примера
X: 0, 1, 2, 3, ..., 10
Y: token_recall для каждого примера
Результат: график, показывающий динамику качества ✅
```

#### **Table (PLOTS):**
```
Все метрики в одной таблице
Удобно для сравнения экспериментов ✅
```

---

## 🎯 **Где найти результаты:**

### **1. Итоговые метрики (числа):**
```
ClearML Web UI → SCALARS → Single Values
```
Здесь видны все финальные значения БЕЗ графиков.

### **2. Таблица результатов:**
```
ClearML Web UI → PLOTS → Final Metrics
```
Красивая таблица со всеми метриками.

### **3. График прогресса:**
```
ClearML Web UI → SCALARS → Training Progress → token_recall
```
График показывает, как менялось качество по примерам.

### **4. Детальные результаты:**
```
ClearML Web UI → CONSOLE
```
Все логи + примеры предсказаний.

---

## 📈 **Типы метрик в ClearML:**

| Тип | Метод | Где отображается | Когда использовать |
|-----|-------|------------------|-------------------|
| **Single Value** | `report_single_value()` | SCALARS (число) | Итоговые метрики |
| **Scalar** | `report_scalar()` | SCALARS (график) | Прогресс обучения |
| **Table** | `report_table()` | PLOTS | Сводная таблица |
| **Text** | `report_text()` | CONSOLE | Логи и примеры |

---

## 💡 **Примеры использования:**

### **Итоговая метрика (одно значение):**
```python
logger.report_single_value("final_accuracy", 0.95)
# Результат: в SCALARS появится "final_accuracy: 0.95"
```

### **График прогресса (серия значений):**
```python
for epoch in range(100):
    loss = train_epoch()
    logger.report_scalar("Training", "loss", loss, iteration=epoch)
# Результат: график loss по эпохам
```

### **Таблица результатов:**
```python
import pandas as pd
df = pd.DataFrame([
    {"Metric": "Accuracy", "Value": "0.95"},
    {"Metric": "F1-Score", "Value": "0.93"}
])
logger.report_table("Results", "Summary", table_plot=df)
# Результат: таблица в PLOTS
```

---

## 🔧 **Настройка отображения:**

### **Убрать графики для финальных метрик:**
Используйте `report_single_value()` вместо `report_scalar()`:

```python
# ❌ Неправильно - создает график
logger.report_scalar("metrics", "accuracy", 0.95, iteration=0)

# ✅ Правильно - только значение
logger.report_single_value("accuracy", 0.95)
```

### **Добавить таблицу в PLOTS:**
```python
import pandas as pd
from src.utils.clearml_config import log_metrics_to_clearml

metrics = {"accuracy": 0.95, "f1_score": 0.93}
log_metrics_to_clearml(logger, metrics)
# Автоматически создает таблицу в PLOTS
```

---

## 📊 **Сравнение экспериментов:**

В ClearML Web UI:
1. Выберите несколько экспериментов (checkbox)
2. Кнопка **COMPARE**
3. Увидите таблицу со всеми метриками side-by-side

Пример:
```
| Experiment | token_recall | num_examples | duration |
|------------|--------------|--------------|----------|
| Exp #1     | 0.6667       | 10           | 0.04s    |
| Exp #2     | 0.7123       | 100          | 0.42s    |
| Exp #3     | 0.7456       | 1000         | 4.21s    |
```

---

## ✅ **Итого:**

После обновления вы получите:

1. ✅ **SCALARS** - итоговые значения метрик (БЕЗ лишних графиков)
2. ✅ **PLOTS** - таблицу с результатами (удобно для анализа)
3. ✅ **Training Progress** - график прогресса (опционально, если нужен)
4. ✅ **CONSOLE** - все логи и примеры

**Больше никаких бессмысленных графиков с iteration=0!** 🎉

