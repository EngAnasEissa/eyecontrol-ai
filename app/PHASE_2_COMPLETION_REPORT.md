# Phase 2 Completion Report

## EyeControl AI

---

## 1. هدف Phase 2

تهدف Phase 2 إلى جمع وتجهيز والتحقق من بيانات حركات واتجاهات العين والحالات المرتبطة بها، بحيث تصبح البيانات جاهزة للاستخدام في المراحل اللاحقة من مشروع **EyeControl AI**.

تم تنفيذ عملية جمع البيانات باستخدام نظام التحقق الموجود في المشروع، ثم تم تحليل البيانات وفحص جودتها وإنشاء Dataset متوازن مستقل.

---

## 2. الحالات السبع

تم جمع البيانات للحالات التالية:

1. `FORWARD`
2. `LEFT`
3. `RIGHT`
4. `UP`
5. `DOWN`
6. `EYES_OPEN`
7. `EYES_CLOSED`

---

## 3. آلية جمع البيانات

يستخدم المشروع نظام Validation لتسجيل البيانات الخاصة بكل حالة.

مدة تسجيل كل جلسة:

```text
VALIDATION_RECORD_SECONDS = 3.0
```

أثناء التسجيل، تتم إضافة العينة عندما تكون خصائص العين صالحة.

### مفاتيح التحقق

| المفتاح | الحالة        |
| ------- | ------------- |
| `1`     | `FORWARD`     |
| `2`     | `LEFT`        |
| `3`     | `RIGHT`       |
| `4`     | `UP`          |
| `5`     | `DOWN`        |
| `6`     | `EYES_OPEN`   |
| `7`     | `EYES_CLOSED` |

---

## 4. ملف البيانات الخام

تم الاحتفاظ بملف البيانات الخام:

```text
phase2_validation.csv
```

ويُعتبر هذا الملف **Raw Source of Truth** للبيانات التي تم جمعها.

البيانات الخام النهائية قبل إنشاء Balanced Dataset كانت:

| الحالة        | عدد العينات |
| ------------- | ----------: |
| `FORWARD`     |         160 |
| `LEFT`        |         173 |
| `RIGHT`       |         225 |
| `UP`          |         133 |
| `DOWN`        |         132 |
| `EYES_OPEN`   |         132 |
| `EYES_CLOSED` |         134 |
| **الإجمالي**  |    **1089** |

لم يتم حذف العينات الزائدة من ملف البيانات الخام.

---

## 5. إنشاء Balanced Dataset

تم إنشاء ملف مستقل:

```text
phase2_balanced.csv
```

وتم اختيار 100 عينة من كل حالة لإنشاء Dataset متوازن.

تم استخدام Random Seed ثابت:

```text
seed = 42
```

ولم يتم تكرار العينات أثناء عملية الاختيار.

### النتيجة

| الحالة        | العينات |
| ------------- | ------: |
| `FORWARD`     |     100 |
| `LEFT`        |     100 |
| `RIGHT`       |     100 |
| `UP`          |     100 |
| `DOWN`        |     100 |
| `EYES_OPEN`   |     100 |
| `EYES_CLOSED` |     100 |
| **الإجمالي**  | **700** |

أي:

```text
7 classes × 100 samples = 700 samples
```

---

## 6. بنية البيانات

يحتوي `phase2_balanced.csv` على الأعمدة التالية:

```text
timestamp
state
left_norm_x
left_norm_y
right_norm_x
right_norm_y
left_ear
right_ear
```

### وصف الخصائص

* `timestamp`: وقت تسجيل العينة.
* `state`: الحالة المرتبطة بالعينة.
* `left_norm_x`: إحداثي X المطبّع للعين اليسرى.
* `left_norm_y`: إحداثي Y المطبّع للعين اليسرى.
* `right_norm_x`: إحداثي X المطبّع للعين اليمنى.
* `right_norm_y`: إحداثي Y المطبّع للعين اليمنى.
* `left_ear`: قيمة Eye Aspect Ratio للعين اليسرى.
* `right_ear`: قيمة Eye Aspect Ratio للعين اليمنى.

---

## 7. فحص بنية الملف

تم فحص `phase2_balanced.csv`.

النتائج:

```text
Total rows = 700
```

الأعمدة مطابقة للبنية المتوقعة.

لا توجد أعمدة مفقودة.

لا توجد حالات غير متوقعة.

النتيجة:

```text
[PASS] Columns are correct.
[PASS] No unexpected states.
```

---

## 8. فحص توازن البيانات

تم التحقق من أن جميع الحالات السبع تحتوي على العدد نفسه من العينات.

كل حالة تحتوي على:

```text
100 samples
```

والإجمالي:

```text
7 × 100 = 700
```

النتيجة:

```text
[PASS] Every state contains exactly 100 samples.
```

---

## 9. فحص جودة البيانات

تم فحص البيانات بحثًا عن القيم الفارغة والقيم غير الرقمية والقيم غير finite.

النتائج:

```text
Empty values       : 0
Non-numeric values : 0
Non-finite values  : 0
```

وبالتالي:

* لا توجد قيم فارغة.
* لا توجد قيم غير رقمية في الخصائص.
* لا توجد قيم `NaN`.
* لا توجد قيم `Infinity`.

النتائج:

```text
[PASS] No empty values.
[PASS] No non-numeric feature values.
[PASS] No NaN/Infinity values.
```

---

## 10. فحص نطاق الخصائص

تم التحقق من أن جميع قيم الخصائص تقع ضمن النطاق `[0, 1]`.

### `left_norm_x`

```text
min = 0.407791
max = 0.691429
violations = 0
```

### `left_norm_y`

```text
min = 0.276182
max = 0.815467
violations = 0
```

### `right_norm_x`

```text
min = 0.316462
max = 0.621422
violations = 0
```

### `right_norm_y`

```text
min = 0.235094
max = 0.828314
violations = 0
```

### `left_ear`

```text
min = 0.034275
max = 0.520562
violations = 0
```

### `right_ear`

```text
min = 0.008426
max = 0.536324
violations = 0
```

النتيجة:

```text
[PASS] All feature values are within valid ranges.
```

---

## 11. فحص التكرار

تم فحص التكرار باستخدام خصائص العين:

```text
left_norm_x
left_norm_y
right_norm_x
right_norm_y
left_ear
right_ear
```

النتيجة:

```text
Duplicate feature rows: 0
```

وبالتالي:

```text
[PASS] No duplicate feature rows.
```

---

## 12. سلامة ملف البيانات الخام

تم الحفاظ على:

```text
phase2_validation.csv
```

باعتباره المصدر الخام للبيانات.

لم يتم تعديل أو حذف العينات الإضافية منه أثناء إنشاء Balanced Dataset.

تم إنشاء:

```text
phase2_balanced.csv
```

كملف مستقل يحتوي على العينات المختارة للتوازن.

---

## 13. أدوات التحقق المستخدمة

تم استخدام الأدوات والسكريبتات التالية خلال Phase 2:

### `phase2_1_stats.py`

يُستخدم لإجراء التحليل الإحصائي لبيانات Phase 2، بما في ذلك أعداد العينات ومتوسطات وخصائص البيانات حسب الحالة.

### `phase2_2_data_quality.py`

يُستخدم لفحص جودة البيانات الخام، بما في ذلك بنية الأعمدة والقيم والمجالات والتكرارات وبعض مؤشرات جودة البيانات.

### `phase2_balance_plan.py`

يُستخدم للتحقق من أعداد العينات المتوفرة لكل حالة وتحديد مقدار البيانات الإضافية المطلوبة للوصول إلى العدد المستهدف.

### `phase2_create_balanced.py`

يُستخدم لإنشاء:

```text
phase2_balanced.csv
```

باختيار 100 عينة لكل حالة باستخدام Random Seed ثابت.

### `phase2_balanced_quality.py`

يُستخدم لإجراء الفحص النهائي لـ `phase2_balanced.csv` والتحقق من:

* عدد العينات.
* الأعمدة.
* الحالات.
* التوازن.
* القيم الفارغة.
* القيم غير الرقمية.
* القيم غير finite.
* نطاق الخصائص.
* التكرار.

---

# 14. النتيجة النهائية

## FINAL STATUS

```text
PHASE 2 = COMPLETED
```

تم تحقيق المتطلبات التالية:

* 7 حالات.
* 700 عينة متوازنة.
* 100 عينة لكل حالة.
* بنية الأعمدة صحيحة.
* لا توجد حالات غير متوقعة.
* لا توجد قيم فارغة.
* لا توجد قيم غير رقمية.
* لا توجد قيم `NaN` أو `Infinity`.
* لا توجد مخالفات في نطاق الخصائص.
* لا توجد صفوف مكررة.
* ملف البيانات الخام محفوظ بدون تعديل.
* تم إنشاء Balanced Dataset مستقل.
* تم التحقق من Balanced Dataset بنجاح.

---

## 15. قرار الانتقال

أصبحت Phase 2 مكتملة بعد تنفيذ عمليات جمع البيانات والتحليل وفحص الجودة والموازنة والتحقق النهائي.

أصبح الملف:

```text
phase2_balanced.csv
```

جاهزًا للاستخدام في المرحلة التالية بعد مراجعة خطة المرحلة التالية واعتمادها.

لا يتم بدء التدريب أو تعديل منطق التطبيق ضمن هذا التقرير.

---

## 16. سجل التحقق النهائي

| الاختبار        | النتيجة |
| --------------- | ------- |
| Dataset exists  | PASS    |
| Columns         | PASS    |
| Expected states | PASS    |
| Balance         | PASS    |
| Empty values    | PASS    |
| Numeric values  | PASS    |
| Finite values   | PASS    |
| Feature ranges  | PASS    |
| Duplicate rows  | PASS    |
| Total samples   | PASS    |

---

## Phase 2 Completion Summary

```text
Phase 2 Completion Status: COMPLETED
Dataset: phase2_balanced.csv
Samples: 700
Classes: 7
Samples per class: 100
Raw Dataset Preserved: YES
Verification Status: PASS
```

---

**End of Phase 2 Completion Report**
