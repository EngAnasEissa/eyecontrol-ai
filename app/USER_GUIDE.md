# EyeControl AI

## دليل المستخدم

---

## 1. مقدمة

**EyeControl AI** هو نظام للتحكم بالماوس باستخدام حركات العين.

يعتمد النظام على كاميرا الكمبيوتر لالتقاط صورة الوجه، ثم يستخدم تقنيات الرؤية الحاسوبية لاكتشاف معالم الوجه والعينين، واستخراج خصائص العين، ثم يستخدم نموذج تعلم آلي للتعرف على اتجاه النظر.

بعد التعرف على اتجاه العين، يمكن للنظام استخدام النتيجة للتحكم في مؤشر الماوس.

يدعم النظام حاليًا:

* التعرف على اتجاه النظر.
* التعرف على حالة العين المفتوحة والمغلقة.
* التحكم في حركة الماوس بالعين.
* النقر بالعين.
* تنعيم التنبؤات لتقليل الاهتزاز.
* سرعات حركة مختلفة للماوس حسب استقرار الاتجاه.
* تسجيل بيانات التحقق الخاصة بالعين.
* واجهة عرض مباشرة لحالة النظام والتنبؤات.

---

# 2. فكرة عمل النظام

يعمل النظام من خلال سلسلة من المراحل:

```text
Camera
   ↓
Face Detection / Face Landmarks
   ↓
Eye Feature Extraction
   ↓
Random Forest Model
   ↓
Prediction
   ↓
Prediction Smoothing
   ↓
Mouse / Eye Click Control
```

### شرح المراحل

### 2.1 Camera

تلتقط الكاميرا صورة المستخدم بشكل مستمر.

---

### 2.2 Face Landmarks

يستخدم النظام نموذج Face Landmarker لاكتشاف نقاط الوجه والعينين.

---

### 2.3 Eye Features

يتم استخراج خصائص من العينين، مثل:

* موقع القزحية.
* الموقع الطبيعي للعين.
* Eye Aspect Ratio - EAR.
* أبعاد العين.
* المسافة بين العينين.

---

### 2.4 Machine Learning

يستخدم المشروع نموذج:

**Random Forest Classifier**

للتعرف على حالة العين واتجاه النظر.

التصنيفات الأساسية:

```text
FORWARD
LEFT
RIGHT
UP
DOWN
EYES_OPEN
EYES_CLOSED
```

---

### 2.5 Smoothing

لا يعتمد النظام على نتيجة إطار واحد فقط.

يتم الاحتفاظ بعدة تنبؤات حديثة واستخدامها لتقليل التغييرات السريعة والخاطئة في الاتجاه.

مثال:

```text
LEFT
LEFT
FORWARD
LEFT
LEFT
```

بدل تغيير الاتجاه مباشرة بسبب `FORWARD`، يستخدم النظام التنبؤ المستقر.

هذا يساعد على تقليل اهتزاز مؤشر الماوس.

---

# 3. متطلبات التشغيل

يحتاج المشروع إلى:

* Windows.
* Python 3.12.
* كاميرا تعمل.
* بيئة Python تحتوي على مكتبات المشروع.
* نموذج Machine Learning الموجود داخل مجلد `models`.

المشروع يستخدم بشكل أساسي:

* OpenCV.
* MediaPipe.
* NumPy.
* Pandas.
* Joblib.
* Scikit-learn.
* PyAutoGUI.

---

# 4. موقع المشروع

المسار الحالي للمشروع:

```text
C:\Users\HP\Desktop\eyecontrol-ai
```

الملف الرئيسي:

```text
app\main.py
```

---

# 5. بنية المشروع

البنية الأساسية للمشروع:

```text
eyecontrol-ai/
│
├── app/
│   ├── main.py
│   │
│   └── vision/
│       ├── camera.py
│       ├── face_landmarks.py
│       └── eye_features.py
│
├── models/
│   └── phase3_baseline_random_forest.joblib
│
├── phase2_validation.csv
│
└── ...
```

---

# 6. تشغيل المشروع

## الخطوة الأولى

افتح:

**PowerShell**

---

## الخطوة الثانية

انتقل إلى مجلد المشروع:

```powershell
cd C:\Users\HP\Desktop\eyecontrol-ai
```

---

## الخطوة الثالثة

شغل البرنامج باستخدام Python الصحيح:

```powershell
& "C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe" -m app.main
```

---

# 7. لماذا نستخدم `-m app.main`؟

يتم تشغيل البرنامج كـ Python Module حتى يستطيع Python التعرف على الحزمة `app` والوحدات الموجودة داخلها.

يجب تجنب تشغيل:

```powershell
python app\main.py
```

لأن ذلك قد يؤدي إلى الخطأ:

```text
ModuleNotFoundError: No module named 'app.vision'
```

الطريقة الصحيحة هي:

```powershell
python -m app.main
```

أو باستخدام مسار Python الكامل:

```powershell
& "C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe" -m app.main
```

---

# 8. عند بدء البرنامج

عند التشغيل تظهر نافذة:

```text
EyeControl AI
```

وتظهر معلومات النموذج في PowerShell.

يجب أن يظهر شيء مشابه:

```text
EyeControl AI
============================================================
Model: ...
Model type: RandomForestClassifier
Feature columns: [...]
Classes: [...]
Random state: 42
N estimators: 200
============================================================
```

بعد ذلك تظهر نافذة الكاميرا.

---

# 9. واجهة البرنامج

داخل نافذة الكاميرا تظهر معلومات مثل:

```text
Prediction: LEFT
Smoothed: LEFT
MOUSE: OFF
EYE CLICK: OFF
M:Mouse  C:EyeClick  Q/ESC:Exit
1:F  2:L  3:R  4:U  5:D  6:OPEN  8:CLOSED
```

---

# 10. معنى Prediction

السطر:

```text
Prediction: LEFT
```

هو التنبؤ الخام الصادر من نموذج Random Forest.

مثلاً:

```text
Prediction: LEFT
```

يعني أن النموذج صنف حركة العين على أنها:

```text
LEFT
```

---

# 11. معنى Smoothed

السطر:

```text
Smoothed: LEFT
```

هو القرار بعد تطبيق نظام التنعيم.

وهذا هو القرار المستخدم في التحكم بالماوس والنقر.

مثال:

```text
Prediction: LEFT
Smoothed: FORWARD
```

يعني أن النموذج أعطى `LEFT`، لكن نظام التنعيم لم يعتبر الحركة مستقرة بما يكفي لتحريك الماوس.

---

# 12. حالات التعرف

يدعم النموذج الحالات التالية:

| الحالة        | المعنى              |
| ------------- | ------------------- |
| `FORWARD`     | النظر للأمام        |
| `LEFT`        | النظر إلى اليسار    |
| `RIGHT`       | النظر إلى اليمين    |
| `UP`          | النظر إلى الأعلى    |
| `DOWN`        | النظر إلى الأسفل    |
| `EYES_OPEN`   | العينان مفتوحتان    |
| `EYES_CLOSED` | العينان مغلقتان     |
| `NO_FACE`     | لم يتم اكتشاف الوجه |

---

# 13. التحكم بالماوس

التحكم بالماوس يكون **متوقفًا افتراضيًا**.

في الشاشة:

```text
MOUSE: OFF
```

لتشغيله اضغط:

```text
M
```

فتصبح:

```text
MOUSE: ON
```

---

# 14. حركة الماوس

عند تشغيل Mouse Control:

| حركة العين  | حركة الماوس             |
| ----------- | ----------------------- |
| LEFT        | تحريك الماوس إلى اليسار |
| RIGHT       | تحريك الماوس إلى اليمين |
| UP          | تحريك الماوس إلى الأعلى |
| DOWN        | تحريك الماوس إلى الأسفل |
| FORWARD     | لا حركة                 |
| EYES_OPEN   | لا حركة                 |
| EYES_CLOSED | لا حركة                 |
| NO_FACE     | لا حركة                 |

---

# 15. نظام سرعة الماوس

لا يتم استخدام حركة واحدة ثابتة دائمًا.

يستخدم النظام ثلاث مستويات للحركة:

```text
Slow
Normal
Fast
```

القيم الحالية:

```text
Slow   = 5 px
Normal = 15 px
Fast   = 25 px
```

كلما أصبح الاتجاه أكثر استقرارًا، يستطيع النظام استخدام حركة أسرع.

مثال:

```text
LEFT
LEFT
LEFT
LEFT
LEFT
```

يعتبر الاتجاه مستقرًا أكثر من:

```text
LEFT
FORWARD
LEFT
RIGHT
LEFT
```

---

# 16. نظام Smoothing

يستخدم النظام نافذة تنعيم:

```text
5 predictions
```

ويحتاج إلى اتفاق عدة تنبؤات قبل اعتماد الاتجاه.

القيمة الحالية:

```text
SMOOTHING_WINDOW_SIZE = 5
SMOOTHING_REQUIRED_COUNT = 3
```

أي أن النظام يحاول تقليل تأثير التنبؤات الفردية الخاطئة.

---

# 17. Dead Zone

هناك حالات لا تحرك الماوس:

```text
FORWARD
EYES_OPEN
EYES_CLOSED
NO_FACE
```

وهذا يمنع تحرك المؤشر عندما لا يوجد اتجاه واضح للحركة.

---

# 18. إيقاف التحكم بالماوس

اضغط:

```text
M
```

مرة أخرى.

ستصبح:

```text
MOUSE: OFF
```

وبذلك يتوقف تحريك الماوس، بينما يستمر التعرف على العين.

---

# 19. Eye Click

يوفر النظام إمكانية النقر باستخدام إغلاق العين.

Eye Click متوقف افتراضيًا:

```text
EYE CLICK: OFF
```

لتشغيله اضغط:

```text
C
```

فتصبح:

```text
EYE CLICK: ON
```

---

# 20. كيفية عمل Eye Click

بعد تشغيل Eye Click:

1. أغلق عينيك.
2. حافظ على الإغلاق.
3. ينتظر النظام مدة محددة.
4. إذا استمر `EYES_CLOSED` لمدة `0.70` ثانية تقريبًا، يتم تنفيذ نقرة يسارية واحدة.

القيمة الحالية:

```text
EYE_CLICK_HOLD_TIME = 0.70
```

---

# 21. منع تكرار النقر

النظام مصمم لمنع النقر المتكرر أثناء استمرار إغلاق العين.

مثال:

```text
EYES_CLOSED
EYES_CLOSED
EYES_CLOSED
EYES_CLOSED
```

لا ينتج عنه عدة نقرات.

يتم تنفيذ:

```text
Left Click
```

مرة واحدة فقط.

بعد فتح العين أو تغير الحالة، يتم إعادة تسليح Eye Click.

---

# 22. إيقاف Eye Click

اضغط:

```text
C
```

مرة أخرى.

ستصبح:

```text
EYE CLICK: OFF
```

---

# 23. عدم تشغيل Mouse وEye Click أثناء الاختبار الأول

عند اختبار النظام لأول مرة، يفضل أن تبدأ بـ:

```text
MOUSE: OFF
EYE CLICK: OFF
```

ثم اختبر التعرف فقط.

بعد التأكد من أن التنبؤات صحيحة:

```text
Prediction
Smoothed
```

يمكن تشغيل:

```text
M
```

ثم اختبار Mouse Control.

وبعد نجاحه يمكن اختبار:

```text
C
```

لتشغيل Eye Click.

---

# 24. اختصارات لوحة المفاتيح

| المفتاح | الوظيفة                          |
| ------- | -------------------------------- |
| `M`     | تشغيل/إيقاف Mouse Control        |
| `C`     | تشغيل/إيقاف Eye Click            |
| `1`     | بدء Validation لحالة FORWARD     |
| `2`     | بدء Validation لحالة LEFT        |
| `3`     | بدء Validation لحالة RIGHT       |
| `4`     | بدء Validation لحالة UP          |
| `5`     | بدء Validation لحالة DOWN        |
| `6`     | بدء Validation لحالة EYES_OPEN   |
| `8`     | بدء Validation لحالة EYES_CLOSED |
| `Q`     | الخروج                           |
| `ESC`   | الخروج                           |

---

# 25. الخروج من البرنامج

يمكن الخروج بطريقتين:

اضغط:

```text
Q
```

أو:

```text
ESC
```

عند الخروج يتم:

* تحرير الكاميرا.
* إغلاق MediaPipe detector.
* إغلاق نافذة OpenCV.
* إيقاف البرنامج.

ويظهر:

```text
EyeControl AI stopped.
```

---

# 26. Phase 2 Validation

يوفر البرنامج نظامًا لتسجيل بيانات التحقق.

يمكن بدء التحقق باستخدام المفاتيح:

```text
1 → FORWARD
2 → LEFT
3 → RIGHT
4 → UP
5 → DOWN
6 → EYES_OPEN
8 → EYES_CLOSED
```

---

# 27. مدة تسجيل Validation

القيمة الحالية:

```text
VALIDATION_RECORD_SECONDS = 3.0
```

أي أن كل جلسة Validation تستمر تقريبًا:

```text
3 seconds
```

أثناء التسجيل تظهر على الشاشة:

```text
VALIDATING: LEFT
```

---

# 28. ملف Validation

يتم تسجيل بيانات التحقق في:

```text
phase2_validation.csv
```

داخل مجلد المشروع.

ويحتوي الملف على خصائص مثل:

```text
left_norm_x
left_norm_y
right_norm_x
right_norm_y
left_ear
right_ear
label
```

---

# 29. مثال على Validation

إذا أردت تسجيل حالة `LEFT`:

1. شغّل البرنامج.
2. تأكد من ظهور الكاميرا.
3. اضغط:

```text
2
```

4. انظر إلى اليسار لمدة التسجيل.
5. ستظهر:

```text
VALIDATING: LEFT
```

6. بعد انتهاء المدة ينتهي التسجيل.

---

# 30. نصائح للحصول على أفضل أداء

للحصول على نتائج أفضل:

### الإضاءة

استخدم إضاءة جيدة وموزعة.

تجنب:

* الظلام الشديد.
* الضوء القوي خلف المستخدم.
* انعكاس الضوء المباشر على العين.

---

### وضع الكاميرا

يفضل وضع الكاميرا:

```text
أمام المستخدم
```

وبارتفاع قريب من مستوى العين.

---

### المسافة

اجلس أمام الكاميرا بمسافة مناسبة بحيث يكون الوجه واضحًا.

يجب أن يكون الوجه ظاهرًا بشكل كامل تقريبًا.

---

### النظارات

إذا كانت النظارات تسبب انعكاسات قوية، يمكن أن يؤثر ذلك على التعرف على العين.

---

### ثبات الرأس

في الإصدار الحالي يعتمد النظام على حركات العين، لذلك يفضل أثناء الاختبار:

* إبقاء الرأس ثابتًا نسبيًا.
* تحريك العينين بدل تحريك الرأس فقط.

---

# 31. أفضل طريقة لاستخدام النظام

لتحريك المؤشر:

```text
1. شغّل البرنامج.
2. تأكد من ظهور الوجه.
3. اضغط M.
4. انظر إلى الاتجاه المطلوب.
5. راقب Smoothed.
6. عندما يكون الاتجاه مستقرًا يتحرك الماوس.
```

مثال:

```text
النظر لليسار
      ↓
Prediction: LEFT
      ↓
Smoothed: LEFT
      ↓
Mouse → LEFT
```

---

# 32. استخدام Eye Click

بعد تشغيل البرنامج:

```text
MOUSE: OFF
EYE CLICK: OFF
```

اضغط:

```text
C
```

فتصبح:

```text
EYE CLICK: ON
```

ثم:

```text
أغلق العين
     ↓
EYES_CLOSED
     ↓
استمرار 0.70 ثانية
     ↓
Left Click
```

---

# 33. ترتيب الاختبار الموصى به

يفضل اختبار النظام بهذا الترتيب:

### الاختبار الأول

التعرف فقط:

```text
MOUSE: OFF
EYE CLICK: OFF
```

اختبر:

```text
LEFT
RIGHT
UP
DOWN
FORWARD
EYES_OPEN
EYES_CLOSED
```

---

### الاختبار الثاني

شغّل:

```text
M
```

واختبر حركة الماوس.

---

### الاختبار الثالث

أوقف Mouse Control:

```text
M
```

ثم شغّل:

```text
C
```

واختبر Eye Click.

---

### الاختبار الرابع

يمكن تشغيل الوظائف معًا بعد التأكد من نجاح كل وظيفة منفردة.

---

# 34. المشاكل الشائعة

## المشكلة: الكاميرا لا تعمل

إذا ظهر:

```text
Camera could not be opened.
```

تحقق من:

* أن الكاميرا متصلة.
* أن الكاميرا غير مستخدمة من برنامج آخر.
* السماح للبرنامج باستخدام الكاميرا.
* اختيار الكاميرا الصحيحة في Windows.

---

# 35. المشكلة: لا يظهر الوجه

إذا ظهرت:

```text
Prediction: NO_FACE
```

تحقق من:

* الإضاءة.
* وضع الوجه أمام الكاميرا.
* عدم الابتعاد كثيرًا عن الكاميرا.
* ظهور الوجه كاملًا.

---

# 36. المشكلة: التنبؤ يتغير بسرعة

إذا رأيت:

```text
Prediction: LEFT
Prediction: RIGHT
Prediction: LEFT
Prediction: FORWARD
```

بشكل متكرر، راقب:

```text
Smoothed
```

لأن النظام يستخدم Smoothing لتقليل تأثير هذه التغييرات.

يمكن لاحقًا ضبط:

```python
SMOOTHING_WINDOW_SIZE
```

و:

```python
SMOOTHING_REQUIRED_COUNT
```

للحصول على توازن بين:

* سرعة الاستجابة.
* ثبات الحركة.

---

# 37. المشكلة: الماوس يتحرك كثيرًا

إذا كانت حركة الماوس أسرع من المطلوب، يمكن لاحقًا تقليل:

```python
MOUSE_SPEED_SLOW
MOUSE_SPEED_NORMAL
MOUSE_SPEED_FAST
```

القيم الحالية:

```text
5
15
25
```

---

# 38. المشكلة: الماوس بطيء

يمكن لاحقًا زيادة قيم السرعة.

مثلاً:

```text
Slow
Normal
Fast
```

مع الحفاظ على التدرج بينها.

يفضل عدم تغيير القيم مباشرة أثناء الاختبارات الأولى.

---

# 39. المشكلة: Eye Click يحدث بالخطأ

تحقق من:

```python
EYE_CLICK_HOLD_TIME
```

القيمة الحالية:

```text
0.70 seconds
```

زيادة القيمة تجعل النظام أكثر تحفظًا.

تقليلها يجعل النقر أسرع ولكنه قد يزيد احتمالية النقر غير المقصود.

---

# 40. المشكلة: خطأ في تحميل النموذج

ملف النموذج:

```text
models\phase3_baseline_random_forest.joblib
```

هو Model Bundle وليس Random Forest مباشرة.

يحتوي على:

```text
model
feature_columns
classes
random_state
n_estimators
```

يتم استخراج النموذج الحقيقي باستخدام:

```python
model = model_bundle["model"]
```

وتستخدم أعمدة الخصائص المحفوظة داخل النموذج:

```python
expected_features = list(
    model_bundle["feature_columns"]
)
```

---

# 41. نموذج التعلم الآلي

النظام يستخدم:

```text
Random Forest Classifier
```

مع مجموعة الخصائص:

```text
left_norm_x
left_norm_y
right_norm_x
right_norm_y
left_ear
right_ear
```

هذه الخصائص تمثل معلومات عن:

* موقع العين.
* موقع القزحية.
* نسبة فتح العين.

---

# 42. سلامة الاستخدام

EyeControl AI يتحكم فعليًا في مؤشر الماوس.

لذلك يجب الانتباه إلى أن:

```text
MOUSE: ON
```

يعني أن حركة العين يمكن أن تحرك مؤشر النظام.

وكذلك:

```text
EYE CLICK: ON
```

يعني أن إغلاق العين بالشكل المطلوب يمكن أن ينفذ نقرة يسارية.

لذلك يفضل:

* إيقاف Mouse Control عند عدم الحاجة.
* إيقاف Eye Click عند عدم الحاجة.
* اختبار الوظائف في بيئة آمنة.
* عدم ترك النظام يعمل دون مراقبة أثناء الاختبارات.

---

# 43. حالة المشروع الحالية

المشروع يدعم حاليًا:

```text
✓ Camera
✓ Face Landmarks
✓ Eye Feature Extraction
✓ Random Forest Prediction
✓ FORWARD
✓ LEFT
✓ RIGHT
✓ UP
✓ DOWN
✓ EYES_OPEN
✓ EYES_CLOSED
✓ Prediction Smoothing
✓ Mouse Control
✓ Dynamic Mouse Speed
✓ Eye Click
✓ Validation Recording
✓ Keyboard Controls
✓ HUD
```

---

# 44. التطويرات المستقبلية

يمكن تطوير المشروع لاحقًا بإضافة:

### Calibration

إنشاء معايرة خاصة بكل مستخدم.

---

### Confidence

عرض نسبة ثقة النموذج:

```text
Prediction: LEFT
Confidence: 94%
```

ويمكن منع حركة الماوس عندما تكون الثقة منخفضة.

---

### Double Blink

استخدام رمشتين لتنفيذ:

```text
Double Click
```

---

### Right Click

إضافة Gesture لتنفيذ:

```text
Right Click
```

---

### Scroll

إضافة التحكم بالتمرير:

```text
Scroll Up
Scroll Down
```

---

### Cursor Dead Zone متقدمة

استخدام موقع العين الفعلي لإنشاء منطقة وسطية تمنع الحركة غير المقصودة.

---

### معايرة سرعة الماوس

السماح للمستخدم بتحديد:

```text
Slow
Normal
Fast
```

من إعدادات البرنامج.

---

### واجهة إعدادات

إضافة واجهة رسومية تسمح للمستخدم بتعديل:

* سرعة الماوس.
* حساسية النقر.
* مدة إغلاق العين.
* مستوى التنعيم.
* إعدادات Calibration.

---

# 45. تشغيل المشروع باختصار

إذا أردت تشغيل المشروع بسرعة:

```powershell
cd C:\Users\HP\Desktop\eyecontrol-ai
& "C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe" -m app.main
```

ثم:

```text
M → Mouse Control
C → Eye Click
Q → Exit
ESC → Exit
```

---

# 46. جدول التحكم السريع

| الأمر                  | المفتاح / الحركة             |
| ---------------------- | ---------------------------- |
| تشغيل Mouse            | `M`                          |
| إيقاف Mouse            | `M`                          |
| تشغيل Eye Click        | `C`                          |
| إيقاف Eye Click        | `C`                          |
| تحريك يسار             | النظر LEFT                   |
| تحريك يمين             | النظر RIGHT                  |
| تحريك أعلى             | النظر UP                     |
| تحريك أسفل             | النظر DOWN                   |
| Left Click             | إغلاق العين لمدة ~0.70 ثانية |
| Validation Forward     | `1`                          |
| Validation Left        | `2`                          |
| Validation Right       | `3`                          |
| Validation Up          | `4`                          |
| Validation Down        | `5`                          |
| Validation Eyes Open   | `6`                          |
| Validation Eyes Closed | `8`                          |
| خروج                   | `Q` أو `ESC`                 |

---

# 47. الخلاصة

EyeControl AI هو نظام للتحكم بالكمبيوتر باستخدام العين.

يبدأ النظام من الكاميرا، ثم يكتشف الوجه والعينين، ويستخرج الخصائص، ويستخدم نموذج Random Forest للتعرف على اتجاه العين.

بعد ذلك يتم تطبيق نظام Smoothing لتقليل التذبذب، ويمكن استخدام النتيجة للتحكم في مؤشر الماوس أو تنفيذ نقرة باستخدام إغلاق العين.

طريقة التشغيل الأساسية:

```text
تشغيل البرنامج
      ↓
ظهور الكاميرا
      ↓
التعرف على العين
      ↓
مراقبة Prediction / Smoothed
      ↓
M لتشغيل الماوس
      ↓
تحريك الماوس بالعين
      ↓
C لتشغيل Eye Click
      ↓
إغلاق العين للنقر
```

**الاختصارات الأساسية التي يجب تذكرها:**

```text
M    → Mouse
C    → Eye Click
Q    → Exit
ESC  → Exit
```

---

## إصدار الدليل

**Project:** EyeControl AI

**Document:** User Guide

**Format:** Markdown

**Main Entry Point:**

```text
app/main.py
```

**Project Path:**

```text
C:\Users\HP\Desktop\eyecontrol-ai
```
