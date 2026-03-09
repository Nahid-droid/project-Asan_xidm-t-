# VOC_Detection_Nahid_Mammadov

## Ümumi Baxış

Bu layihə urban mühitdə hərəkət edən avtonom sistem üçün object detection modeli hazırlamağı hədəfləyir. PASCAL VOC 2012 datasetindən seçilmiş 4 klass üzərindəki şəkillər CVAT alətindən istifadə edilərək əl ilə annotasiya edilmiş, ardından YOLOv8n modeli bu data üzərində fine-tune edilmişdir.

**Hədəf klasslar:**

| Klass | VOC Adı |
|---|---|
| Avtomobil | car |
| Avtobus | bus |
| Velosiped | bicycle |
| Motosiklet | motorbike |

**Klass üzrə annotasiya sayı (train set):**

| Klass | Train | Val | Test | Cəmi |
|---|---|---|---|---|
| bicycle | 99 | 18 | 19 | 136 |
| bus | 123 | 28 | 25 | 176 |
| car | 194 | 33 | 53 | 280 |
| motorbike | 88 | 26 | 32 | 146 |
| **Cəmi** | **504** | **105** | **129** | **738** |

---

## Qurulum

### Tələblər

- Python 3.10+
- Anaconda (tövsiyə olunur)
- PASCAL VOC 2012 dataseti

### Addım-addım

**1. Repo-nu clone et:**
```bash
git clone https://github.com/Nahid-droid/project-Asan_xidm-t-.git
cd project
```

**2. Conda environment yarat:**
```bash
conda create -n your_env python=3.11
conda activate your_env
```

**3. Asılılıqları qur:**
```bash
pip install -r requirements.txt
```

**4. PASCAL VOC 2012 datasetini yüklə:**
- [Rəsmi sayt](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) və ya [Kaggle](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012)
- Aşağıdakı path-a çıxart:
```
C:\Users\[username]\Desktop\VOCtrainval_11-May-2012\VOCdevkit\VOC2012
```

**5. `notebooks/01_data_prep.ipynb`-də path-ları dəyiş:**
```python
VOC_ROOT     = Path(r'C:\Users\[username]\Desktop\VOCtrainval_11-May-2012\VOCdevkit\VOC2012')
PROJECT_ROOT = Path(r'C:\Users\[username]\Desktop\project')
```

**6. Notebook-ları ardıcıl run et:**
```
notebooks/01_data_prep.ipynb          # CVAT-dan əvvəl — şəkilləri böl
# → CVAT-da annotasiya et, YOLO 1.1 formatında export et
notebooks/01_data_prep_part2.ipynb    # CVAT-dan sonra — labellar, data.yaml
notebooks/02_training.ipynb           # Model training
notebooks/03_evaluation.ipynb         # Qiymətləndirmə
```

---

## Annotasiya Prosesi

**Alət:** CVAT (cvat.ai)

**Proses:**
1. `01_data_prep.ipynb` run edilərək hər klasdan 100 şəkil seçildi və `data/images/train/` qovluğuna kopyalandı
2. CVAT-da `VOC_Detection_Nahid_Mammadov` adlı project yaradıldı, 4 label əlavə edildi, bbox çəkilən zaman label-lar qarışdırılmasın deyə hər birinə ayrı rəng verildi
3. `data/images/train/` qovluğundakı ~280 şəkil CVAT-a yükləndi
4. Rectangle (bounding box) aləti ilə hər obyekt annotasiya edildi
5. YOLO 1.1 formatında export edildi

**Annotasiya müddəti:** ~3 saat

**Çətin annotasiya halları:**
- Bəzi şəkillərdə hədəf obyektlər çox kiçik və ya uzaqda görünürdü — bu hallarda şəkil böyüdülərək bbox mümkün qədər dəqiq çəkildi
- Bəzi şəkillərdə obyektlər bir-birinə girmiş (overlapping) vəziyyətdə idi — hər obyekt üçün ayrıca bbox çəkildi
- Bəzi şəkillərdə hədəf klass ümumiyyətlə görünmürdü (məsələn, maşının içindən çəkilmiş şəkillər) — bu şəkillər annotasiya edilmədən keçildi
- Val və test şəkilləri üçün annotasiyalar VOC XML-dən avtomatik çevrildi — qaydalarda qeyd edildiyi kimi yalnız train annotasiyaları manual edildi
---

## Model Seçimi

**Seçilən model: YOLOv8n**

| Xüsusiyyət | YOLOv8n | RT-DETR-l |
|---|---|---|
| Arxitektura | One-stage, anchor-free CNN | Transformer-based |
| Training sürəti (CPU) | ~4 saat / 75 epoch | ~8+ saat / 75 epoch |
| Inference sürəti | ~95 ms/şəkil (CPU) | ~300+ ms/şəkil (CPU) |
| Güclü tərəfi | Sürət, kiçik obyektlər | Kontekst anlayışı, izdihamlı səhnələr |
| Pretrained weights | yolov8n.pt (COCO) | rtdetr-l.pt (COCO) |

**Seçim əsaslandırması:**

### Tapşırığın şərtləri və model seçimi məntiqi

Model seçimi "ən güclü model" kriteriyasına deyil, **tapşırığın real şərtlərinə** əsaslanmalıdır. Bu tapşırıqda şərtlər aşağıdakı idi: 4 klass, ~400 şəkil, CPU mühiti, 1 həftə deadline. Bu şəraitdə ən vacib faktorlar sürət, yüngüllük və məhdud data ilə yaxşı ümumiləşmə qabiliyyətidir.

### YOLOv8n nədir?

YOLOv8n (You Only Look Once v8 nano) — Ultralytics tərəfindən hazırlanmış **one-stage, anchor-free CNN** detektordur. "Nano" versiyası ailəsinin ən yüngül modelidir (~3M parametr). One-stage arxitektura şəkli bir dəfə işlədir — lokalizasiya və klassifikasiya eyni anda həll edilir. Bu onu CPU-da praktiki edir.

Model COCO dataseti üzərində pretrained weights ilə gəlir — yəni model artıq ümumi vizual xüsusiyyətlər öyrənib. Bizim tapşırıqda bu weights-dən fine-tuning edildi ki, model 4 hədəf klassı tanısın.

### Alternativ model: RT-DETR-l

RT-DETR-l (Real-Time Detection Transformer large) — transformer əsaslı detektordur (~32M parametr). Self-attention mexanizmi sayəsində şəkildəki hər bölgə digər bütün bölgələrlə əlaqəni hesablayır. Bu ona izdihamlı şəhər səhnələrində (bir kadrda çox sayda maşın, avtobus) güclü kontekst anlayışı verir.

### Müqayisə

| Meyar | YOLOv8n | RT-DETR-l |
|---|---|---|
| Arxitektura | One-stage CNN | Transformer encoder-decoder |
| Parametr sayı | ~3M | ~32M |
| CPU training (75 epoch) | ~4 saat | ~15+ saat |
| Inference sürəti (CPU) | ~95 ms/şəkil | ~300+ ms/şəkil |
| Kiçik dataset (~400 şəkil) | ✅ Uyğundur | ⚠️ Daha çox data tələb edir |
| İzdihamlı səhnələr | ⚠️ Orta | ✅ Güclü |
| Uzaq/kiçik obyektlər | ⚠️ Orta | ✅ Global attention ilə daha yaxşı |
| Overfitting riski (az data) | Aşağı | Yüksək |

### Trade-off-lar

**YOLOv8n-in üstünlükləri bu tapşırıqda:**
- 3M parametr — 400 şəkil ilə öyrətmək üçün optimaldir, overfitting riski azdır
- CPU-da ~4 saatda training tamamlandı — praktiki və reproducible nəticə
- Inference sürəti ~95ms — real-time avtonom sistem tələblərinə yaxındır
- One-stage arxitektura sadədir, debug etmək asandır

**YOLOv8n-in zəif tərəfləri:**
- Lokal CNN feature-larına əsaslanır — bir kadrda çox sayda bir-birinə girmiş obyektlərdə RT-DETR-dən geri qalır
- Global konteksti transformer qədər yaxşı anlamır — confusion matrix-dən göründüyü kimi motorbike-ı 16% hallarda bicycle ilə qarışdırır
- Çox kiçik və uzaq obyektlərdə zəifdir — test setindəki bəzi şəkillərdə aşağı confidence göstərdi

**RT-DETR-in bu tapşırıqda impraktik olmasının səbəbləri:**
- 32M parametri 400 şəkil ilə öyrətmək underfitting və ya overfitting riskini artırır — transformer modelləri güclü nəticə üçün daha böyük dataset tələb edir
- CPU-da ~15+ saat training — 1 həftəlik deadline şəraitində yenidən eksperiment aparmaq imkansızlaşır
- Inference sürəti ~300ms — real-time avtonom sistem üçün qəbuledilməzdir

### Nəticə
Optimal model ən güclü model deyil — şəraitə ən uyğun modeldir. 400 şəkil, CPU mühiti və 1 həftəlik deadline şəraitində YOLOv8n ən rasional seçimdir. RT-DETR-in üstünlükləri (izdihamlı səhnələr, kiçik uzaq obyektlər) yalnız böyük dataset və GPU mühitində tam reallaşa bilər.


---

## Training Qeydləri

**Hardware:** Intel Core i5-1135G7 2.40GHz (CPU)

**Training müddəti:** 3 saat 55 dəqiqə

**Sabit parametrlər:**

| Parametr | Dəyər |
|---|---|
| Model | yolov8n.pt (COCO pretrained) |
| Input size | 640×640 |
| Epochs | 75 |
| Batch size | 8 |
| Seed | 42 |
| Klass sayı | 4 |

**Sərbəst hyperparametrlər və əsaslandırma:**

| Hyperparametr | Dəyər | Əsaslandırma |
|---|---|---|
| Learning rate (lr0) | 0.005 | Default 0.01-dən aşağı — pretrained model üçün kiçik LR daha stabil fine-tuning verir |
| Optimizer | AdamW | Adaptiv LR ilə daha stabil konvergensiya |
| Patience | 25 | Modelin platoda qalmasına daha çox şans verir |
| flipud | 0.0 | Avtomobillər həmişə düz olur — üst-aşağı flip mənasızdır |
| fliplr | 0.5 | Üfüqi flip urban mühitdə realdır |
| mosaic | 1.0 | Müxtəlif ölçü və kontekstdə obyektlər öyrənmək üçün |
| scale | 0.5 | Müxtəlif məsafələrdəki obyektlər üçün |
| hsv_h/s/v | 0.015/0.7/0.4 | Müxtəlif işıq şəraitləri üçün |

---

## Nəticələr

**Test set metrikları (augment=True, conf=0.20):**

| Metrika | Dəyər |
|---|---|
| mAP@0.5 | 0.6814 |
| mAP@0.5:0.95 | 0.4974 |
| Mean Precision | 0.8028 |
| Mean Recall | 0.5771 |
| Mean F1 | 0.6715 |
| Inference sürəti | ~95 ms/şəkil (CPU) |

**Klass üzrə nəticələr:**

| Klass | Precision | Recall | F1 | mAP@0.5 |
|---|---|---|---|---|
| bicycle | 0.875 | 0.737 | 0.800 | 0.812 |
| bus | 0.827 | 0.600 | 0.695 | 0.665 |
| car | 0.759 | 0.534 | 0.627 | 0.691 |
| motorbike | 0.751 | 0.438 | 0.553 | 0.558 |
| **Ortalama** | **0.803** | **0.577** | **0.672** | **0.681** |

**Klass üzrə müşahidələr:**
- `bicycle` ən yaxşı nəticə verdi (AP50=0.812) — nisbətən fərqli vizual görünüşü var
- `bus` yaxşı nəticə verdi (AP50=0.665) — böyük obyekt, tanımaq asandır
- `car` orta nəticə verdi (AP50=0.691) — çox sayda annotasiya olmasına baxmayaraq müxtəlif açılardan çəkilmiş şəkillər çətinlik yaratdı
- `motorbike` ən zəif nəticə verdi (AP50=0.558) — az annotasiya (88) və bicycle ilə vizual oxşarlıq

---

## Vizuallaşdırmalar

### Training Loss Curve
![Training Curves](runs/exp1/results.png)

Bütün loss-lar (box, cls, dfl) həm train, həm val setdə sabit azalır. Val loss train loss-a paralel azalır — overfitting yoxdur. mAP@0.5 epoch 25-dən etibarən sürətlə artmağa başlayır.

### Precision-Recall Curve
![PR Curve](notebooks/runs/detect/val3/BoxPR_curve.png)

- bicycle ən yüksək əyri altı sahəsinə malikdir (AP=0.812) — model bu klassı ən yaxşı tanıyır
- motorbike ən aşağı əyri altı sahəsi (AP=0.558) — az data və bicycle ilə oxşarlıq
- Ümumi mAP@0.5 = 0.681

### Confusion Matrix
![Confusion Matrix](notebooks/runs/detect/val3/confusion_matrix_normalized.png)

### Predicted Bounding Box-lar (Test Set)
![Predictions](predictions.png)

---

## Xəta Analizi

**Confusion matrix-dən əldə edilən müşahidələr:**

1. **Motorbike → background (0.50):** Motosikletlərin 50%-i background kimi buraxıldı — bu ən böyük problemdir. Az annotasiya (88) modelin motorbike-ı tanımasını çətinləşdirir.

2. **Motorbike → bicycle qarışıqlığı (0.16):** Motosikletlərin 16%-i bicycle kimi detect edildi — hər iki klass iki təkərli, bənzər formaya malikdir.

3. **Car → background (0.40):** Maşınların 40%-i detect edilmədi — müxtəlif açı və ölçülərdən çəkilmiş şəkillər modeli çaşdırır.

4. **Bus → car qarışıqlığı (0.09):** Avtobusların 9%-i maşın kimi detect edildi — hər ikisi böyük dörd təkərli nəqliyyat vasitəsidir.

5. **Predictions şəkilinə görə:** Model yaxın planda olan obyektləri yüksək confidence ilə (0.74-0.90) detect edir, uzaq və qismən görünən obyektlərdə isə aşağı confidence (0.30-0.40) göstərir.

---

## LLM İstifadəsi

**İstifadə edilən LLM:** Claude Sonnet 4.5 (claude.ai)

**Nə üçün istifadə edildi:** Bütün layihə boyu kod yazmaq, debug etmək, konseptləri anlamaq və README hazırlamaq üçün

**Əsas promptlar və istifadə:**

1. *"PASCAL VOC XML-dən YOLO formatına çevirmə + train/val/test bölgüsü edən notebook yaz"* → `01_data_prep.ipynb` kodu əsasən istifadə edildi, path-lar dəyişdirildi

2. *"CVAT export-undan sonra val/test üçün VOC XML-dən label yaradan, data.yaml yaradan notebook yaz"* → `01_data_prep_part2.ipynb` kodu istifadə edildi

3. *"YOLOv8n fine-tuning notebook-u yaz, CPU üçün optimallaşdır"* → `02_training.ipynb` kodu istifadə edildi, parametrlər özüm seçdim və əsaslandırdım

4. *"Test setində qiymətləndirmə, inference sürəti, PR curve, predicted bbox vizuallaşdırma notebook-u yaz"* → `03_evaluation.ipynb` kodu istifadə edildi, PR curve hissəsi debug edildi

**İstifadə edilməyən hissələr:** LLM-in bəzi tövsiyələri (məsələn epoch sayı, batch size) özüm dəyişdirdim — model haqqında öz qərarlarımı verdim.

---

## İstifadə Edilən Alətlər

| Alət | Məqsəd |
|---|---|
| CVAT (cvat.ai) | Manual bounding box annotasiyası — tapşırıq tələbi |
| Ultralytics YOLOv8 | Model training və evaluation framework |
| VS Code + Jupyter | Kod yazmaq və notebook idarəetməsi |
| Anaconda | Python environment idarəetməsi |
| Claude (claude.ai) | Kod yazmaq, debug, konsept izahı |
| CPU(ntel Core i5-1135G7 2.40GHz) | Reproducibility, Data idarəetməsi, Sessiya məhdudiyyəti |


### Niyə lokal mühit, Colab/Kaggle yox?

Bu layihə lokal Windows mühitində (CPU) icra edildi. Bunun bir neçə əsaslandırması var:

**Reproducibility:** Lokal mühitdə `conda env`, paket versiyaları və bütün fayllar sabitdir. Colab-da session yenidən başladıqda mühit sıfırlanır, paketlər yenidən qurulur — bu reproducibility-ni çətinləşdirir.

**Data idarəetməsi:** ~400 şəkil, CVAT annotasiyaları və nəticə faylları lokal olaraq idarə edildi. Colab-a hər sessiyada data yükləmək (Google Drive sync və ya upload) əlavə vaxt və komplekslik yaradardı.

**Sessiya məhdudiyyəti:** Colab pulsuz hesabda GPU sessiyanı ~2-4 saatdan sonra kəsir.Yəni mən T4 GPU-dan istifadə etsəm və hər hansı bir problem yaransa və ya hər hansı hyperparameterləri dəyişmək istəsəm kodu yenidən run etməliyəm ki, buda zaman limitinin keçməsi ilə nəticələnə bilər və sessiya kəsilsəydi nəticələr itə bilərdi. Lokal mühitdə training fasiləsiz davam etdi.

**Nəticə:** 
Training uzun çəkdi (~4 saat), amma mühit tam nəzarət altında idi, nəticələr etibarlı və reproducible oldu.
---

## Çətinliklər və Həllər

| Çətinlik | Həll |
|---|---|
| CVAT-da pulsuz hesabda şəkilləri export etmək mümkün olmadı | Yalnız label `.txt` faylları export edildi, şəkillər artıq lokal olaraq mövcud idi |
| Val/test şəkilləri üçün CVAT annotasiyası tələb olunmurdu, amma label lazım idi | VOC XML-dən avtomatik YOLO formatına çevirdik |
| PR curve Ultralytics tərəfindən avtomatik yaranmadı | `test_results.box` obyektindən manual olaraq qraf çəkildi |
| CPU-da training uzun çəkdi | Batch size=8, workers=0 ilə optimallaşdırıldı |
| İlk 50 epoch-da mAP=0.50 idi | lr0=0.005, flipud=0.0, epochs=75 ilə yenidən train edildi → mAP=0.681 |

---

## Gələcək Addımlar

Daha çox vaxt olsaydı:

1. **Daha çox annotasiya** — xüsusilə `motorbike` klassi üçün 100 əvəzinə 200+ annotasiya edərdim. Bu klassın AP50-si (0.558) digərlərindən aşağıdır — data artımı birbaşa nəticəni yaxşılaşdırardı.

2. **GPU-da training** — eyni parametrlərlə GPU-da train etsəydim batch size 16-ya qaldırıb daha stabil gradient hesablaması əldə edərdim.

3. **RT-DETR ilə müqayisə** — GPU mövcud olsaydı RT-DETR-l modelini də train edib nəticələri müqayisə edərdim — xüsusilə izdihamlı şəhər səhnələrində transformer modelinin üstünlüyünü ölçmək maraqlı olardı.

4. **Val/test annotasiyalarını əl ilə etmək** — biz val/test üçün VOC XML-dən avtomatik çevirdik. Əl ilə annotasiya daha dəqiq qiymətləndirmə verərdi.

5. **Hyperparameter axtarışı** — `lr0`, `mosaic`, `scale` parametrləri üçün sistematik axtarış (Optuna və ya Ray Tune ilə) aparardım.
