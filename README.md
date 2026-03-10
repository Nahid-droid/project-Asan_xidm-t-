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
git clone https://github.com/Nahid-droid/VOC_Detection_Nahid_Mammadov.git
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

**bicycle (AP50=0.812):** Ən yüksək nəticəni verən klassdır. Velosipedlər digər klasslardan vizual cəhətdən fərqlidir — incə çərçivə, iki təkər və spesifik forma modelin tanımasını asanlaşdırır. Həmçinin annotasiya keyfiyyəti yüksək idi, çünki velosipedlər adətən aydın görünürdü.

**bus (AP50=0.665):** İkinci ən yaxşı nəticəni verən klassdır. Avtobuslar böyük, düzbucaqlı formaya malik olduğundan model onları nisbətən asanlıqla lokalizasiya edir. Confusion matrix-ə görə avtobusların 9%-i maşın kimi detect edildi — hər ikisi dörd təkərli nəqliyyat vasitəsi olduğundan bu anlaşılandır.

**car (AP50=0.691):** Ən çox annotasiyaya malik klass olmasına (194 annotasiya) baxmayaraq orta nəticə verdi. Bunun əsas səbəbi müxtəlif açılardan çəkilmiş şəkillərdir — qarşıdan, arxadan, yandan çəkilmiş maşınlar model üçün fərqli görünür. Bundan əlavə, confusion matrix-ə görə maşınların 40%-i detect edilmədi.

**motorbike (AP50=0.558):** Ən zəif nəticəni verən klassdır. İki əsas problem var: birincisi, az annotasiya sayı (88) — digər klasslara nisbətən model bu klassı daha az nümunə üzərində öyrəndi. İkincisi, motosikletlər bicycle ilə vizual oxşarlığa malikdir — confusion matrix göstərir ki, motosikletlərin 16%-i bicycle kimi, 50%-i isə background kimi buraxıldı.

---

## Vizuallaşdırmalar

### Training Loss Curve
![Training Curves](runs/exp1/results.png)

Training boyu bütün loss-lar (box, cls, dfl) həm train, həm val setdə sabit azalır. Val/cls_loss-da ilk epochlarda kəskin sıçrayış görünür — bu warmup mərhələsinin normal davranışıdır, lr0=0.005 ilə başlayan model ilk epochlarda qeyri-stabil olur. Sonrakı epochdan etibarən val loss train loss-a paralel azalmağa başlayır — overfitting  yoxdur. mAP@0.5 ilk epochlardan etibarən sürətlə artır, 75-ci epocha qədər yüksəlişini davam etdirir — model hələ öyrənməkdə idi, daha çox epoch mümkün ola bilərdi.

### Precision-Recall Curve
![PR Curve](notebooks/runs/detect/val3/BoxPR_curve.png)

- motorbike ən aşağı AP (=0.558) malikdir— az data və bicycle ilə oxşarlıq
- Ümumi mAP@0.5 = 0.681
PR curve hər klass üçün confidence threshold dəyişdikcə Precision-Recall balansını göstərir. Bicycle əyrisi ən geniş sahəni əhatə edir (AP=0.812) — yüksək recall dəyərlərinə (0.8+) qədər precision yüksək qalır. Motorbike əyrisi ən sürətlə aşağı düşür — artıq recall=0.4-də precision 0.44-ə enir, bu klassın ən çətin olduğunu təsdiqləyir. Bus əyrisi maraqlıdır: recall=0.5-ə qədər precision 0.80+ saxlayır, lakin sonra kəskin düşür — model ya çox əminliklə tapır, ya da ümumiyyətlə buraxır.

### Confusion Matrix
![Confusion Matrix](notebooks/runs/detect/val3/confusion_matrix_normalized.png)

Matrix bir neçə vacib məlumat verir. Bicycle-ın 74%-i düzgün detect edildi, 15%-i background kimi buraxıldı, 12%-i motorbike ilə qarışdı — bu qarışıqlıq gözləniləndir. Bus-un 68%-i düzgün detect edildi, 18%-i background kimi buraxıldı, 9%-i car kimi predict edildi. Car-ın 62%-i düzgün detect edildi, amma 40%-i background kimi buraxıldı — bu ən yüksək "miss" rəqəmidir. Motorbike ən problemli  klassdır: yalnız 37%-i düzgün detect edildi, 50%-i background kimi buraxıldı, 16%-i bicycle kimi predict edildi.

### Predicted Bounding Box-lar (Test Set)
![Predictions](predictions.png)

Test setindən 10 şəkil üzərindəki predict nəticələri bir neçə müşahidəni ortaya qoyur. Yaxın planda olan aydın obyektlər yüksək confidence ilə detect edilir: motorbike 0.74, bus 0.86, car 0.90. Uzaqda və ya qismən görünən obyektlərdə confidence aşağı düşür: car 0 30, bicycle 0.60-0.63. 2008_002374.jpg şəkilindəki uçan motosiklet maraqlı haldır — model onu  aşağı confidence ilə detect etdi, çünki bu pozisiya training datada nadir idi. 2008_000619.jpg-də bir kadrda həm bus, həm car eyni anda detect edildi — model overlapping səhnələri də idarə edə bilir.

---

## Xəta Analizi

**Confusion matrix-dən əldə edilən müşahidələr:**

1. **Motorbike → background (0.50):** Ən böyük problemdir — motosikletlərin yarısı detect edilmədi. Az annotasiya (88) və müxtəlif açılardan çəkilmiş şəkillər modelin bu klassı öyrənməsini çətinləşdirdi.

2. **Motorbike → bicycle qarışıqlığı (0.16):** Hər iki klass iki təkərli və oxşar formaya malik olduğundan, xüsusilə uzaq məsafədən çəkilmiş şəkillərdə model fərqi tapa bilmir.

3. **Car → background (0.40):** Maşınların 40%-i detect edilmədi. Müxtəlif açılardan (qarşıdan, arxadan, yandan) çəkilmiş şəkillər və sıx park səhnələrindəki overlapping obyektlər əsas səbəbdir.

4. **Bus → car qarışıqlığı (0.09):** Hər ikisi böyük, düzbucaqlı nəqliyyat vasitəsidir. Minivan tipli avtobus şəkillərində bu qarışıqlıq xüsusilə anlaşılandır.

5. **Bicycle ↔ Motorbike simmetrik qarışıqlığı:** Bicycle-ın 12%-i motorbike, motorbike-ın 16%-i bicycle kimi predict edildi — qarışıqlıq iki tərəflidir.

6. **Yaxın vs uzaq obyektlər:** Model yaxın planda olan obyektləri yüksək confidence ilə detect edir (bus 0.86, car 0.90), uzaq və qismən görünən obyektlərdə isə confidence aşağı düşür (car 0.30).

**Detect edilməyən şəkillərin analizi:**

![Failure Cases](failure_cases.png)

Detect edilməyən şəkillərin analizi üç əsas xəta kateqoriyasını ortaya qoyur. Birincisi, qeyri-adi pozisiya və bucaq — havada uçan motosiklet və kəskin döngədə əyilmiş motosiklet kimi training datada nadir olan hallarda model obyekti tanıya bilmir. İkincisi, qismən örtülmüş obyektlər — insanlar, dirəklər və ya digər obyektlər tərəfindən örtülmüş velosiped və motosikletləri model detect etmir, çünki görünən hissə kifayət qədər fərqli feature vermir. Üçüncüsü, həddən artıq yaxın çəkilmiş şəkillər — velosipedin yalnız çərçivə və dişli hissəsinin göründüyü şəkillərdə model tam obyekti tanıya bilmir.

**Ümumi nəticə:** Bu xəta hallarının hamısı daha çox müxtəlif annotasiya nümunəsi — qeyri-adi açılar, qismən görünən obyektlər, həddən yaxın çəkilmiş şəkillər — ilə həll oluna bilər.

---

## LLM İstifadəsi

**İstifadə edilən LLM:** Claude Sonnet 4.5 (claude.ai)

**Nə üçün istifadə edildi:** Bütün layihə boyu kod yazmaq, debug etmək,  konseptləri dərindən anlamaq, alternativ yanaşmaları müzakirə etmək və README hazırlamaq üçün istifadə edildi.

**Əsas promptlar və istifadə:**

1. *"PASCAL VOC XML-dən YOLO formatına çevirmə, hər klasdan bərabər sayda şəkil seçmə və train/val/test bölgüsü edən notebook yaz. Seed=42 istifadə et, class imbalance-a diqqət et"* → `01_data_prep.ipynb`(CVAT-dan əvvəl) kodu əsas götürülüb path-lar və parametrlər özüm tənzimlədim.

2. *"CVAT YOLO 1.1 export-undan label-ları götür, val/test üçün VOC XML-dən avtomatik label yarat, data.yaml yarat və annotasiya statistikasını vizuallaşdır"* → `01_data_prep.ipynb`(CVAT-dan sonra) kodu istifadə edildi, CVAT export strukturu debug mərhələsində əlavə tənzimləndi.

3. *"YOLOv8n-i CPU-da fine-tune edən notebook yaz. Windows-da Jupyter-də  multiprocessing xətasının qarşısını al, training progress-i izlə"* → `02_training.ipynb` kodu istifadə edildi. Hyperparameter seçimi (lr0, epochs, patience, flipud) müstəqil qərar verdim və hər birini əsaslandırdım.

4. *"Test setində qiymətləndirmə notebook-u yaz: mAP, precision, recall, F1 klass üzrə göstər, inference sürəti ölç, PR curve çək, predicted bbox-ları vizuallaşdır, detect edilməyən şəkilləri ayrıca göstər"* → `03_evaluation.ipynb` kodu istifadə edildi. PR curve Ultralytics tərəfindən avtomatik yaranmadığı üçün `test_results.box` obyektindən manual olaraq çəkildi.

5. *"Confusion matrix-dən və failure cases şəkillərindən konkret xəta kateqoriyalarını çıxar"* → Xəta analizi bölməsi bu müzakirə əsasında yazıldı.

LLM tərəfindən yazılmış bütün kodlar əvvəlcə nəzərdən keçirilib, lazımi yerlərdə düzəldilərək sonra fayllara yazıldı. Məsələn `01_data_prep.ipynb`-də LLM şəkilləri klasslara görə seçərkən hər şəkili yalnız bir klassa aid edirdi — amma PASCAL VOC-da bir şəkildə eyni anda həm car, həm bus ola bilər. Bu data itkisinə səbəb olurdu. Bunu müstəqil aşkar edərək deduplication məntiqini əlavə etdim: şəkil artıq seçilibsə, ikinci dəfə əlavə edilmir, beləliklə hər şəkil yalnız bir dəfə dataset-ə daxil olur.

**Müstəqil qərarlar:** Model seçimi, LLM-in bəzi ilkin tövsiyələrinə (epoch=50, lr0=0.01) baxmayaraq, 50 epoch-dan sonra val loss-un platoya çatdığını müşahidə edərək lr0=0.005 və epochs=75 ilə yenidən train etdim — mAP@0.5 0.436-dan 0.681-ə yüksəldi. flipud=0.0 qərarını da müstəqil verdim: avtomobillər real mühitdə heç vaxt üst-aşağı olmur, bu augmentasiya mənasızdır. README.md-ni bəzi müzakilər çıxmaqla tamamilə özüm yazdım.

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
