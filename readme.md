# Detektor mnoštva na temelju Haralickovih značajki

Detektor je napisan u Pythonu. Detektor provodi detekciju na temelju Haralickovih značajki manjih podslika u jednoj slici.

Struktura projekta:

* dataset - skup podataka za treniranje i testiranje klasifikatora koji se koristi za treniranje i testiranje klasifikatora
* train
   + true  - pozitivni primjeri
   + false - negativni primjeri
*  test
   - true  - pozitivni primjeri
   - false - negativni primjeri
* detections - primjeri segmentacija nad testnim videosekvencama
* labels - oznake videosekvenci koje se koriste prilikom izvođenja programa
* logs - zapisi izvođenja programa
* videos - videosekvence koje se koriste prilikom izvođenja programa
* pretrained_models - prethodno trenirani modeli klasifikacije

Upute za pokretanje:

1. potrebno se pozicionirati u mapu projekta u naredbenom retku.
2. potrebno je unijeti naredbu *python main.py* te se po potrebi mogu dodati sljedeći argumenti:
* --verbose - zastavica za detaljan ispis koraka prilikom izvođenja programa
* --model-path - putanja u koju će se spremati ili iz koje će se dohvaćati klasifikacijski model. Model je u obliku *.joblib* datoteke
* --save-model - zastavica koja kada je postavljena i kada je unesen argument *model-path* sprema istrenirani model u putanju navedenu u *model-path* argumentu
* --load-model - zastavica koja kada je postavljena, dohvaća prethodno istreniran model klasifikacije i koristi ga prilikom izvođenja progama
* --log-results - zastavica kojom se označava da je potrebno spremiti mjerene podatke sustava u tekstualnu datoteku
* --log-path - kao argument se predaje putanja u koju će se spremiti tekstualna datoteka sa zabilježenim rezultatima sustava
* --window-size - visina i širina kliznog prostora
* --stride - korak koji prozor radi prilikom pomicanja udesno ili prema dolje
* --skip-frame-count - broj okvira koji se preskače prilikom slijedne obrade videosekvenci
