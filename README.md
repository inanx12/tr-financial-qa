TQuAD-fine-tuned BERTurk for Turkish financial question answering, domain transfer analysis on KAP filings.
# Türkçe Finansal Soru-Cevap: BERTurk'un Domain Transfer Analizi

**BİL 477 Derin Öğrenme — Proje Teslimi**
**Tarih:** Mayıs 2026

## Proje Özeti

Bu projede `dbmdz/bert-base-turkish-cased` (BERTurk) modeli Türkçe okuduğunu anlama veri seti TQuAD üzerinde fine-tune edilmiş, ardından kendi oluşturduğumuz Kamuyu Aydınlatma Platformu (KAP) faaliyet raporları eval setinde test edilerek **low-resource domain transfer** performansı ölçülmüştür.

**Ana bulgu:** TQuAD-trained model KAP eval set'inde F1=46.61 (TQuAD dev'de F1=73.33). %36 göreceli kayıp, Wikipedia-tarzı QA'dan finansal/kurumsal Türkçeye transfer'in sınırlı olduğunu göstermektedir.

## Kullanılan Yöntemler

- **Base model:** dbmdz/bert-base-turkish-cased
- **Train set:** TQuAD v0.1 (8308 train + 892 dev, SQuAD format)
- **Eval set (özgün):** 5 BIST şirketinin 2025 KAP faaliyet raporlarından 60 soru-cevap (12 soru/şirket)
- **Şirketler:** ASELSAN (ASELS), BİM (BIMAS), CW Enerji (CWENE), Katılımevim (KTLEV), YEO Teknoloji (YEOTK)
- **Fine-tune:** 2 epoch, lr=3e-5, batch=16/device, fp16, GPU: Kaggle T4×2
- **Eval pipeline:** Sliding window tokenization (max_len=384, stride=128) + offset-mapping based postprocessing + SQuAD metric

## Eval Set — Soru Taksonomisi

Kendi oluşturduğumuz 60 sorulu KAP eval set'inde 5 farklı soru tipi:

| Tip | Sayı | Örnek |
|---|---|---|
| Factoid | 21 | "ASELSAN'ın Genel Müdürü kimdir?" |
| Numerik | 28 | "BİM'in 2025 mağaza sayısı kaçtır?" |
| List | 7 | "FİLE'nin kendi markaları hangileridir?" |
| Neden-sonuç | 3 | "Bedelli sermaye artırımı neden iptal edilmiştir?" |
| Definitional | 1 | "LFP kısaltması neyin kısaltmasıdır?" |

## Sonuçlar

### Genel
| Eval Set | F1 | EM |
|---|---|---|
| TQuAD dev (savasy baseline) | 81.08 | 62.78 |
| TQuAD dev (bizim ft_model) | 73.33 | 52.13 |
| **KAP eval (bizim ft_model)** | **46.61** | **30.00** |

### Soru tipi bazında
| Tip | F1 | EM | n |
|---|---|---|---|
| Factoid | 57.74 | 38.10 | 21 |
| Numerik | 47.50 | 35.71 | 28 |
| Neden-sonuç | 27.74 | 0.00 | 3 |
| List | 24.40 | 0.00 | 7 |
| Definitional | 0.00 | 0.00 | 1 |

### Şirket bazında
| Şirket | F1 | EM |
|---|---|---|
| BIMAS | 61.67 | 50.00 |
| KTLEV | 54.37 | 33.33 |
| CWENE | 53.91 | 33.33 |
| YEOTK | 41.73 | 16.67 |
| ASELS | 21.40 | 16.67 |

## Repo Yapısı
├── notebook/
│   └── bil-tr-finance-qa-day1-sanity.ipynb   # ana pipeline
├── eval_data/
│   ├── kap_eval_asels.json                   # 12 soru
│   ├── kap_eval_bimas.json                   # 12 soru
│   ├── kap_eval_cwene.json                   # 12 soru
│   ├── kap_eval_ktlev.json                   # 12 soru
│   └── kap_eval_yeotk.json                   # 12 soru
├── results/
│   ├── kap_eval_results.json                 # F1/EM breakdown
│   └── kap_eval_per_question.csv             # her sorunun pred/gold/skor
├── docs/
│   └── analiz_raporu.pdf                     # proje teslim raporu
└── README.md

## Notebook Nasıl Çalıştırılır

Notebook **Kaggle T4×2 GPU** üzerinde geliştirilmiş, Python 3.12 + Transformers 5.0.0 + datasets + evaluate kütüphaneleri ile çalışır.

### Pipeline:
1. **Cell 1-3:** Dataset yükleme (TQuAD), savasy/dbmdz tokenizer'lar
2. **Cell 4:** Savasy baseline F1/EM (sanity check)
3. **Cell 5:** Tokenization + sliding window preprocess
4. **Cell 6a-6b:** Postprocessing fonksiyonu + savasy üzerinde validation
5. **Cell 7:** Fine-tune (2 epoch, ~15 dk)
6. **Cell 7b:** Modeli diskten yükle (re-train atlama)
7. **Cell 8+:** KAP eval pipeline + breakdown

### Model dosyası:
Boyut nedeniyle GitHub'da değil. Fine-tune'u tekrar etmek için Cell 7'yi çalıştırın (T4 GPU ile yaklaşık 15dk). Veya bana mail atın, model.safetensors yollayayım.

## Veri Kaynakları

- **TQuAD v0.1:** https://github.com/TQuad/turkish-nlp-qa-dataset
- **KAP raporları:** https://www.kap.org.tr (kamuya açık, telif gereği bu repoda PDF yok)

## Referanslar

- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
- Schweter, S. (2020). BERTurk - BERT models for Turkish
- TQuAD: Turkish Question Answering Dataset
- HuggingFace Transformers, datasets, evaluate libraries

## Yazar

İnan Esen — Ostim Teknik Üniversitesi — Bilgisayar Mühendisliği

inanesen2004@gmail.com
