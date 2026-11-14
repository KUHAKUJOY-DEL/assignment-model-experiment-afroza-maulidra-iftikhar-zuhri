# AI Model Experiment & Evaluation: Classical ML vs Gemini AI API untuk Sentiment Analysis Ulasan E-commerce

Individual Assignment, AI Engineering Bootcamp (ReWork)

## 1. Project Overview

Project ini membandingkan dua pendekatan AI untuk mengklasifikasikan sentimen ulasan pelanggan e-commerce (positif/negatif):

1. **Classical Machine Learning**: TF-IDF + Logistic Regression (Scikit-learn), dilatih sendiri pada dataset ulasan.
2. **Large Language Model**: Gemini AI API (`gemini-3.5-flash-lite`) dengan zero-shot prompting, tanpa training.

Kedua pendekatan dievaluasi pada **test set yang identik** dengan metrik yang sama (Accuracy, Precision, Recall, F1-Score, Confusion Matrix). Hasilnya dipakai untuk menganalisis trade-off dan menyusun rekomendasi technical approach.

## 2. Problem Statement

Tim produk e-commerce ingin membangun fitur otomatis untuk mengklasifikasikan sentimen ulasan pelanggan pada halaman produk. Sebelum memilih pendekatan untuk sistem produksi, diperlukan eksperimen awal yang membandingkan model klasik yang dilatih sendiri dengan LLM berbasis API, lalu memberikan rekomendasi berbasis data.

## 3. Objectives

- Memahami problem statement, target, dan batasan use case.
- Melatih model klasik (TF-IDF + Logistic Regression) dan memprediksi test set.
- Menjalankan inference Gemini AI API dengan prompt yang dirancang, pada test set yang sama.
- Mengevaluasi kedua pendekatan dengan Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix.
- Menganalisis trade-off performa, kecepatan, biaya, implementasi, dan keterbatasan.
- Menyusun rekomendasi technical approach berdasarkan hasil eksperimen.

**Target/label:** kolom `sentiment` dengan dua kelas, `positif` dan `negatif` (klasifikasi biner). Input yang digunakan hanya `review_text`.

**Batasan dan asumsi:**
- Dataset resmi dari instruktur digunakan apa adanya dan tidak dimodifikasi.
- Label dataset dianggap sebagai ground truth.
- `product_name` tidak dipakai sebagai fitur, karena sentimen ditentukan oleh isi ulasan.
- Gemini diakses melalui free tier, sehingga eksperimen tunduk pada rate limit API.

## 4. Dataset Description

File: `data/customer_reviews_sentiment.csv`

| Item | Nilai |
|---|---|
| Jumlah baris | 200 |
| Kolom | `review_id`, `product_name`, `review_text`, `sentiment` |
| Missing values | 0 |
| Distribusi kelas | positif 110 (55%), negatif 90 (45%) |
| Jumlah produk | 15 |
| Panjang ulasan | 6–11 kata (rata-rata 7,8) |
| **Kalimat unik** | **40** (20 positif, 20 negatif) |
| Duplikat `review_text` | 160 baris |
| Teks sama dengan label berbeda | 0 |

**Temuan kritis:** setiap kalimat diulang rata-rata 5 kali (maksimum 12) dengan `product_name` berbeda. Split acak biasa menempatkan kalimat yang sama di train dan test sekaligus, sehingga terjadi **data leakage**. Karena itu eksperimen memakai split berbasis grup (lihat Experimental Methodology).

## 5. Project Structure

```
assignment-model-experiment-afroza-maulidra-iftikhar-zuhri/
├── data/
│   └── customer_reviews_sentiment.csv     # dataset asli (tidak dimodifikasi)
├── notebook/
│   └── experiment_notebook.ipynb          # seluruh eksperimen
├── documentation/
│   └── model_comparison_summary.png       # confusion matrix 4 kombinasi
├── results/
│   ├── test_ids_group_split.csv           # review_id test set skenario utama
│   ├── test_ids_random_split.csv          # review_id test set skenario pembanding
│   ├── predictions_classic_group.csv
│   ├── predictions_classic_random.csv
│   ├── predictions_gemini_group.csv
│   ├── predictions_gemini_random.csv
│   ├── gemini_responses.csv               # cache respons mentah Gemini
│   └── metrics_summary.csv                # tabel metrik perbandingan
├── .env.example                           # template environment variable
├── .gitignore
├── README.md
└── requirements.txt
```

File `.env` (berisi API key asli) **tidak** ada di repository karena diabaikan oleh `.gitignore`.

## 6. Tools and Technologies

| Tool | Versi | Fungsi |
|---|---|---|
| Python | 3.12 | Bahasa pemrograman |
| pandas | 3.1.0 | Pemrosesan data |
| scikit-learn | 1.9.1 | TF-IDF, Logistic Regression, split, metrik |
| google-genai | 3.3.0 | SDK Gemini AI API |
| python-dotenv | 1.2.4 | Membaca API key dari `.env` |
| matplotlib | 3.11.2 | Visualisasi confusion matrix |
| ipykernel | 7.4.0 | Kernel Jupyter di VS Code |
| VS Code + Jupyter extension | - | Lingkungan pengembangan |
| Git & GitHub | - | Version control dan submission |

## 7. Experimental Methodology

```
Prepare  →  Split (2 skenario)  →  Experiment A (Classical ML)
                                →  Experiment B (Gemini API)    →  Evaluate  →  Recommend
```

**Dua skenario split** (rasio 80:20, `random_state=42`, stratifikasi kelas):

| Skenario | Metode | Train / Test | Kalimat unik di test | Baris test yang kalimatnya ada di train |
|---|---|---|---|---|
| **Utama** | `StratifiedGroupKFold` dengan grup = `review_text` | 160 / 40 | 8 | **0 (0%)** |
| **Pembanding** | `train_test_split` acak (seperti starter notebook) | 160 / 40 | 26 | **39 (97,5%)** |

- **Skenario utama** mengukur kemampuan model pada ulasan yang benar-benar baru dan menjadi **dasar kesimpulan**.
- **Skenario pembanding** dipakai untuk menunjukkan seberapa besar leakage menggelembungkan skor.
- Pada setiap skenario, model klasik dan Gemini dievaluasi pada test set yang identik. `review_id` setiap test set disimpan di folder `results/`.
- Proporsi kelas di train dan test sama-sama 55% positif : 45% negatif.

## 8. Classical Machine Learning Implementation

- **Pipeline Scikit-learn:** `TfidfVectorizer` → `LogisticRegression`. Pipeline menjamin kosakata dan IDF hanya dipelajari dari training set (anti-leakage).
- **TF-IDF:** `lowercase=True`, `ngram_range=(1, 2)`. Bigram dipakai agar frasa negasi seperti "tidak mengecewakan" tertangkap sebagai satu fitur.
- **Tanpa stopword removal:** daftar stopword Bahasa Indonesia memuat kata "tidak", yang jika dibuang akan membalik makna ulasan.
- **Logistic Regression:** `max_iter=1000`, `random_state=42`, tanpa hyperparameter tuning (diposisikan sebagai baseline).
- **Kosakata:** 333 fitur (group split), 384 fitur (random split).
- **Waktu:** training ±0,007 detik; inference ±0,0012 detik untuk 40 ulasan (±0,00003 detik per ulasan).

## 9. Gemini AI Implementation

| Parameter | Nilai | Alasan |
|---|---|---|
| Model | `gemini-3.5-flash-lite` | Flash-Lite dirancang untuk tugas sederhana bervolume tinggi dan tersedia di free tier; seri 2.5 dihindari karena dijadwalkan deprecated |
| Teknik | Zero-shot prompting | Mewakili pendekatan tanpa training dan tanpa data berlabel |
| Temperature | 0.0 | Klasifikasi membutuhkan jawaban yang paling mungkin dan konsisten antar-run |
| Format output | Satu kata: `POSITIVE` / `NEGATIVE` | Mudah divalidasi; dipetakan ke `positif` / `negatif` |

**Desain prompt** (system instruction): peran sebagai sistem klasifikasi ulasan e-commerce berbahasa Indonesia, definisi tiap label, aturan menilai makna keseluruhan (termasuk negasi), aturan untuk ulasan ambigu, dan larangan menambahkan penjelasan. Contoh di prompt sengaja **tidak** diambil dari dataset, agar isi test set tidak bocor ke prompt.

**Robustness dan efisiensi:**
- Hanya **28 kalimat unik** dari gabungan kedua test set yang dikirim ke API. Hasilnya dipetakan kembali ke 80 baris test.
- Respons disimpan di `results/gemini_responses.csv` sebagai cache. Run berikutnya tidak memanggil API ulang.
- Jeda 6 detik antar-request dan retry dengan exponential backoff (10, 20, 40 detik) untuk error 429/500/503.
- Output divalidasi. Respons gagal atau invalid ditandai `GAGAL` dan **tidak** dikonversi menjadi prediksi.

**Hasil inference:** 28/28 panggilan sukses (0 error, 0 invalid), latency rata-rata **0,727 detik per ulasan**, total 6.151 token input (±220 per ulasan), 28 token output, 0 thinking token.

## 10. Evaluation Metrics

- **Accuracy:** proporsi prediksi benar dari seluruh data. Bermakna karena kelas cukup seimbang (55:45).
- **Precision:** dari semua yang diprediksi suatu kelas, berapa yang benar.
- **Recall:** dari semua data yang benar-benar termasuk suatu kelas, berapa yang berhasil ditangkap.
- **F1-Score:** rata-rata harmonik precision dan recall.
- **Confusion Matrix:** distribusi prediksi benar/salah per kelas, dengan urutan label `["negatif", "positif"]`.

Tabel perbandingan memakai **macro average** karena kedua kelas sama penting bagi bisnis: keluhan yang terlewat merugikan penanganan komplain, dan pujian yang dianggap keluhan merusak reputasi produk. Semua metrik dihitung dengan Scikit-learn (`zero_division=0`). Prediksi `GAGAL` dikecualikan dari perhitungan dan dilaporkan di kolom `n_gagal`.

## 11. Results Comparison Table

| Skenario | Model | n_test | n_gagal | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|---|---|
| **Group split (utama)** | Classical ML | 40 | 0 | 0,6250 | 0,6278 | 0,6288 | 0,6248 |
| **Group split (utama)** | **Gemini** | 40 | 0 | **1,0000** | **1,0000** | **1,0000** | **1,0000** |
| Random split (pembanding) | Classical ML | 40 | 0 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| Random split (pembanding) | Gemini | 40 | 0 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |

**Confusion matrix Classical ML, group split** (baris = label asli, kolom = prediksi):

| | Pred negatif | Pred positif |
|---|---|---|
| **Asli negatif** | 12 | 6 |
| **Asli positif** | 9 | 13 |

Tiga confusion matrix lainnya diagonal sempurna (`[[18, 0], [0, 22]]`). Visualisasi lengkap ada di `documentation/model_comparison_summary.png`.

**Akurasi per kalimat unik** (tanpa efek pengulangan kalimat):

| Skenario | Model | Kalimat benar |
|---|---|---|
| Group split | Classical ML | 6 / 8 (75%) |
| Group split | Gemini | 8 / 8 (100%) |
| Random split | Classical ML | 26 / 26 (100%) |
| Random split | Gemini | 26 / 26 (100%) |

**Kesalahan prediksi (group split):** seluruhnya dari Classical ML, dan keduanya melibatkan negasi.

| Kalimat | Label asli | Prediksi | Baris terdampak |
|---|---|---|---|
| Produknya rapi dan bersih, tidak ada cacat sama sekali. | positif | negatif | 9 |
| Bahan murahan, tidak sesuai dengan harga yang dibayarkan. | negatif | positif | 6 |

**Interpretasi:**
- Pada skenario utama, Gemini unggul di keempat metrik. Accuracy model klasik (62,5%) hanya sedikit di atas tebakan "selalu positif" (55%).
- Recall negatif model klasik 66,7%: 6 dari 18 keluhan pelanggan lolos sebagai ulasan positif.
- Bobot model klasik menunjukkan kata netral menjadi fitur terkuat (`sesuai` +0,77, `dengan` +0,76, `sekali` −0,83). Model mempelajari pola template dataset, bukan makna, sehingga gagal pada frasa negasi.
- Pada random split, model klasik melonjak dari 62,5% menjadi 100% karena 97,5% kalimat test sudah dilihat saat training. Split acak akan menghasilkan kesimpulan keliru bahwa kedua pendekatan setara.

## 12. Trade-off Analysis

| Aspek | Classical ML | Gemini AI API |
|---|---|---|
| Performa (skenario utama) | Accuracy 62,5%, F1 0,62 | Accuracy 100%, F1 1,00 |
| Implementasi | Butuh data berlabel, preprocessing, dan training | Tanpa training; effort pada prompt, parsing, retry, rate limit, dan caching |
| Kecepatan inference | ±0,00003 detik/ulasan, lokal | ±0,727 detik/ulasan, puluhan ribu kali lebih lambat |
| Waktu untuk 1 juta ulasan (berurutan) | ±30 detik | ±202 jam (dapat diparalelkan dalam batas rate limit) |
| Biaya | Komputasi lokal, biaya API 0 | Free tier gratis (rate limit). Paid tier ±USD 0,0000684/ulasan ≈ **USD 68,40 per 1 juta ulasan** (±USD 34 dengan Batch API) |
| Fleksibilitas | Terbatas pada pola kata di training; perlu retrain untuk perubahan | Menangani kalimat baru dan negasi tanpa contoh; aturan diubah lewat prompt |
| Maintenance | Retrain berkala dan pengumpulan data berlabel | Bergantung pada layanan eksternal: deprecation model, perubahan harga dan kuota |
| Privasi data | Data tetap lokal | Data dikirim ke Google; pada free tier data dapat digunakan untuk meningkatkan produk Google |

Harga mengacu pada paid tier Gemini 3.5 Flash-Lite (USD 0,30 per 1 juta token input, USD 2,50 per 1 juta token output; [ai.google.dev/pricing](https://ai.google.dev/pricing), dicek 14 November 2026) dan rata-rata token aktual eksperimen (±220 input + 1 output per ulasan). Sekitar 99% token input berasal dari system prompt, sehingga panjang prompt adalah pendorong biaya utama.

## 13. Limitations

**Classical ML:**
- Gagal memahami negasi (seluruh kesalahan berasal dari frasa negasi).
- Mempelajari korelasi kata yang menyesatkan dari data template.
- Hanya 32 kalimat unik untuk training, sehingga kosakata sangat terbatas.

**Gemini:**
- Latensi dan rate limit membatasi throughput pada volume besar.
- Biaya bertambah linear dengan jumlah ulasan.
- Ketergantungan pada layanan eksternal (ketersediaan model, harga, kuota).
- Isu privasi data pada free tier.
- Output LLM tetap perlu divalidasi, walaupun dengan temperature 0 seluruh respons valid.

**Eksperimen:**
- Dataset hanya 40 kalimat unik berbentuk template. Test set utama hanya 8 kalimat unik, sehingga satu kalimat salah dapat menggeser accuracy hingga 22,5 poin.
- Ulasan pendek dengan sentimen jelas. Skor 100% Gemini belum membuktikan performa yang sama pada ulasan nyata yang panjang, campuran, ambigu, atau sarkastik.
- Model klasik tidak di-tuning, dan Gemini hanya diuji dengan satu desain prompt zero-shot.

## 14. Technical Recommendation

1. **Unggul berdasarkan metrik:** Gemini. Pada skenario bebas leakage, Gemini mencapai Accuracy, Precision, Recall, dan F1 sebesar 1,00 dibanding 0,62–0,63 pada Classical ML.
2. **Lebih efisien:** Classical ML. Inference lokal dalam orde mikrodetik per ulasan, tanpa jaringan maupun rate limit.
3. **Lebih ekonomis jangka panjang:** bergantung pada volume. Biaya marginal Classical ML mendekati nol, sedangkan Gemini ±USD 68 per 1 juta ulasan. Namun model klasik membawa biaya tersembunyi berupa pelabelan data dan retrain berkala.
4. **Risiko:** Classical ML gagal pada negasi dan kalimat baru, serta rentan evaluasi menyesatkan bila terjadi leakage. Gemini menanggung biaya, latensi, ketergantungan vendor, dan isu privasi.
5. **Yang dikembangkan lebih lanjut:** **Gemini sebagai pendekatan utama**, dengan Classical ML dipertahankan sebagai baseline.

**Alasan:** Gemini benar 8/8 kalimat baru, sedangkan model klasik 6/8 dengan kesalahan yang sistematis (negasi). Gemini tidak membutuhkan data berlabel, sehingga fitur bisa diluncurkan lebih cepat. Biaya terukurnya rendah untuk use case ini. Klasifikasi ulasan juga tidak harus real-time, sehingga latensi dapat diterima dan biaya bisa ditekan dengan Batch API.

**Syarat dan langkah lanjutan** (rekomendasi bersifat kondisional karena bukti berasal dari 8–28 kalimat template):
1. Uji pada sampel ulasan nyata yang lebih besar dan beragam sebelum keputusan produksi final.
2. Gunakan paid tier untuk produksi agar rate limit memadai dan data pelanggan tidak dipakai untuk pengembangan produk Google.
3. Pertahankan validasi output, retry, dan caching. Pantau biaya token dan pertimbangkan prompt yang lebih ringkas.
4. Jangka panjang (hybrid): label Gemini yang sudah diverifikasi dapat dijadikan data training untuk model klasik yang lebih kuat. Volume besar ditangani lokal dengan biaya rendah, sementara Gemini menangani kasus sulit.

## 15. Installation Instructions

Prasyarat: Python 3.12, Git, VS Code dengan extension **Python** dan **Jupyter**. Perintah di bawah untuk Windows PowerShell.

```powershell
git clone https://github.com/<username>/assignment-model-experiment-afroza-maulidra-iftikhar-zuhri.git
cd assignment-model-experiment-afroza-maulidra-iftikhar-zuhri

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Jika muncul error `running scripts is disabled on this system`, jalankan sekali:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 16. Environment Variables Configuration

1. Buat API key di [Google AI Studio](https://aistudio.google.com).
2. Salin template dan isi dengan API key:
   ```powershell
   Copy-Item .env.example .env
   ```
3. Isi file `.env` (tanpa tanda kutip, tanpa spasi di sekitar `=`):
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

`.env` tercantum di `.gitignore` sehingga tidak pernah ikut ter-commit. Notebook membaca key melalui `python-dotenv` dan hanya menampilkan 4 karakter terakhirnya.

## 17. How to Run the Notebook

1. Buka folder project di VS Code (`File → Open Folder`).
2. Buka `notebook/experiment_notebook.ipynb`.
3. Pilih kernel **Select Kernel → Python Environments → `.venv`**.
4. Jalankan **Restart → Run All**.

Catatan:
- Notebook harus dijalankan dari folder `notebook/` karena path data memakai `../data/` (perilaku default VS Code).
- Jika `results/gemini_responses.csv` sudah ada, **tidak ada panggilan API** yang dilakukan, sehingga hasil identik dengan laporan ini.
- Untuk mengulang inference Gemini dari awal, hapus `results/gemini_responses.csv` lalu Run All (±28 panggilan API, sekitar 3–4 menit).
- Smoke test API dinonaktifkan secara default (`RUN_SMOKE_TEST = False` di Section 5.2).
