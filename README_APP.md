# CineMatch AI — Aplikasi Rekomendasi Film Berbasis Content-Based Filtering

Aplikasi web rekomendasi film real-time yang dibangun menggunakan **Django**, **Scikit-Learn**, serta penggabungan dataset **MovieLens (ml-latest-small)** dan metadata visual **TMDb (The Movies Dataset)**.

Aplikasi ini berfokus 100% pada metode **Content-Based Filtering (Item-to-Item)** dengan fondasi ilmiah:
- **TF-IDF (Term Frequency — Inverse Document Frequency)**
- **Kalkulasi Sudut Vektor (Cosine Similarity)**
- **Evaluasi Presisi (Precision@K)**

---

## 1. Landasan Teori & Formulasi Matematis

### A. Ekstraksi Fitur (Feature Soup) & Pembobotan TF-IDF
Setiap film diwakili oleh gabungan karakteristik konten (*feature soup*):
$$\text{Soup} = (\text{Genres} \times 3) + (\text{Tags MovieLens} \times 2) + \text{Tagline} + \text{Sinopsis / Overview}$$

Pembobotan bobot kata $t$ pada film $d$ dihitung dengan rumus:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Di mana:
- $\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$ (Frekuensi kemunculan term)
- $\text{IDF}(t, D) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$ (Penurunan bobot kata umum & penguatan kata spesifik)
- Normalisasi Vektor Euclidean ($L_2\text{-norm}$): $\mathbf{v} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$

Menghasilkan matriks sparse berukuran **$9.742 \text{ film} \times 12.000 \text{ fitur}$**.

---

### B. Kalkulasi Sudut Vektor (Cosine Similarity)
Kedekatan antara film anchor $\mathbf{A}$ dan film kandidat $\mathbf{B}$ dihitung berdasarkan **kosinus dari sudut $\theta$** pada ruang vektor 12.000 dimensi:

$$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2} = \sum_{i=1}^{n} A_i B_i$$

Derajat sudut vektor $\theta$:
$$\theta = \arccos(\cos(\theta)) \times \frac{180^\circ}{\pi}$$

- **$\theta \to 0^\circ \implies \cos(\theta) \to 1.0$**: Vektor sangat searah (film memiliki kesamaan tema, genre, dan konteks yang sangat presisi).
- **$\theta \to 90^\circ \implies \cos(\theta) \to 0.0$**: Vektor saling tegak lurus / orthogonal (tidak relevan).

---

### C. Metrik Evaluasi: Precision@K
Mengukur ketepatan sistem dalam menyajikan rekomendasi yang relevan pada $K$ film teratas:

$$\text{Precision@K} = \frac{\text{Jumlah Film Relevan di Top-}K}{K} \times 100\%$$

*Kriteria Relevansi:* Film rekomendasi dianggap relevan jika memiliki irisan genre ($\text{Genre}_A \cap \text{Genre}_B \neq \emptyset$) dengan film yang sedang ditonton.

---

## 2. Struktur Proyek

```
c:\3. TA BOSS\Project\
├── ml_artifacts/                 # Model & Matriks TF-IDF tersimpan (Joblib)
│   ├── tfidf_matrix.joblib       # Matriks sparse TF-IDF (9742 x 12000)
│   ├── tfidf_vectorizer.joblib   # Vectorizer Scikit-learn
│   ├── movie_to_idx.joblib       # Mapping MovieLens ID -> Matriks Index
│   ├── idx_to_movie.joblib       # Mapping Index -> MovieLens ID
│   └── movies_cleaned.pkl        # Dataframe film terintegrasi (Posters + Tags)
├── recommender/                  # Django App Recommender Core
│   ├── engine.py                 # Singleton ContentBasedRecommender (RAM Caching & Inferensi < 10ms)
│   ├── views.py                  # View Controllers & API Endpoints
│   ├── urls.py                   # Routing Aplikasi
│   └── ...
├── movieweb/                     # Konfigurasi Django Project (settings, urls)
├── templates/                    # Antarmuka Modern (Netflix-style UI)
│   ├── base.html                 # Layout utama, Live Search Autocomplete, Navbar
│   ├── home.html                 # Hero showcase, filter genre, katalog film MovieLens
│   ├── movie_detail.html         # Detail film + Rekomendasi Real-Time + Badge Sudut Vektor
│   └── methodology.html          # Penjelasan komprehensif rumus matematika & teori
├── build_pipeline.py             # Skrip pembuatan artefak ML & preprocessing
├── test_app.py                   # Unit test verifikasi endpoint & akurasi
└── manage.py                     # CLI Django
```

---

## 3. Cara Menjalankan Aplikasi Web

Buka terminal PowerShell / Command Prompt di folder `c:\3. TA BOSS\Project`, lalu jalankan:

```bash
python manage.py runserver
```

Akses peramban (browser) di alamat:
👉 **http://127.0.0.1:8000/**

### Fitur yang Dapat Dicoba:
1. **Pencarian Real-Time (Live Search)**: Ketikkan judul film seperti `"Toy Story"`, `"Harry Potter"`, atau `"Inception"` di bilah pencarian untuk melihat dropdown autocomplete instan.
2. **Katalog & Filter Genre**: Eksplorasi film berdasarkan genre (*Action*, *Adventure*, *Animation*, *Comedy*, *Drama*, dll).
3. **Item-to-Item Real-Time Recommendation**: Klik tombol **"Tonton Film Ini"** pada salah satu film, sistem langsung:
   - Menghitung cosine similarity dan derajat sudut $\theta$ secara realtime (< 10 ms).
   - Menampilkan Top-K rekomendasi (pilihan $K=5, 10, 15, 20$).
   - Menampilkan skor **Precision@K** (misal: 100%).
   - Menampilkan badge sudut vektor: misal `θ = 28.5°` dan persentase kemiripan `88.5%`.
4. **Halaman Metodologi & Teori**: Kunjungi menu **"Metodologi & Teori"** untuk melihat penjelasan matematis lengkap untuk bahan sidang skripsi / tugas akhir.
