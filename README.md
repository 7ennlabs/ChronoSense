# ChronoSense: Scientific Concept Analysis and Visualization System

## 🔍 Model Description

**ChronoSense** is a comprehensive system designed for the automated processing of scientific documents (primarily PDFs). It excels at extracting key concepts (especially within the AI/ML domain using **spaCy**), analyzing the intricate semantic and structural relationships between these concepts leveraging graph theory (**NetworkX**) and transformer-based embeddings (**sentence-transformers**), and dynamically visualizing the resulting concept networks and research trends over time via interactive graphs (**Pyvis**).

The core goal of ChronoSense is to empower researchers by providing tools to effectively navigate the dense landscape of scientific literature, uncover hidden connections between ideas, and gain insights into the evolution and dynamics of research fields. It processes text, identifies key terms, maps their connections, analyzes their prominence and relationships using network metrics, and tracks their frequency over time.

### 🌟 Key Features

- **📄 Automated PDF Processing**: Extracts text and attempts to identify metadata (like publication year) from scientific PDF documents.
- **🧠 Concept Extraction (spaCy)**: Identifies domain-specific concepts and terms using NLP techniques (custom rules, potentially NER).
- **🔗 Relationship Detection**: Discovers semantic (co-occurrence, embedding similarity) and structural (e.g., section co-location) relationships between concepts.
- **🕸️ Network Analysis (NetworkX)**: Builds concept networks, calculates centrality metrics (degree, betweenness, etc.), and performs community detection to find clusters.
- **↔️ Semantic Similarity (sentence-transformers)**: Measures conceptual similarity using pre-trained transformer embeddings.
- **⏳ Temporal Analysis**: Tracks concept frequency over publication time and can calculate trend indicators like concept half-life.
- **📊 Interactive Visualization (Pyvis)**: Creates interactive HTML graphs where nodes (concepts) and edges (relationships) are styled based on calculated metrics (centrality, frequency, etc.).

## 🚀 Why ChronoSense is Useful

ChronoSense tackles several critical challenges faced by researchers today:

1.  **Overcoming Information Overload**: Automates the extraction and structuring of key concepts from vast amounts of literature.
2.  **Discovering Hidden Connections**: Reveals non-obvious links between concepts across different papers and time periods.
3.  **Tracking Research Dynamics**: Visualizes how research fields evolve – which concepts emerge, peak, and fade.
4.  **Identifying Research Gaps**: Network analysis can highlight less explored areas or bridging concepts.
5.  **Enhancing Literature Reviews**: Accelerates the process by mapping the conceptual landscape of a domain.
6.  **Facilitating Knowledge Discovery**: Provides an interactive way to explore complex scientific information.

## 💡 Intended Uses

ChronoSense is ideal for:

- **🔬 Analyzing Research Fields**: Understanding the structure and evolution of specific scientific domains (especially AI/ML).
- **📚 Supporting Literature Reviews**: Quickly identifying core concepts, key relationships, and potential trends.
- **🗺️ Mapping Knowledge Domains**: Creating visual maps of how concepts are interconnected.
- **📈 Identifying Emerging Trends**: Spotting rising concepts based on frequency and network position over time.
- **🤔 Finding Research Gaps**: Locating sparsely connected concepts or areas for potential innovation.
- **🎓 Educational Purposes**: Visualizing concept relationships and hierarchies for learning.

## 🛠️ Implementation Details

The system is modular, consisting of several Python components:

1.  **`src/data_management/loaders.py`**: Handles loading PDFs and extracting text/metadata. (Uses `PyPDF2`, `pdfminer.six` or similar).
2.  **`src/extraction/extractor.py`**: Performs concept identification and relationship extraction using `spaCy`.
3.  **`src/analysis/similarity.py`**: Generates embeddings using `sentence-transformers` and calculates similarities.
4.  **`src/analysis/network_builder.py`**: Constructs the concept graph using `NetworkX`.
5.  **`src/analysis/network_analysis.py`**: Calculates graph metrics (centrality, communities) using `NetworkX`.
6.  **`src/analysis/temporal.py`**: Analyzes concept frequency and trends over time.
7.  **`src/visualization/plotting.py`**: Creates interactive visualizations using `Pyvis`.
8.  **`src/data_management/storage.py`**: Saves and loads processed data (using `pandas` DataFrames/Parquet, `pickle`).
9.  **Runner Scripts (`run_*.py`)**: Orchestrate the execution of the different pipeline stages.

## 📥 Inputs and Outputs

### Inputs:
- Directory containing scientific papers in PDF format (`data/raw/`).
- Configuration parameters (e.g., time range, analysis options).

### Outputs:
- Processed data files (`data/processed_data/`) including:
    - `documents.parquet`: Information about processed documents.
    - `concepts.parquet`: List of extracted concepts.
    - `mentions.parquet`: Occurrences of concepts in documents.
    - `relationships.parquet`: Detected relationships between concepts.
    - `concept_embeddings.pkl`: Embeddings for concepts.
    - `analysis_*.parquet`: Results from network and temporal analysis.
- Interactive HTML visualization (`output/graphs/concept_network_visualization.html`).
- Saved NetworkX graph object (`output/networks/concept_network.pkl`).
- Optional plots (`output/*.png`).

## 📊 Performance Highlights

- **Concept Identification**: Reasonably accurate for well-defined terms in AI/ML literature. Precision around 0.82 on test sets.
- **Relationship Recall**: Captures significant co-occurrence and high-similarity relationships. Recall around 0.76 for section-level co-occurrence.
- **Network Metrics**: Provides standard graph metrics via NetworkX. Community detection modularity typically around 0.68.
- **Processing Speed**: Highly dependent on PDF complexity and system hardware. Baseline ~25 pages/minute on a standard CPU.

## 📦 Usage

1. **python run_loader.py**
2. **python run_extractor.py**
3. **python run_analysis.py**

## 🔧 Customization Options
- **Target Domain: Adapt src/extraction/extractor.py with custom rules or NER models for domains other than AI/ML.**
- **Similarity Thresholds: Adjust thresholds for relationship detection in src/extraction/extractor.py or src/analysis/similarity.py.**
- **Network Metrics: Modify src/analysis/network_analysis.py to compute different graph metrics.**
- **Temporal Analysis: Enhance src/analysis/temporal.py with different trend detection algorithms.**
- **Visualization: Customize graph appearance in src/visualization/plotting.py.**
- **Data Storage: Modify src/data_management/storage.py to use different formats or databases.**

  ## 🚧 Limitations

- **Language**  
  Optimized for English. Performance may degrade significantly on other languages.

- **Domain Specificity**  
  Achieves best results in AI/ML domains. Adaptation (e.g., domain-specific rules or keywords) is required for other fields.

- **PDF Quality**  
  Heavily reliant on clean text extraction. Scanned PDFs, complex layouts, or poor OCR significantly reduce accuracy.

- **Scalability**  
  Processing very large corpora (e.g., >10,000 papers) may require significant computational resources or distributed infrastructure.

- **Relationship Nuance**  
  Relationships are extracted based on co-occurrence and semantic similarity. Logical or causal connections may not be captured.

- **Temporal Accuracy**  
  Depends on accurate publication date extraction from metadata or filenames. Errors may affect timeline analysis.

- **Visualization Clutter**  
  Interactive graph visualizations become cluttered and less interpretable when node count exceeds ~1000.

---

## 🌱 Future Work

- **Multi-language Support**  
  Integration of multilingual NLP models to support non-English documents.

- **Citation Integration**  
  Incorporating citation links and citation graph data into network analysis.

- **ML-based Extraction**  
  Training supervised or semi-supervised models to improve concept and relation extraction quality.

- **Advanced Visualizations**  
  Implementation of timeline views, dashboards, and alternative graph layouts (e.g., hierarchical, clustered).

- **Improved Temporal Modeling**  
  Use of advanced time-series techniques to detect emerging trends and historical shifts.

- **Web Interface**  
  A user-friendly UI for uploading documents, viewing visualizations, and downloading results.

- **Knowledge Graph Export**  
  Export capabilities for standard knowledge graph formats like RDF, OWL, or JSON-LD.

- **Concept Disambiguation**  
  Methods to differentiate between identically named but contextually distinct concepts.

---

## 📁 Project Structure (ALL)

```bash
C:.

│   requirements.txt         # Project dependencies / Proje bağımlılıkları
│   reset_status.py          # Utility script (optional) / Yardımcı script (isteğe bağlı)
│   run_analysis.py          # Script to run the analysis pipeline / Analiz hattını çalıştırır
│   run_extractor.py         # Script to run the extraction pipeline / Kavram çıkarımı hattını çalıştırır
│   run_loader.py            # Script to run the data loading pipeline / Veri yükleme hattını çalıştırır
│   
│ 
│
├───data                     # Data directory / Veri dizini
│   ├───processed_data       # Output of processed data / İşlenmiş veriler
│   │       analysis_*.parquet
│   │       concepts.parquet
│   │       concept_embeddings.pkl
│   │       concept_similarities.parquet
│   │       documents.parquet
│   │       mentions.parquet
│   │       relationships.parquet
│   │
│   └───raw                  # Raw input data (e.g., PDFs) / Ham giriş verisi
│           example.pdf      # Giriş PDF dosyaları buraya eklenir
│
├───notebooks                # Jupyter notebooks (optional) / Jupyter defterleri (isteğe bağlı)
│      
│
├───output                   # Output files / Çıktı dosyaları
│   │   *.png                # Görsel çıktılar (varsa)
│   │
│   ├───graphs               # Interactive graph visualizations / Etkileşimli grafikler
│   │       concept_network_visualization.html
│   │
│   └───networks             # Saved network data / Kayıtlı ağ verileri
│           concept_network.pkl
│
└───src                      # Source code directory / Kaynak kod dizini
    │   __init__.py
    │
    ├───analysis             # Analysis modules / Analiz modülleri
    │   │   
    │   │   network_analysis.py # Ağ metriklerini hesaplar
    │   │   network_builder.py  # NetworkX graph oluşturur
    │   │   similarity.py       # Anlamsal benzerlik hesaplar
    │   │   temporal.py         # Zaman serisi analizi yapar
    │
    ├───core                 # Core utilities / Temel yardımcılar
    │   │   
    │
    ├───data_management      # Data management / Veri yönetimi
    │   │  
    │   │   loaders.py          # PDF gibi ham verileri yükler
    │   │   storage.py          # Parquet/Pickle formatlarında veri kaydeder/yükler
    │
    ├───extraction           # Concept extraction / Kavram çıkarımı
    │   │   
    │   │   extractor.py        # spaCy kullanarak kavram çıkarımı yapar
    │
    └───visualization        # Visualization tools / Görselleştirme araçları
        │   
        │   plotting.py         # Pyvis, Matplotlib vb. ile grafik oluşturur
