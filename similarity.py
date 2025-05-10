# src/analysis/similarity.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path

# Yerel modüller
from src.data_management import storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Benzerlik matrisini kaydetmek için dosya adı
SIMILARITY_FILENAME = "concept_similarities"
EMBEDDINGS_FILENAME = "concept_embeddings" # Vektörleri de kaydedebiliriz

def calculate_concept_embeddings(model_name: str = 'all-MiniLM-L6-v2', force_recalculate: bool = False) -> dict[str, np.ndarray] | None:
    """
    Her konsept için ortalama embedding vektörünü hesaplar.
    Mention'ların context_snippet'lerini kullanır.
    Hesaplanmış embedding'leri yüklemeye çalışır, yoksa hesaplar.

    Args:
        model_name (str): Kullanılacak Sentence Transformer modeli.
        force_recalculate (bool): Daha önce hesaplanmış olsa bile yeniden hesaplamaya zorla.

    Returns:
        dict[str, np.ndarray] | None: Concept ID -> Ortalama Embedding Vektörü sözlüğü veya hata durumunda None.
    """
    embeddings_filepath = storage.DATA_PATH / f"{EMBEDDINGS_FILENAME}.pkl" # Pickle ile saklayalım

    if not force_recalculate and embeddings_filepath.exists():
        try:
            embeddings = pd.read_pickle(embeddings_filepath)
            logging.info(f"Önceden hesaplanmış embedding'ler '{embeddings_filepath}' dosyasından yüklendi.")
            # Dosyadan yüklenen bir sözlük olmalı
            if isinstance(embeddings, dict):
                 return embeddings
            else:
                 logging.warning("Yüklenen embedding dosyası beklenen formatta (dict) değil. Yeniden hesaplanacak.")
        except Exception as e:
            logging.error(f"Embedding'ler yüklenirken hata: {e}. Yeniden hesaplanacak.")

    logging.info("Konsept embedding'leri hesaplanıyor...")
    mentions_df = storage.load_dataframe('mentions', storage.MENTION_COLUMNS)

    if mentions_df is None or mentions_df.empty:
        logging.warning("Hesaplama için mention verisi bulunamadı.")
        return None

    # Geçerli context snippet'i olan mention'ları al
    mentions_df.dropna(subset=['context_snippet', 'concept_id'], inplace=True)
    if mentions_df.empty:
        logging.warning("Geçerli context snippet bulunamadı.")
        return None

    # Modeli yükle (ilk seferde internetten indirilebilir)
    try:
        model = SentenceTransformer(model_name)
        logging.info(f"Sentence Transformer modeli '{model_name}' yüklendi.")
    except Exception as e:
        logging.exception(f"Sentence Transformer modeli '{model_name}' yüklenirken hata: {e}")
        return None

    # Konseptlere göre grupla
    grouped_mentions = mentions_df.groupby('concept_id')['context_snippet'].apply(list)

    concept_embeddings = {}
    logging.info(f"{len(grouped_mentions)} konsept için embedding hesaplanacak...")

    # Her konsept için embedding'leri hesapla ve ortalamasını al
    for concept_id, snippets in grouped_mentions.items():
        if not snippets: continue # Boş snippet listesi varsa atla
        try:
            # Tüm snippet'ların embedding'lerini tek seferde hesapla (daha verimli)
            embeddings = model.encode(snippets, show_progress_bar=False) # İlerleme çubuğunu kapat
            # Ortalama embedding'i hesapla
            avg_embedding = np.mean(embeddings, axis=0)
            concept_embeddings[concept_id] = avg_embedding
        except Exception as e:
            logging.error(f"Concept ID {concept_id} için embedding hesaplanırken hata: {e}")
            continue # Bu konsepti atla

    # Hesaplanan embedding'leri kaydet
    try:
        storage.DATA_PATH.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(concept_embeddings, embeddings_filepath)
        logging.info(f"Hesaplanan embedding'ler '{embeddings_filepath}' dosyasına kaydedildi.")
    except Exception as e:
        logging.error(f"Embedding'ler kaydedilirken hata: {e}")


    logging.info(f"{len(concept_embeddings)} konsept için ortalama embedding hesaplandı.")
    return concept_embeddings


def calculate_similarity_matrix(concept_embeddings: dict, force_recalculate: bool = False) -> pd.DataFrame | None:
    """
    Verilen embedding vektörleri arasındaki kosinüs benzerliğini hesaplar.
    Hesaplanmış benzerlikleri yüklemeye çalışır, yoksa hesaplar.

    Args:
        concept_embeddings (dict[str, np.ndarray]): Concept ID -> Embedding Vektörü sözlüğü.
        force_recalculate (bool): Daha önce hesaplanmış olsa bile yeniden hesaplamaya zorla.

    Returns:
        pd.DataFrame | None: 'concept_id_1', 'concept_id_2', 'similarity' sütunlarını
                             içeren DataFrame veya hata durumunda None.
    """
    similarity_filepath = storage.DATA_PATH / f"{SIMILARITY_FILENAME}.parquet"

    if not force_recalculate and similarity_filepath.exists():
        try:
            similarity_df = storage.load_dataframe(SIMILARITY_FILENAME, ['concept_id_1', 'concept_id_2', 'similarity'])
            logging.info(f"Önceden hesaplanmış benzerlik matrisi '{similarity_filepath}' dosyasından yüklendi.")
            if similarity_df is not None and not similarity_df.empty:
                return similarity_df
            else:
                 logging.warning("Yüklenen benzerlik dosyası boş veya hatalı. Yeniden hesaplanacak.")
        except Exception as e:
            logging.error(f"Benzerlik matrisi yüklenirken hata: {e}. Yeniden hesaplanacak.")


    if not concept_embeddings:
        logging.error("Benzerlik hesaplamak için embedding verisi bulunamadı.")
        return None

    logging.info("Konseptler arası benzerlik matrisi hesaplanıyor...")

    # Sözlükten sıralı liste ve matris oluştur
    concept_ids = list(concept_embeddings.keys())
    embedding_matrix = np.array(list(concept_embeddings.values()))

    # Boyut kontrolü
    if embedding_matrix.ndim != 2 or embedding_matrix.shape[0] != len(concept_ids):
        logging.error(f"Embedding matrisinin boyutları ({embedding_matrix.shape}) beklenenden farklı.")
        return None

    # Kosinüs benzerliğini hesapla
    try:
        similarity_matrix = cosine_similarity(embedding_matrix)
    except Exception as e:
        logging.exception(f"Kosinüs benzerliği hesaplanırken hata: {e}")
        return None

    # Matrisi DataFrame'e dönüştür (uzun format)
    similarity_data = []
    num_concepts = len(concept_ids)
    for i in range(num_concepts):
        for j in range(i + 1, num_concepts): # Sadece üçgenin üstünü al (j > i) ve kendini (i=j) atla
            similarity_data.append({
                'concept_id_1': concept_ids[i],
                'concept_id_2': concept_ids[j],
                'similarity': similarity_matrix[i, j]
            })

    similarity_df = pd.DataFrame(similarity_data)

    if similarity_df.empty:
        logging.warning("Hesaplama sonucu benzerlik verisi üretilemedi.")
        # Boş DataFrame kaydetmeyelim, None döndürelim
        return None

    # Hesaplanan benzerlikleri kaydet
    storage.save_dataframe(similarity_df, SIMILARITY_FILENAME)

    logging.info(f"Benzerlik matrisi hesaplandı ve kaydedildi. {len(similarity_df)} çift.")
    return similarity_df