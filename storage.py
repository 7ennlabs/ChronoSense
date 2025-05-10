# src/data_management/storage.py (TÜM SABİTLERİ İÇEREN DOĞRU TAM KOD)

import pandas as pd
from pathlib import Path
import logging
import uuid
from datetime import datetime
import networkx as nx
import pickle
import string

# Temel veri klasörünün yolu
DATA_PATH = Path("data/processed_data")
# NetworkX graf dosyalarının yolu
NETWORK_PATH = Path("output/networks")

# --- TÜM GEREKLİ SABİT TANIMLARI ---
FREQUENCY_FILENAME = "analysis_concept_frequencies"
SIMILARITY_FILENAME = "analysis_concept_similarities"
NETWORK_ANALYSIS_FILENAME = "analysis_network_results"
GRAPH_FILENAME = "concept_network"
EMBEDDINGS_FILENAME = "concept_embeddings"
# ------------------------------------

# DataFrame sütun isimleri
DOC_COLUMNS = ['doc_id', 'filepath', 'publication_date', 'status', 'processed_text_path']
CONCEPT_COLUMNS = ['concept_id', 'name', 'aliases']
MENTION_COLUMNS = ['mention_id', 'doc_id', 'concept_id', 'context_snippet', 'start_char', 'end_char']
RELATIONSHIP_COLUMNS = ['relationship_id', 'source_concept_id', 'target_concept_id', 'type', 'mention_id', 'doc_id', 'sentence']
NETWORK_ANALYSIS_COLUMNS = ['concept_id', 'name', 'degree_centrality', 'betweenness_centrality', 'eigenvector_centrality', 'community_id']

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DataFrame Yükleme/Kaydetme (Değişiklik yok) ---
def load_dataframe(filename: str, columns: list) -> pd.DataFrame:
    filepath = DATA_PATH / f"{filename}.parquet"
    if filepath.exists():
        try:
            df = pd.read_parquet(filepath)
            logging.info(f"'{filepath}' başarıyla yüklendi.")
            if columns: # Check columns only if a list is provided
                for col in columns:
                    if col not in df.columns:
                        logging.warning(f"'{filepath}' dosyasında '{col}' sütunu eksik. Ekleniyor...")
                        df[col] = None
            return df
        except Exception as e:
            logging.error(f"'{filepath}' yüklenirken hata oluştu: {e}")
            return pd.DataFrame(columns=columns if columns else None)
    else:
        logging.info(f"'{filepath}' bulunamadı. Boş DataFrame oluşturuluyor.")
        return pd.DataFrame(columns=columns if columns else None)

def save_dataframe(df: pd.DataFrame, filename: str):
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    filepath = DATA_PATH / f"{filename}.parquet"
    try:
        for col in df.select_dtypes(include=['object']).columns:
             if df[col].map(type).isin([list, dict, datetime, pd.Timestamp]).any(): continue
             df[col] = df[col].where(pd.notnull(df[col]), None)
             try: df[col] = df[col].astype(pd.StringDtype())
             except TypeError: logging.debug(f"Sütun '{col}' StringDtype'a çevrilemedi, orijinal tip korunuyor.")
        df.to_parquet(filepath, index=False)
        logging.info(f"DataFrame başarıyla '{filepath}' olarak kaydedildi.")
    except Exception as e:
        logging.error(f"DataFrame '{filepath}' olarak kaydedilirken hata oluştu: {e}")

# --- Doküman Yönetimi (Değişiklik yok) ---
def add_document(filepath_str: str, publication_date) -> str | None:
    documents_df = load_dataframe('documents', DOC_COLUMNS)
    filepath_str = str(Path(filepath_str).resolve())
    existing_doc = documents_df[documents_df['filepath'] == filepath_str]
    if not existing_doc.empty:
        existing_doc_id = existing_doc['doc_id'].iloc[0]
        logging.warning(f"Doküman zaten kayıtlı: {filepath_str} (ID: {existing_doc_id})")
        return str(existing_doc_id)
    new_doc_id = str(uuid.uuid4())
    try: pub_date_obj = pd.to_datetime(publication_date).date()
    except ValueError: logging.error(f"Geçersiz tarih formatı: {publication_date}. None olarak kaydedilecek."); pub_date_obj = None
    new_document_data = {'doc_id': new_doc_id, 'filepath': filepath_str, 'publication_date': pub_date_obj, 'status': 'added', 'processed_text_path': None}
    new_row_df = pd.DataFrame([new_document_data])
    if pub_date_obj is not None: new_row_df['publication_date'] = pd.to_datetime(new_row_df['publication_date']); dtype_dict = {'publication_date': 'datetime64[s]'}
    else: dtype_dict = {}
    documents_df = pd.concat([documents_df, new_row_df], ignore_index=True)
    for col, dtype in dtype_dict.items():
        try: documents_df[col] = documents_df[col].astype(dtype)
        except TypeError: logging.warning(f"Sütun '{col}' tipi '{dtype}' olarak ayarlanamadı.")
    save_dataframe(documents_df, 'documents')
    logging.info(f"Yeni doküman eklendi: {filepath_str} (ID: {new_doc_id})")
    return new_doc_id

def update_document_status(doc_id: str, new_status: str, text_path: str | None = None):
    docs_df = load_dataframe('documents', DOC_COLUMNS)
    doc_index = docs_df[docs_df['doc_id'] == doc_id].index
    if not doc_index.empty:
        idx = doc_index[0]
        docs_df.loc[idx, 'status'] = new_status
        if text_path: docs_df.loc[idx, 'processed_text_path'] = text_path
        save_dataframe(docs_df, 'documents')
        logging.info(f"Doküman durumu güncellendi: ID {doc_id} -> {new_status}")
    else: logging.warning(f"Durumu güncellenecek doküman bulunamadı: ID {doc_id}")

# --- Konsept, Mention, İlişki Yönetimi (Değişiklik yok) ---
def add_concept(raw_name: str) -> str | None:
    concepts_df = load_dataframe('concepts', CONCEPT_COLUMNS)
    name = raw_name.lower().strip().strip(string.punctuation + string.whitespace)
    if name.endswith("'s"): name = name[:-2].strip()
    name = ' '.join(name.split())
    if not name or len(name) < 2: return None
    existing_concept = concepts_df[concepts_df['name'] == name]
    if not existing_concept.empty: return str(existing_concept['concept_id'].iloc[0])
    new_concept_id = str(uuid.uuid4()); new_concept_data = {'concept_id': new_concept_id, 'name': name, 'aliases': [raw_name]}
    new_row_df = pd.DataFrame([new_concept_data]); concepts_df = pd.concat([concepts_df, new_row_df], ignore_index=True)
    concepts_df['aliases'] = concepts_df['aliases'].astype('object')
    save_dataframe(concepts_df, 'concepts')
    logging.info(f"Yeni konsept eklendi: '{name}' (Orijinal: '{raw_name}', ID: {new_concept_id})")
    return new_concept_id

def add_mention(doc_id: str, concept_id: str, context: str, start: int, end: int) -> str | None:
    if concept_id is None: return None
    mentions_df = load_dataframe('mentions', MENTION_COLUMNS); new_mention_id = str(uuid.uuid4())
    new_mention_data = {'mention_id': new_mention_id, 'doc_id': doc_id, 'concept_id': concept_id, 'context_snippet': context[:500], 'start_char': start, 'end_char': end}
    new_row_df = pd.DataFrame([new_mention_data]); mentions_df = pd.concat([mentions_df, new_row_df], ignore_index=True)
    save_dataframe(mentions_df, 'mentions'); return new_mention_id

def add_relationship(source_concept_id: str, target_concept_id: str, rel_type: str, mention_id: str | None, doc_id: str, sentence: str) -> str | None:
    if source_concept_id is None or target_concept_id is None: return None
    relationships_df = load_dataframe('relationships', RELATIONSHIP_COLUMNS); new_relationship_id = str(uuid.uuid4())
    new_relationship_data = {'relationship_id': new_relationship_id, 'source_concept_id': source_concept_id, 'target_concept_id': target_concept_id, 'type': rel_type, 'mention_id': mention_id, 'doc_id': doc_id, 'sentence': sentence[:500]}
    new_row_df = pd.DataFrame([new_relationship_data]); relationships_df = pd.concat([relationships_df, new_row_df], ignore_index=True)
    save_dataframe(relationships_df, 'relationships'); return new_relationship_id

# --- NetworkX Graf Yükleme/Kaydetme (Değişiklik yok) ---
def save_network(graph: nx.Graph, filename: str):
    NETWORK_PATH.mkdir(parents=True, exist_ok=True); filepath = NETWORK_PATH / f"{filename}.pkl"
    try:
        with open(filepath, 'wb') as f: pickle.dump(graph, f)
        logging.info(f"NetworkX grafı başarıyla '{filepath}' olarak kaydedildi.")
    except Exception as e: logging.error(f"Graf '{filepath}' olarak kaydedilirken hata: {e}")

def load_network(filename: str) -> nx.Graph | None:
    filepath = NETWORK_PATH / f"{filename}.pkl"
    if filepath.exists():
        try:
            with open(filepath, 'rb') as f: graph = pickle.load(f)
            logging.info(f"NetworkX grafı '{filepath}' başarıyla yüklendi.")
            return graph
        except Exception as e: logging.error(f"Graf '{filepath}' yüklenirken hata: {e}"); return nx.Graph()
    else: logging.warning(f"Graf dosyası bulunamadı: '{filepath}'"); return nx.Graph()