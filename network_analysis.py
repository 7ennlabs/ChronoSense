# src/analysis/network_analysis.py

import networkx as nx
import pandas as pd
import logging

# Topluluk tespiti için Louvain metodu (önce 'pip install python-louvain community' yapılmalı)
try:
    import community.community_louvain as community_louvain
    community_lib_available = True
except ImportError:
    logging.warning("'community' (python-louvain) kütüphanesi bulunamadı. Topluluk tespiti yapılamayacak. Kurulum için: pip install python-louvain community")
    community_lib_available = False

# Yerel modüller
from src.data_management import storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_centrality(graph: nx.Graph) -> dict:
    """
    Graf üzerindeki düğümler için merkeziyet metriklerini hesaplar.

    Args:
        graph (nx.Graph): Analiz edilecek NetworkX grafı.

    Returns:
        dict: {node_id: {'degree': float, 'betweenness': float, 'eigenvector': float (veya None)}}
              formatında metrikleri içeren sözlük.
    """
    metrics = {}
    if not graph or graph.number_of_nodes() == 0:
        return metrics

    try:
        degree_centrality = nx.degree_centrality(graph)
    except Exception as e:
        logging.error(f"Degree Centrality hesaplanırken hata: {e}")
        degree_centrality = {}

    try:
        betweenness_centrality = nx.betweenness_centrality(graph)
    except Exception as e:
        logging.error(f"Betweenness Centrality hesaplanırken hata: {e}")
        betweenness_centrality = {}

    try:
        # Eigenvector centrality bağlantısız (disconnected) graflarda veya bazı durumlarda hata verebilir
        # max_iter artırılabilir veya hata yakalanabilir
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=500, tol=1e-06)
    except Exception as e:
        logging.warning(f"Eigenvector Centrality hesaplanırken hata (graf bağlantısız olabilir): {e}")
        eigenvector_centrality = {} # Hata durumunda boş bırak

    # Metrikleri birleştir
    for node in graph.nodes():
        metrics[node] = {
            'degree_centrality': degree_centrality.get(node, 0.0),
            'betweenness_centrality': betweenness_centrality.get(node, 0.0),
            'eigenvector_centrality': eigenvector_centrality.get(node, None) # Hata durumunda None olabilir
        }
    logging.info("Merkeziyet metrikleri hesaplandı.")
    return metrics

def detect_communities(graph: nx.Graph) -> dict | None:
    """
    Louvain algoritması kullanarak graf üzerindeki toplulukları tespit eder.

    Args:
        graph (nx.Graph): Analiz edilecek NetworkX grafı.

    Returns:
        dict | None: {node_id: community_id} formatında bölümleme sözlüğü veya hata/kütüphane yoksa None.
    """
    if not community_lib_available:
        return None # Kütüphane yoksa hesaplama yapma
    if not graph or graph.number_of_nodes() == 0:
        return None # Boş graf

    # Louvain metodu yönlendirilmemiş graflarda daha iyi çalışır.
    # Eğer graf yönlü ise, yönlendirilmemişe çevir (veya uyarı ver).
    # Bizim grafımız zaten yönlendirilmemiş (nx.Graph).
    # Ağırlıklı kenarları kullanabilir (varsayılan weight='weight')
    try:
        partition = community_louvain.best_partition(graph, weight='weight') # Kenar ağırlıklarını dikkate al
        num_communities = len(set(partition.values()))
        logging.info(f"Louvain ile topluluk tespiti tamamlandı. {num_communities} topluluk bulundu.")
        return partition
    except Exception as e:
        logging.exception(f"Topluluk tespiti sırasında hata oluştu: {e}")
        return None


def get_network_analysis_results(graph: nx.Graph) -> pd.DataFrame | None:
    """
    Merkeziyet ve topluluk analizlerini yapar ve sonuçları bir DataFrame'de birleştirir.

    Args:
        graph (nx.Graph): Analiz edilecek NetworkX grafı.

    Returns:
        pd.DataFrame | None: 'concept_id', 'name', 'degree_centrality', 'betweenness_centrality',
                             'eigenvector_centrality', 'community_id' sütunlarını içeren DataFrame
                             veya hata durumunda None.
    """
    if not graph or graph.number_of_nodes() == 0:
        logging.warning("Analiz için boş veya geçersiz graf sağlandı.")
        return None

    logging.info("Ağ analizi metrikleri hesaplanıyor...")
    centrality_metrics = calculate_centrality(graph)
    community_partition = detect_communities(graph)

    # Sonuçları bir DataFrame'e dönüştür
    analysis_data = []
    concepts_df = storage.load_dataframe('concepts', storage.CONCEPT_COLUMNS) # İsimler için yükle

    for node_id, metrics in centrality_metrics.items():
        node_data = {
            'concept_id': node_id,
            'name': graph.nodes[node_id].get('name', 'N/A'), # Graf düğümünden al
            'degree_centrality': metrics.get('degree_centrality'),
            'betweenness_centrality': metrics.get('betweenness_centrality'),
            'eigenvector_centrality': metrics.get('eigenvector_centrality'),
            'community_id': community_partition.get(node_id, -1) if community_partition else -1 # Topluluk yoksa -1
        }
        analysis_data.append(node_data)

    if not analysis_data:
        logging.warning("Ağ analizi sonucu veri üretilemedi.")
        return None

    analysis_df = pd.DataFrame(analysis_data)

    # Eğer graf düğümlerinde isim yoksa, concepts_df'ten almayı dene (yedek)
    if 'N/A' in analysis_df['name'].values and concepts_df is not None:
         analysis_df = analysis_df.drop(columns=['name']) # Eski 'name' sütununu sil
         analysis_df = pd.merge(analysis_df, concepts_df[['concept_id', 'name']], on='concept_id', how='left')
         # Sütun sırasını ayarla
         cols = ['concept_id', 'name'] + [col for col in analysis_df.columns if col not in ['concept_id', 'name']]
         analysis_df = analysis_df[cols]


    logging.info("Ağ analizi sonuçları DataFrame'e dönüştürüldü.")
    return analysis_df


def save_network_analysis(analysis_df: pd.DataFrame):
    """ Ağ analizi sonuçlarını Parquet dosyasına kaydeder. """
    if analysis_df is not None and not analysis_df.empty:
        storage.save_dataframe(analysis_df, storage.NETWORK_ANALYSIS_FILENAME)
        logging.info(f"Ağ analizi sonuçları '{storage.NETWORK_ANALYSIS_FILENAME}.parquet' olarak kaydedildi.")
    else:
        logging.warning("Kaydedilecek ağ analizi sonucu bulunamadı.")