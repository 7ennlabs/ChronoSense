# src/analysis/network_builder.py (DÜZELTİLMİŞ TAM KOD)

import networkx as nx
import pandas as pd
import logging

# Yerel modüller
from src.data_management import storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Grafı kaydetmek için dosya adı
GRAPH_FILENAME = "concept_network"
# Benzerlik dosyasının adı (Doğrudan burada tanımlayalım veya similarity'den import edelim)
# storage modülünde değil!
SIMILARITY_FILENAME = "concept_similarities"

def build_concept_network(similarity_threshold: float = 0.60,
                            include_similarity_edges: bool = True,
                            include_extracted_edges: bool = True) -> nx.Graph | None:
    """
    Konseptler, çıkarılmış ilişkiler ve anlamsal benzerliklerden bir NetworkX grafı oluşturur.

    Args:
        similarity_threshold (float): Grafiğe eklenecek minimum anlamsal benzerlik skoru.
        include_similarity_edges (bool): Benzerlik kenarlarını dahil et.
        include_extracted_edges (bool): Metinden çıkarılan ilişki kenarlarını dahil et.

    Returns:
        nx.Graph | None: Oluşturulan NetworkX grafı veya hata durumunda None.
    """
    logging.info("Konsept ağı oluşturuluyor...")
    if not include_similarity_edges and not include_extracted_edges:
        logging.warning("Hem benzerlik hem de çıkarılmış ilişki kenarları devre dışı bırakıldı.")

    # Temel verileri yükle
    concepts_df = storage.load_dataframe('concepts', storage.CONCEPT_COLUMNS)
    relationships_df = storage.load_dataframe('relationships', storage.RELATIONSHIP_COLUMNS)
    # *** DÜZELTME: SIMILARITY_FILENAME doğrudan kullanılıyor ***
    similarity_df = storage.load_dataframe(SIMILARITY_FILENAME, ['concept_id_1', 'concept_id_2', 'similarity'])

    if concepts_df is None or concepts_df.empty:
        logging.error("Ağ oluşturmak için konsept verisi bulunamadı.")
        return None

    G = nx.Graph()

    # 1. Adım: Konseptleri Düğüm Olarak Ekle
    node_count = 0
    valid_concept_ids = set() # Grafiğe eklenen geçerli ID'leri takip et
    for index, row in concepts_df.iterrows():
        concept_id = row['concept_id']
        concept_name = row['name']
        if pd.notna(concept_id) and pd.notna(concept_name):
             G.add_node(concept_id, name=concept_name)
             valid_concept_ids.add(concept_id)
             node_count += 1
        else:
             logging.warning(f"Geçersiz konsept verisi atlandı: ID={concept_id}, Name={concept_name}")
    logging.info(f"{node_count} konsept düğüm olarak eklendi.")

    edge_count_extracted = 0
    edge_count_similarity = 0
    updated_edge_count = 0

    # 2. Adım: Çıkarılmış İlişkileri Kenar Olarak Ekle
    if include_extracted_edges and relationships_df is not None and not relationships_df.empty:
        logging.info("Çıkarılmış ilişkiler kenar olarak ekleniyor...")
        for index, row in relationships_df.iterrows():
            source_id = row['source_concept_id']
            target_id = row['target_concept_id']
            rel_type = row['type'] or 'RELATED_TO'

            # Düğümlerin grafide olduğundan ve geçerli olduğundan emin ol
            if source_id in valid_concept_ids and target_id in valid_concept_ids:
                if G.has_edge(source_id, target_id):
                     G.edges[source_id, target_id]['relation_type'] = rel_type
                     G.edges[source_id, target_id]['type'] = 'extracted'
                else:
                     G.add_edge(source_id, target_id, type='extracted', relation_type=rel_type, weight=0.8)
                     edge_count_extracted += 1
            else:
                 logging.warning(f"İlişki için düğüm(ler) bulunamadı veya geçersiz: {source_id} -> {target_id}")
        logging.info(f"{edge_count_extracted} çıkarılmış ilişki kenarı eklendi.")

    # 3. Adım: Anlamsal Benzerlikleri Kenar Olarak Ekle
    if include_similarity_edges and similarity_df is not None and not similarity_df.empty:
        logging.info(f"Anlamsal benzerlikler (Eşik > {similarity_threshold:.2f}) kenar olarak ekleniyor...")
        filtered_similarity = similarity_df[(similarity_df['similarity'] >= similarity_threshold) & (similarity_df['similarity'] < 1.0)]
        logging.info(f"{len(similarity_df)} benzerlik çiftinden {len(filtered_similarity)} tanesi eşik değerinin üzerinde (ve < 1.0).")

        for index, row in filtered_similarity.iterrows():
            id1 = row['concept_id_1']
            id2 = row['concept_id_2']
            similarity = row['similarity']

            if id1 in valid_concept_ids and id2 in valid_concept_ids:
                if G.has_edge(id1, id2):
                     G.edges[id1, id2]['similarity'] = similarity
                     if 'weight' not in G.edges[id1, id2] or similarity > G.edges[id1, id2].get('weight', 0):
                          G.edges[id1, id2]['weight'] = similarity
                     # Eğer extracted ilişki varsa, tipi 'combined' yapabiliriz?
                     G.edges[id1, id2]['type'] = 'combined' if G.edges[id1, id2].get('type') == 'extracted' else G.edges[id1, id2].get('type', 'similarity') # Önceliği koru veya birleştir
                     updated_edge_count += 1
                else:
                     G.add_edge(id1, id2, type='similarity', weight=similarity)
                     edge_count_similarity += 1
            else:
                logging.warning(f"Benzerlik için düğüm(ler) bulunamadı veya geçersiz: {id1} <-> {id2}")
        logging.info(f"{edge_count_similarity} yeni benzerlik kenarı eklendi, {updated_edge_count} mevcut kenara benzerlik/tip bilgisi eklendi.")

    total_edges = G.number_of_edges()
    logging.info(f"Konsept ağı oluşturuldu. Düğüm sayısı: {G.number_of_nodes()}, Kenar sayısı: {total_edges}.")

    # 4. Adım: Grafı Kaydet
    storage.save_network(G, GRAPH_FILENAME)

    return G