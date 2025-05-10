# src/visualization/plotting.py (Ağ Metrikleri ile Görselleştirme Güncellendi)

import networkx as nx
from pyvis.network import Network
import logging
from pathlib import Path
import pandas as pd
import random # Renk paleti için

# Yerel modüller
from src.data_management import storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Görselleştirme dosyalarının kaydedileceği yer
OUTPUT_DIR = Path("output/graphs")
DEFAULT_GRAPH_FILENAME = "concept_network"
# Analiz sonuçları dosyasının adı (storage'dan da alınabilirdi)
DEFAULT_ANALYSIS_FILENAME = storage.NETWORK_ANALYSIS_FILENAME


# Basit bir renk paleti (daha fazla renk eklenebilir veya matplotlib colormap kullanılabilir)
# Viridis, tab10, Set3 gibi paletler iyi çalışır
# Örnek: import matplotlib.cm as cm; colors = [cm.tab10(i) for i in range(10)]
DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

def get_color_for_community(community_id, colors=DEFAULT_COLORS):
    """ Verilen community ID için paletten bir renk döndürür. """
    if community_id < 0 or community_id is None or pd.isna(community_id): # Topluluk yoksa veya geçersizse
        return "#CCCCCC" # Gri
    return colors[int(community_id) % len(colors)] # Modulo ile renk tekrarı

def scale_value(value, min_val=0, max_val=1, new_min=10, new_max=50):
    """ Bir değeri belirli bir aralığa ölçekler (örn: merkeziyet -> düğüm boyutu). """
    if max_val == min_val or value is None or pd.isna(value): # Bölme hatasını veya None değerini engelle
        return new_min # Veya ortalama bir değer?
    # Ölçekleme: (value - min) / (max - min) * (new_max - new_min) + new_min
    scaled = ((value - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
    return max(new_min, min(scaled, new_max)) # Sonuçların min/max arasında kalmasını sağla


def visualize_network(graph: nx.Graph | None = None,
                      graph_filename: str = DEFAULT_GRAPH_FILENAME,
                      analysis_filename: str = DEFAULT_ANALYSIS_FILENAME,
                      output_filename: str = "concept_network_visualization.html",
                      show_buttons: bool = True,
                      physics_solver: str = 'barnesHut',
                      size_metric: str = 'degree_centrality', # Boyut için kullanılacak metrik
                      color_metric: str = 'community_id',    # Renk için kullanılacak metrik
                      height: str = "800px",
                      width: str = "100%"
                     ) -> str | None:
    """
    Ağ grafını Pyvis ile görselleştirir. Düğüm boyutu ve rengi için ağ
    analizi metriklerini kullanır.
    """
    if graph is None:
        logging.info(f"Graf sağlanmadı, '{graph_filename}.pkl' dosyasından yükleniyor...")
        graph = storage.load_network(graph_filename)

    if graph is None or not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        logging.error("Görselleştirilecek geçerli veya boş olmayan bir graf bulunamadı.")
        return None

    # Ağ analizi sonuçlarını yükle
    logging.info(f"Ağ analizi sonuçları '{analysis_filename}.parquet' dosyasından yükleniyor...")
    analysis_df = storage.load_dataframe(analysis_filename, []) # Sütunları bilmediğimiz için boş liste
    metrics_dict = {}
    min_size_val, max_size_val = 0, 1 # Boyut ölçekleme için min/max

    if analysis_df is not None and not analysis_df.empty and 'concept_id' in analysis_df.columns:
        # Eksik metrik sütunlarını kontrol et ve ekle (NaN ile)
        required_metrics = [size_metric, color_metric]
        for metric in required_metrics:
            if metric not in analysis_df.columns:
                 logging.warning(f"Analiz sonuçlarında '{metric}' sütunu bulunamadı. Varsayılan değerler kullanılacak.")
                 analysis_df[metric] = None

        # Boyut metriği için min/max değerleri bul (NaN olmayanlardan)
        if size_metric in analysis_df.columns and analysis_df[size_metric].notna().any():
            min_size_val = analysis_df[size_metric].min()
            max_size_val = analysis_df[size_metric].max()

        # Kolay erişim için sözlüğe çevir
        metrics_dict = analysis_df.set_index('concept_id').to_dict('index')
        logging.info("Ağ analizi metrikleri yüklendi.")
    else:
        logging.warning("Ağ analizi sonuçları yüklenemedi veya boş. Varsayılan düğüm boyutları/renkleri kullanılacak.")


    logging.info(f"'{output_filename}' için Pyvis ağı oluşturuluyor...")
    net = Network(notebook=False, height=height, width=width, heading='ChronoSense Konsept Ağı (Metriklerle)', cdn_resources='remote')
    net.barnes_hut(gravity=-8000, central_gravity=0.1, spring_length=150, spring_strength=0.005, damping=0.09)

    # Düğümleri (Nodes) Pyvis'e ekle (Boyut ve Renk ile)
    for node, attrs in graph.nodes(data=True):
        node_label = attrs.get('name', str(node))
        node_metrics = metrics_dict.get(node, {}) # Bu düğüm için metrikleri al, yoksa boş dict

        # Boyutu hesapla
        size_val = node_metrics.get(size_metric)
        node_size = scale_value(size_val, min_size_val, max_size_val, new_min=10, new_max=40) # 10-40 arası boyut

        # Rengi hesapla
        color_val = node_metrics.get(color_metric)
        node_color = get_color_for_community(color_val)

        # Başlığı (Title) güncelle (metrikleri ekle)
        node_title = f"ID: {node}<br>Name: {attrs.get('name', 'N/A')}"
        node_title += f"<br>{size_metric}: {size_val:.3f}" if pd.notna(size_val) else ""
        node_title += f"<br>{color_metric}: {int(color_val)}" if pd.notna(color_val) else ""

        net.add_node(node, label=node_label, title=node_title, size=node_size, color=node_color)

    # Kenarları (Edges) Pyvis'e ekle (Öncekiyle aynı, sadece renk/kalınlık ayarları biraz daha belirgin)
    for source, target, attrs in graph.edges(data=True):
        edge_title = f"Type: {attrs.get('type', 'N/A')}"
        edge_value = 0.5 ; edge_color = "#DDDDDD" # Daha soluk varsayılan

        edge_type = attrs.get('type')
        weight = attrs.get('weight', 0)

        if edge_type == 'extracted':
             edge_title += f"<br>Relation: {attrs.get('relation_type', 'N/A')}"
             edge_value = max(0.6, weight) # extracted ilişkiler biraz daha belirgin olsun
             edge_color = "#FF6347" # Koyu turuncu/kırmızımsı
        elif edge_type == 'similarity':
             sim_score = attrs.get('similarity', weight)
             edge_title += f"<br>Similarity: {sim_score:.3f}"
             edge_value = sim_score # Benzerlikle orantılı
             edge_color = "#4682B4" # Çelik mavisi
        elif edge_type == 'combined':
             edge_title += f"<br>Relation: {attrs.get('relation_type', 'N/A')}"
             sim_score = attrs.get('similarity', weight)
             edge_title += f"<br>Similarity: {sim_score:.3f}"
             edge_value = max(0.6, sim_score) # Combined da belirgin olsun
             edge_color = "#9370DB" # Orta mor

        net.add_edge(source, target, title=edge_title, value=max(0.1, edge_value), color=edge_color)

    if show_buttons:
        net.show_buttons(filter_=['physics', 'nodes', 'edges'])

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / output_filename
        net.save_graph(str(output_path))
        logging.info(f"Ağ görselleştirmesi başarıyla '{output_path}' olarak kaydedildi.")
        return str(output_path)
    except Exception as e:
        logging.exception(f"Ağ görselleştirmesi kaydedilirken hata oluştu: {e}")
        return None