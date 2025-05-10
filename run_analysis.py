# run_analysis.py (Ağ Analizi Metrikleri Eklendi)
import time
import pandas as pd
import sys
from pathlib import Path
import networkx as nx
import webbrowser
import logging

# src klasöründeki modüllere erişim için
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.temporal import calculate_concept_frequencies
from src.analysis.similarity import calculate_concept_embeddings, calculate_similarity_matrix
from src.analysis.network_builder import build_concept_network
# YENİ importlar:
from src.analysis.network_analysis import get_network_analysis_results, save_network_analysis
from src.visualization.plotting import visualize_network
from src.data_management.storage import load_dataframe, save_dataframe, CONCEPT_COLUMNS, FREQUENCY_FILENAME, SIMILARITY_FILENAME, NETWORK_ANALYSIS_FILENAME # YENİ: NETWORK_ANALYSIS_FILENAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

if __name__ == "__main__":
    print(">>> Analizler Çalıştırılıyor (Frekans + Benzerlik + Ağ + Metrikler + Görselleştirme) <<<")
    overall_start_time = time.time()
    concepts_df = None
    frequency_results_df = None
    similarity_results_df = None
    concept_network = None # Grafı saklamak için
    network_analysis_df = None # Analiz sonuçlarını saklamak için

    # --- 1. Frekans Analizi ---
    print("\n--- 1. Frekans Hesaplaması ---"); start_time = time.time()
    # ... (önceki kodla aynı, sadece print süresi değişebilir) ...
    frequency_df = calculate_concept_frequencies(time_period='YS')
    if frequency_df is not None:
        concepts_df = load_dataframe('concepts', CONCEPT_COLUMNS)
        if not frequency_df.empty:
             print(f"Toplam {len(frequency_df)} frekans kaydı hesaplandı.")
             if concepts_df is not None and not concepts_df.empty:
                 frequency_results_df = pd.merge(frequency_df, concepts_df[['concept_id', 'name']], on='concept_id', how='left')
                 frequency_results_df = frequency_results_df[['concept_id', 'name', 'time_period_start', 'frequency']]
                 frequency_results_df.sort_values(by=['name', 'time_period_start'], inplace=True)
                 print("\n--- Konsept Frekansları (Yıllık) ---"); print(frequency_results_df.to_string())
                 save_dataframe(frequency_results_df, FREQUENCY_FILENAME)
             else: print("\nKonsept isimleri yüklenemedi..."); print(frequency_df.to_string())
        else: print("Frekans hesaplandı ancak sonuç boş."); save_dataframe(pd.DataFrame(columns=['concept_id', 'name', 'time_period_start', 'frequency']), FREQUENCY_FILENAME)
    else: print("Frekans hesaplaması sırasında bir hata oluştu.")
    print(f"--- Frekans Hesaplaması Tamamlandı. Süre: {time.time() - start_time:.2f} saniye ---")

    # --- 2. Anlamsal Benzerlik Analizi ---
    print("\n--- 2. Anlamsal Benzerlik Hesaplaması ---"); start_time = time.time()
    # ... (önceki kodla aynı, sadece print süresi değişebilir) ...
    try:
        concept_embeddings = calculate_concept_embeddings(force_recalculate=False)
        if concept_embeddings:
            similarity_df = calculate_similarity_matrix(concept_embeddings, force_recalculate=False)
            if similarity_df is not None and not similarity_df.empty:
                print(f"Toplam {len(similarity_df)} konsept çifti için benzerlik hesaplandı/yüklendi.")
                if concepts_df is None or concepts_df.empty: concepts_df = load_dataframe('concepts', CONCEPT_COLUMNS)
                if concepts_df is not None and not concepts_df.empty:
                    sim_results = pd.merge(similarity_df, concepts_df[['concept_id', 'name']], left_on='concept_id_1', right_on='concept_id', how='left').rename(columns={'name': 'name_1'}).drop(columns=['concept_id'])
                    sim_results = pd.merge(sim_results, concepts_df[['concept_id', 'name']], left_on='concept_id_2', right_on='concept_id', how='left').rename(columns={'name': 'name_2'}).drop(columns=['concept_id'])
                    sim_results = sim_results[['concept_id_1', 'name_1', 'concept_id_2', 'name_2', 'similarity']]
                    sim_results.sort_values(by='similarity', ascending=False, inplace=True)
                    similarity_results_df = sim_results
                    print("\n--- En Benzer Konsept Çiftleri (Top 20) ---"); print(similarity_results_df.head(20).to_string(index=False))
                    save_dataframe(similarity_results_df, SIMILARITY_FILENAME)
                else: print("\nKonsept isimleri yüklenemedi..."); print(similarity_df.sort_values(by='similarity', ascending=False).head(20).to_string(index=False))
            elif similarity_df is not None: print("Benzerlik hesaplandı ancak sonuç boş."); save_dataframe(pd.DataFrame(columns=['concept_id_1', 'name_1', 'concept_id_2', 'name_2', 'similarity']), SIMILARITY_FILENAME)
    except Exception as e: logging.exception("Benzerlik hesaplama sırasında beklenmedik hata oluştu.")
    print(f"--- Benzerlik Hesaplaması Tamamlandı. Süre: {time.time() - start_time:.2f} saniye ---")

    # --- 3. Ağ Oluşturma ---
    print("\n--- 3. Konsept Ağı Oluşturma ---"); start_time = time.time()
    # GÜNCELLEME: Ağ nesnesini değişkende tut
    concept_network = build_concept_network(similarity_threshold=0.60)
    if concept_network is not None:
        print("\n--- Oluşturulan Ağ Bilgileri ---")
        print(f"Düğüm Sayısı (Konseptler): {concept_network.number_of_nodes()}")
        print(f"Kenar Sayısı (İlişkiler/Benzerlikler): {concept_network.number_of_edges()}")
        print(f"Ağ başarıyla oluşturuldu ve kaydedildi.")
    else:
        print("Konsept ağı oluşturulamadı.")
    print(f"--- Ağ Oluşturma Tamamlandı. Süre: {time.time() - start_time:.2f} saniye ---")


    # --- YENİ: 4. Ağ Analizi (Metrik Hesaplama) ---
    print("\n--- 4. Ağ Analizi Metrikleri ---"); start_time = time.time()
    if concept_network is not None and concept_network.number_of_nodes() > 0:
        network_analysis_df = get_network_analysis_results(concept_network)
        if network_analysis_df is not None and not network_analysis_df.empty:
             # Sonuçları kaydet
             save_network_analysis(network_analysis_df)
             print("Ağ metrikleri hesaplandı ve kaydedildi.")
             # En yüksek derece merkeziyetine sahip ilk 10 konsepti göster
             print("\n--- En Merkezi Konseptler (Degree Centrality Top 10) ---")
             print(network_analysis_df.sort_values(by='degree_centrality', ascending=False).head(10).to_string(index=False))
        else:
             print("Ağ metrikleri hesaplanamadı veya sonuç boş.")
    else:
        print("Ağ analizi yapmak için geçerli bir ağ bulunamadı.")
    print(f"--- Ağ Analizi Tamamlandı. Süre: {time.time() - start_time:.2f} saniye ---")


    # --- YENİ SIRA: 5. Ağ Görselleştirme ---
    print("\n--- 5. Ağ Görselleştirmesi Oluşturma ---"); start_time = time.time()
    visualization_path = None
    if concept_network is not None:
        # GÜNCELLEME: Analiz sonuçlarını da görselleştirmeye gönderebiliriz (ileride plotting.py'ı güncelleyince)
        # Şimdilik sadece grafı gönderiyoruz.
        visualization_path = visualize_network(graph=concept_network, output_filename="concept_network_visualization.html")
        if visualization_path:
            print(f"\nBaşarılı! İnteraktif ağ görselleştirmesi oluşturuldu:\n-> {visualization_path}")
            print("\nBu HTML dosyasını web tarayıcınızda açarak ağı inceleyebilirsiniz.")
        else: print("Ağ görselleştirmesi oluşturulurken bir sorun oluştu.")
    else: print("Ağ oluşturulamadığı için görselleştirme yapılamıyor.")
    print(f"--- Ağ Görselleştirme Tamamlandı. Süre: {time.time() - start_time:.2f} saniye ---")


    overall_end_time = time.time()
    print(f"\n<<< Tüm İşlemler Tamamlandı. Toplam Süre: {overall_end_time - overall_start_time:.2f} saniye >>>")