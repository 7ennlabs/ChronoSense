import time
# src klasöründeki modüllerimize erişmek için
from src.data_management.loaders import process_raw_documents

if __name__ == "__main__":
    print(">>> Veri yükleyici çalıştırılıyor...")
    start_time = time.time()

    # Ana işlem fonksiyonumuzu çağırıyoruz
    process_raw_documents()

    end_time = time.time()
    print(f"<<< Veri yükleyici tamamlandı. Süre: {end_time - start_time:.2f} saniye.")
    print(f"Kontrol edilmesi gereken dosya: data/processed_data/documents.parquet")