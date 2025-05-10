import time
# src klasöründeki modüllerimize erişmek için
from src.extraction.extractor import process_documents_for_extraction

if __name__ == "__main__":
    print(">>> Bilgi çıkarıcı çalıştırılıyor...")
    print("Not: Bu işlem dokümanların uzunluğuna ve sayısına göre biraz zaman alabilir.")
    start_time = time.time()

    # Ana çıkarım fonksiyonumuzu çağırıyoruz
    process_documents_for_extraction()

    end_time = time.time()
    print(f"<<< Bilgi çıkarıcı tamamlandı. Süre: {end_time - start_time:.2f} saniye.")
    print(f"Kontrol edilmesi gereken dosyalar: data/processed_data/ klasöründeki concepts.parquet, mentions.parquet, relationships.parquet ve güncellenmiş documents.parquet")