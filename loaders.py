import PyPDF2 # PDF dosyalarını okumak için
from pathlib import Path
from datetime import datetime
import logging
import re # Tarih ayrıştırma için Regular Expressions

# Mevcut modüldeki storage fonksiyonlarını içe aktar (aynı klasörde olduğu için .)
from .storage import add_document, load_dataframe, save_dataframe, DOC_COLUMNS

# Ham veri klasörünün yolu
RAW_DATA_PATH = Path("data/raw")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path: Path) -> str | None:
    """
    Verilen PDF dosyasının metin içeriğini çıkarır.

    Args:
        pdf_path (Path): PDF dosyasının yolu.

    Returns:
        str | None: Çıkarılan metin veya hata durumunda None.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n" # Sayfalar arasına yeni satır ekle
            logging.info(f"Metin çıkarıldı: {pdf_path.name}")
            return text
    except Exception as e:
        logging.error(f"PDF metni çıkarılırken hata ({pdf_path.name}): {e}")
        # Şifreli PDF'ler veya bozuk dosyalar PyPDF2 tarafından hata verebilir
        if "password" in str(e).lower():
             logging.warning(f"Dosya şifreli olabilir: {pdf_path.name}")
        return None

def parse_date_from_filename(filename: str) -> datetime | None:
    """
    Dosya adından YYYY-MM-DD veya YYYYMMDD formatında tarih ayrıştırmaya çalışır.

    Args:
        filename (str): Dosya adı.

    Returns:
        datetime | None: Bulunan tarih veya None.
    """
    # Örnek: 2023-10-26_paper.pdf, 20231026-paper.pdf, 2023_10_26 paper.pdf
    patterns = [
        r"(\d{4}-\d{2}-\d{2})", # YYYY-MM-DD
        r"(\d{4}_\d{2}_\d{2})", # YYYY_MM_DD
        r"(\d{8})"             # YYYYMMDD
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            date_str = match.group(1).replace("_", "-") # Alt çizgiyi tireye çevir
            try:
                # Sadece tarih kısmını al, saat bilgisi ekleme
                return datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                continue # Geçersiz tarih formatı varsa diğer deseni dene
    logging.warning(f"Dosya adından geçerli tarih ayrıştırılamadı: {filename}")
    return None

def process_raw_documents():
    """
    'data/raw/' klasöründeki tüm PDF dosyalarını işler,
    tarihlerini ayrıştırır ve sisteme ekler (eğer zaten ekli değillerse).
    """
    if not RAW_DATA_PATH.exists():
        logging.error(f"Ham veri klasörü bulunamadı: {RAW_DATA_PATH}")
        return

    logging.info(f"'{RAW_DATA_PATH}' klasöründeki PDF dosyaları işleniyor...")
    processed_count = 0
    added_count = 0

    # Tüm PDF dosyalarını bul
    pdf_files = list(RAW_DATA_PATH.glob('*.pdf'))

    if not pdf_files:
        logging.warning(f"'{RAW_DATA_PATH}' klasöründe işlenecek PDF dosyası bulunamadı.")
        return

    for pdf_path in pdf_files:
        processed_count += 1
        filename = pdf_path.name
        filepath_str = str(pdf_path.resolve()) # Tam dosya yolunu al

        # Dosya adından tarihi ayrıştır
        publication_date = parse_date_from_filename(filename)

        if publication_date:
            # Dokümanı sisteme ekle (storage modülünü kullanarak)
            # add_document, zaten varsa None yerine mevcut ID'yi döndürecek şekilde güncellendi
            doc_id = add_document(filepath_str, publication_date)
            if doc_id:
                 # Eğer yeni eklendiyse (veya mevcut ID döndüyse), sayacı artırabiliriz
                 # Şimdilik sadece eklenip eklenmediğini kontrol etmek yeterli
                 # Gerçek ekleme 'add_document' içinde loglanıyor
                 pass # Şimdilik ek bir işlem yapmıyoruz

        else:
            logging.warning(f"'{filename}' için yayın tarihi bulunamadı, doküman eklenemedi.")

    logging.info(f"Toplam {processed_count} PDF dosyası tarandı.")
    # Gerçekte kaç tane yeni eklendiği bilgisini storage loglarından takip edebiliriz.

# --- Metin Çıkarma ve Kaydetme (Sonraki Fazlar İçin Hazırlık) ---
# İleride bu fonksiyonu çağırıp metinleri ayrı dosyalara kaydedebiliriz
# ve documents_df'i güncelleyebiliriz.
#
# def extract_and_save_text(doc_id: str, pdf_path: Path):
#    text = extract_text_from_pdf(pdf_path)
#    if text:
#        # Metni kaydet (örn: data/processed_data/text/{doc_id}.txt)
#        text_path = DATA_PATH / "text" / f"{doc_id}.txt"
#        text_path.parent.mkdir(parents=True, exist_ok=True)
#        try:
#            with open(text_path, 'w', encoding='utf-8') as f:
#                f.write(text)
#            logging.info(f"Metin '{text_path}' olarak kaydedildi.")
#            # documents_df'i güncelle (status='text_extracted', processed_text_path=str(text_path))
#            docs_df = load_dataframe('documents', DOC_COLUMNS)
#            doc_index = docs_df[docs_df['doc_id'] == doc_id].index
#            if not doc_index.empty:
#                docs_df.loc[doc_index, 'status'] = 'text_extracted'
#                docs_df.loc[doc_index, 'processed_text_path'] = str(text_path)
#                save_dataframe(docs_df, 'documents')
#        except Exception as e:
#            logging.error(f"Metin kaydedilirken hata ({doc_id}): {e}")