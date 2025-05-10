# src/analysis/temporal.py (Yarı ömür fonksiyonu eklendi)

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import logging
from pathlib import Path
from datetime import datetime

# Yerel modüllerimizi içe aktaralım
from src.data_management import storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_concept_frequencies(time_period: str = 'Y') -> pd.DataFrame | None:
    """
    Konseptlerin zaman içindeki kullanım sıklıklarını hesaplar. (Önceki kodla aynı)
    """
    logging.info(f"Konsept frekansları '{time_period}' periyodu için hesaplanıyor...")
    mentions_df = storage.load_dataframe('mentions', storage.MENTION_COLUMNS)
    documents_df = storage.load_dataframe('documents', storage.DOC_COLUMNS)

    if mentions_df is None or documents_df is None:
        logging.error("Mention veya Document verisi yüklenemedi. Frekans hesaplanamıyor.")
        return None
    if mentions_df.empty:
        logging.warning("Mention verisi boş. Frekans hesaplanamıyor.")
        return pd.DataFrame(columns=['concept_id', 'time_period_start', 'frequency'])
    if documents_df.empty:
        logging.warning("Document verisi boş. Tarih bilgisi alınamıyor, frekans hesaplanamıyor.")
        return pd.DataFrame(columns=['concept_id', 'time_period_start', 'frequency'])

    docs_subset = documents_df[['doc_id', 'publication_date']].copy()
    try:
        docs_subset['publication_date'] = pd.to_datetime(docs_subset['publication_date'], errors='coerce')
    except Exception as e:
         logging.error(f"Dokümanlardaki 'publication_date' sütunu datetime'a çevrilemedi: {e}")
         return None

    original_doc_count = len(docs_subset)
    docs_subset.dropna(subset=['publication_date'], inplace=True)
    valid_date_count = len(docs_subset)
    if original_doc_count > valid_date_count:
        logging.warning(f"{original_doc_count - valid_date_count} dokümanın geçerli yayın tarihi yok, frekans hesaplamasına dahil edilmeyecek.")

    if docs_subset.empty:
        logging.warning("Geçerli yayın tarihine sahip doküman bulunamadı. Frekans hesaplanamıyor.")
        return pd.DataFrame(columns=['concept_id', 'time_period_start', 'frequency'])

    mentions_with_dates = pd.merge(mentions_df, docs_subset, on='doc_id', how='inner')

    if mentions_with_dates.empty:
        logging.warning("Mention'lar ile doküman tarihleri birleştirilemedi veya sonuç boş.")
        return pd.DataFrame(columns=['concept_id', 'time_period_start', 'frequency'])

    logging.info(f"{len(mentions_with_dates)} mention için tarih bilgisi bulundu.")

    try:
        frequency_df = mentions_with_dates.groupby(
            ['concept_id', pd.Grouper(key='publication_date', freq=time_period)]
        ).size().reset_index(name='frequency')
        frequency_df.rename(columns={'publication_date': 'time_period_start'}, inplace=True)
        logging.info(f"Frekans hesaplaması tamamlandı. {len(frequency_df)} satır sonuç üretildi.")
        frequency_df.sort_values(by=['concept_id', 'time_period_start'], inplace=True)
        return frequency_df
    except Exception as e:
        logging.exception(f"Frekans hesaplanırken hata oluştu: {e}")
        return None

# --- YENİ: Yarı Ömür Hesaplama ---

def exponential_decay(t, A, decay_rate):
    """Üstel bozulma fonksiyonu: A * exp(-decay_rate * t)."""
    # Decay rate negatif olmamalı (bozunma varsayımı)
    decay_rate = max(0, decay_rate) # Negatifse sıfır yap
    return A * np.exp(-decay_rate * t)

def calculate_half_life(concept_id: str,
                        frequency_df: pd.DataFrame,
                        concept_name: str | None = None,
                        min_data_points: int = 4,
                        min_decay_rate: float = 1e-6) -> float | None:
    """
    Verilen konsept için frekans verisine üstel bozulma modeli uygulayarak
    yarı ömrü (yıl olarak) hesaplar.

    Args:
        concept_id (str): Hesaplanacak konseptin ID'si.
        frequency_df (pd.DataFrame): calculate_concept_frequencies'ten dönen DataFrame.
                                     ('concept_id', 'time_period_start', 'frequency' sütunları olmalı).
        concept_name (str | None): Loglama için konseptin adı (opsiyonel).
        min_data_points (int): Yarı ömür hesaplamak için gereken minimum zaman noktası sayısı.
        min_decay_rate (float): Kabul edilebilir minimum bozunma oranı (çok küçükse yarı ömür sonsuz kabul edilir).

    Returns:
        float | None: Hesaplanan yarı ömür (yıl olarak) veya hesaplanamazsa None.
                      np.inf dönebilir eğer bozunma oranı çok küçükse.
    """
    log_prefix = f"Yarı Ömür ({concept_name or concept_id}):"

    if frequency_df is None or frequency_df.empty:
        logging.warning(f"{log_prefix} Frekans verisi boş.")
        return None

    # Konsepte ait veriyi filtrele ve zamana göre sırala
    concept_data = frequency_df[frequency_df['concept_id'] == concept_id].sort_values(by='time_period_start').copy()

    # Yeterli veri noktası var mı?
    if len(concept_data) < min_data_points:
        logging.info(f"{log_prefix} Yeterli veri noktası yok ({len(concept_data)} < {min_data_points}). Hesaplama yapılamıyor.")
        return None

    # Zamanı sayısal değere çevir (ilk yıldan itibaren geçen yıl sayısı)
    try:
        # İlk zaman noktasını t=0 kabul et
        start_date = concept_data['time_period_start'].min()
        # Zaman farkını gün olarak hesapla ve yıla çevir
        concept_data['time_elapsed_years'] = (concept_data['time_period_start'] - start_date).dt.days / 365.25
    except Exception as e:
        logging.error(f"{log_prefix} Zaman farkı hesaplanırken hata: {e}")
        return None

    time_values = concept_data['time_elapsed_years'].values
    frequency_values = concept_data['frequency'].values

    # Frekanslar artıyor mu veya sabit mi kontrol et (basit kontrol)
    # Eğer son değer ilk değerden büyükse veya tüm değerler aynıysa, bozunma yok kabul et
    if frequency_values[-1] > frequency_values[0] or np.all(frequency_values == frequency_values[0]):
         logging.info(f"{log_prefix} Veride belirgin bir azalma gözlenmedi. Yarı ömür hesaplanamıyor.")
         return None # Veya np.inf? Şimdilik None.

    # Modeli uydurmak için başlangıç tahminleri
    initial_A_guess = frequency_values[0] # İlk frekans değeri
    initial_lambda_guess = 0.1 # Küçük pozitif bir bozunma oranı tahmini

    try:
        # curve_fit ile modeli verilere uydur
        params, covariance = curve_fit(
            exponential_decay,
            time_values,
            frequency_values,
            p0=[initial_A_guess, initial_lambda_guess],
            bounds=([0, 0], [np.inf, np.inf]) # Parametrelerin pozitif olmasını sağla
            # maxfev artırılabilir eğer "Optimal parameters not found" hatası alınırsa
        )

        A_fit, decay_rate_fit = params

        # Bozunma oranı anlamlı mı?
        if decay_rate_fit < min_decay_rate:
            logging.info(f"{log_prefix} Hesaplanan bozunma oranı ({decay_rate_fit:.4f}) çok düşük. Yarı ömür sonsuz kabul ediliyor.")
            return np.inf # Sonsuz yarı ömür

        # Yarı ömrü hesapla: ln(2) / decay_rate
        half_life_years = np.log(2) / decay_rate_fit
        logging.info(f"{log_prefix} Başarıyla hesaplandı. A={A_fit:.2f}, Bozunma Oranı={decay_rate_fit:.4f}, Yarı Ömür={half_life_years:.2f} yıl.")
        return half_life_years

    except RuntimeError as e:
        logging.warning(f"{log_prefix} Üstel bozulma modeli uydurulamadı: {e}. Yarı ömür hesaplanamıyor.")
        return None
    except Exception as e:
        logging.exception(f"{log_prefix} Yarı ömür hesaplanırken beklenmeyen hata: {e}")
        return None