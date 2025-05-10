# reset_status.py
import pandas as pd
# storage modülünü doğru import etmek için src'yi sys.path'e ekleyebilir veya PYTHONPATH ayarlayabiliriz.
# En kolayı çalıştırmadan önce PYTHONPATH ayarlamak veya geçici olarak sys.path'e eklemek.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_management.storage import load_dataframe, save_dataframe, DOC_COLUMNS

print("Doküman durumları 'added' olarak sıfırlanıyor...")
df = load_dataframe('documents', DOC_COLUMNS)

if not df.empty:
    # Sadece işlenmiş veya hata almış olanları sıfırla
    reset_mask = df['status'].str.startswith('processed', na=False) | df['status'].str.contains('failed', na=False)
    if reset_mask.any():
        df.loc[reset_mask, 'status'] = 'added'
        save_dataframe(df, 'documents')
        print(f"{reset_mask.sum()} dokümanın durumu 'added' olarak sıfırlandı.")
    else:
        print("Durumu sıfırlanacak doküman bulunamadı ('processed' veya 'failed' durumunda olan).")
else:
    print("Doküman DataFrame'i bulunamadı veya boş.")