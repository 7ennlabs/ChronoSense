# src/extraction/extractor.py (AttributeError DÜZELTİLMİŞ TAM KOD)

import spacy
from pathlib import Path
import logging
import itertools
import re
import string

# Yerel modüllerimizi içe aktaralım
from src.data_management import storage
from src.data_management import loaders # extract_text_from_pdf için

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- spaCy Model Yükleme ---
nlp = None
STOP_WORDS = set()
try:
    nlp = spacy.load("en_core_web_lg")
    logging.info("spaCy 'en_core_web_lg' modeli başarıyla yüklendi.")
    STOP_WORDS = nlp.Defaults.stop_words
except OSError:
    logging.error("spaCy 'en_core_web_lg' modeli bulunamadı. Lütfen indirin: python -m spacy download en_core_web_lg")

# --- Konsept Belirleme Kriterleri (Aynı kaldı) ---
TRUSTED_ENTITY_LABELS = {"PRODUCT", "ORG", "WORK_OF_ART"}
OTHER_ENTITY_LABELS = {"PERSON", "EVENT", "LAW", "NORP", "FAC", "GPE", "LOC"}
NOUN_CHUNK_PATTERNS = re.compile(r".*\b(learning|network|model|algorithm|system|technique|approach|agent|layer|architecture|transformer|attention)\b$", re.IGNORECASE)
MIN_CONCEPT_WORDS = 1
MAX_CONCEPT_WORDS = 6
AI_KEYWORDS = {"artificial intelligence", "machine learning", "deep learning",
               "neural network", "reinforcement learning", "transformer", "llm",
               "large language model", "computer vision", "natural language processing",
               "algorithm", "model", "gpt", "bert", "agent", "attention", "supervised",
               "unsupervised", "classification", "regression", "clustering"}
# --- İlişki Çıkarımı için Fiiller ve Desenler ---
RELATION_VERBS = {
    "use": "USES", "utilize": "USES", "apply": "USES", "employ": "USES",
    "improve": "IMPROVES", "enhance": "IMPROVES", "extend": "IMPROVES", "outperform": "IMPROVES",
    "base on": "BASED_ON", "rely on": "BASED_ON",
    "compare": "COMPARES_TO", "relate": "RELATED_TO", "associate": "RELATED_TO", "link": "RELATED_TO",
    "propose": "PROPOSES", "introduce": "PROPOSES", "develop": "PROPOSES",
}

def normalize_and_validate_concept(text: str, is_entity: bool = False, entity_label: str = "") -> str | None:
    """ Verilen metni temizler, doğrular... """
    cleaned_text = text.strip()
    word_count = len(cleaned_text.split())
    if not (MIN_CONCEPT_WORDS <= word_count <= MAX_CONCEPT_WORDS): return None
    if cleaned_text and all(word.lower() in STOP_WORDS for word in re.findall(r'\b\w+\b', cleaned_text)): return None
    if cleaned_text.isdigit() or all(c in string.punctuation for c in cleaned_text): return None
    generic_phrases = {"this approach", "these models", "this technique", "this system",
                       "the model", "the algorithm", "the method", "the approach",
                       "the system", "the technique", "our model", "our approach"}
    if cleaned_text.lower() in generic_phrases: return None
    return cleaned_text

def find_verb_relation(token1: spacy.tokens.Token, token2: spacy.tokens.Token) -> tuple[str, str] | None:
    """ İki token arasındaki dependency path'e bakarak fiil ilişkisi bulur. """
    common_ancestor = None
    ancestors1 = list(token1.ancestors)
    ancestors2 = list(token2.ancestors)
    for t in reversed(ancestors1):
        if t in ancestors2:
            common_ancestor = t
            break
    if not common_ancestor: return None

    verb1 = None; head = token1
    while head != common_ancestor:
        if head.pos_ == "VERB": verb1 = head; break
        head = head.head
    verb2 = None; head = token2
    while head != common_ancestor:
        if head.pos_ == "VERB": verb2 = head; break
        head = head.head

    verb_token = None
    if common_ancestor.pos_ == "VERB": verb_token = common_ancestor
    elif verb1 and verb1 == verb2: verb_token = verb1
    # elif verb1: verb_token = verb1 # Tek taraflı fiilleri şimdilik yoksayalım
    # elif verb2: verb_token = verb2
    elif common_ancestor.head.pos_ == "VERB": verb_token = common_ancestor.head

    if verb_token:
        verb_lemma = verb_token.lemma_
        # *** HATA DÜZELTME: Bu satırı geçici olarak kaldırıyoruz/yorum yapıyoruz ***
        # if verb_token.is_aux or verb_token.is_stop:
        #     return None
        # **********************************************************************
        for verb, rel_type in RELATION_VERBS.items():
            if verb_lemma == verb or verb_lemma in verb.split():
                 logging.debug(f"Fiil ilişkisi bulundu: {token1.text}... {verb_lemma} ({rel_type}) ...{token2.text}")
                 return rel_type, verb_lemma
    return None

def extract_entities_and_relations(text: str, doc_id: str):
    """ Metinden konseptleri, mention'ları ve İYİLEŞTİRİLMİŞ ilişkileri çıkarır. """
    if not nlp: raise RuntimeError("spaCy modeli yüklenemedi.")
    spacy_doc = nlp(text)
    potential_concepts = {}; mentions_in_doc = []; valid_mentions = {}
    processed_spans = set(); added_relations = set()

    # 1. Adayları Bul
    candidates = []
    for ent in spacy_doc.ents:
         if ent.label_ in TRUSTED_ENTITY_LABELS or ent.label_ in OTHER_ENTITY_LABELS:
             candidates.append({"span": ent, "is_entity": True, "label": ent.label_})
    for chunk in spacy_doc.noun_chunks:
         is_covered = any(ent_data["span"].start_char <= chunk.start_char and ent_data["span"].end_char >= chunk.end_char
                          for ent_data in candidates if ent_data["is_entity"])
         if not is_covered:
             candidates.append({"span": chunk, "is_entity": False, "label": ""})

    # 2. Adayları Filtrele, Normalleştir ve Kaydet
    for data in candidates:
        span = data["span"];
        if span in processed_spans: continue
        validated_text = normalize_and_validate_concept(span.text, data["is_entity"], data["label"])
        if not validated_text: processed_spans.add(span); continue
        concept_lemma = span.lemma_.lower().strip() if span.lemma_ else validated_text.lower()
        is_concept = False
        if data["is_entity"] and data["label"] in TRUSTED_ENTITY_LABELS: is_concept = True
        elif NOUN_CHUNK_PATTERNS.match(validated_text): is_concept = True
        elif any(keyword in concept_lemma.split() or keyword in validated_text.lower().split() for keyword in AI_KEYWORDS): is_concept = True
        elif validated_text.isupper() and len(validated_text) > 1 and len(validated_text) < 6: is_concept = True

        if is_concept:
            concept_id = storage.add_concept(validated_text)
            if concept_id:
                mention_id = storage.add_mention(
                    doc_id=doc_id, concept_id=concept_id,
                    context=span.sent.text, start=span.start_char, end=span.end_char
                )
                if mention_id:
                    mention_data = {
                        "mention_id": mention_id, "concept_id": concept_id,
                        "start_char": span.start_char, "end_char": span.end_char,
                        "sentence": span.sent, "root_token": span.root
                    }
                    mentions_in_doc.append(mention_data); valid_mentions[mention_id] = mention_data
        processed_spans.add(span)

    # 3. İlişkileri Çıkar
    for sentence in spacy_doc.sents:
        mentions_in_sentence = [m for m in mentions_in_doc if m["sentence"] == sentence]
        if len(mentions_in_sentence) >= 2:
            for m1_data, m2_data in itertools.combinations(mentions_in_sentence, 2):
                c1_id = m1_data["concept_id"]; c2_id = m2_data["concept_id"]
                if c1_id == c2_id: continue
                rel_pair = tuple(sorted((c1_id, c2_id)))
                if rel_pair in added_relations: continue
                relation_found = False
                relation_info = find_verb_relation(m1_data["root_token"], m2_data["root_token"])
                if relation_info:
                    rel_type, verb = relation_info
                    storage.add_relationship(
                        source_concept_id=c1_id, target_concept_id=c2_id, rel_type=rel_type,
                        mention_id=m1_data["mention_id"], doc_id=doc_id, sentence=sentence.text
                    )
                    relation_found = True; added_relations.add(rel_pair)
                if not relation_found:
                    storage.add_relationship(
                        source_concept_id=c1_id, target_concept_id=c2_id, rel_type="RELATED_TO",
                        mention_id=m1_data["mention_id"], doc_id=doc_id, sentence=sentence.text
                    )
                    added_relations.add(rel_pair)

def process_documents_for_extraction():
    """ Dokümanları işler ve durumu günceller... (Öncekiyle aynı) """
    if not nlp: raise RuntimeError("spaCy modeli yüklenemedi.")
    logging.info("Gelişmiş bilgi çıkarımı için dokümanlar işleniyor...")
    documents_df = storage.load_dataframe('documents', storage.DOC_COLUMNS)
    docs_to_process = documents_df[documents_df['status'] == 'added']
    if docs_to_process.empty:
        logging.info("Durumu 'added' olan ve işlenecek doküman bulunamadı.")
        return
    processed_count = 0; failed_count = 0
    for index, doc_row in docs_to_process.iterrows():
        doc_id = doc_row['doc_id']; filepath = Path(doc_row['filepath'])
        logging.info(f"İşleniyor: {filepath.name} (ID: {doc_id})")
        text = loaders.extract_text_from_pdf(filepath)
        if text:
            try:
                extract_entities_and_relations(text, doc_id)
                storage.update_document_status(doc_id, 'processed_v3') # Yeni versiyon durumu
                processed_count += 1
            except Exception as e:
                logging.exception(f"'{filepath.name}' işlenirken BEKLENMEYEN HATA oluştu: {e}")
                storage.update_document_status(doc_id, 'extraction_failed_v3')
                failed_count += 1
        else:
            logging.warning(f"Metin çıkarılamadı: {filepath.name}")
            storage.update_document_status(doc_id, 'text_extraction_failed')
            failed_count += 1
    logging.info(f"Gelişmiş bilgi çıkarımı tamamlandı. Başarılı: {processed_count}, Başarısız: {failed_count}")