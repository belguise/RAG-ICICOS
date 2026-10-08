"""
Query Understanding Agent.
Phase 2 Agentic Layer: Normalizes raw user input, expands domain acronyms,
resolves typos, and transforms informal queries into search-optimized procedural questions.
"""

from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from agentic_rag_sop.core.llm import get_llm

QUERY_REWRITE_PROMPT = """Anda adalah Query Understanding Agent untuk sistem SOP (Standard Operating Procedure) dan Proses Bisnis Konferensi Internasional (ICICoS).

TUGAS:
Ubah pertanyaan mentah dari pengguna yang mungkin mengandung bahasa santai/gaul, salah ketik (typo), singkatan informal, atau struktur yang ambigu menjadi SATU kalimat pertanyaan formal, lengkap, dan baku yang optimal untuk pencarian semantik pada dokumen SOP.

ATURAN NORMALISASI:
1. PERBAIKI TYPO & BAHASA INFORMAL:
   - "gmn", "gimana" -> "Bagaimana"
   - "byr", "bayar" -> "pembayaran / membayar"
   - "kl", "kalo", "klo" -> "apabila / jika"
   - "apa aja", "apakah", "apa sajakah" -> "Apa saja"
   - "ngajuin" -> "mengajukan"
   - "dpt", "bs" -> "dapat / bisa"
   - "kpn" -> "kapan"

2. PERTAHANKAN SINGKATAN RESMI & PERLUAS SECARA CERMAT:
   - "SPJ" -> "SPJ" (Pertahankan kata "SPJ" karena seluruh judul dan bab SOP ICICoS menggunakan istilah "SPJ")
   - "VA" -> "Virtual Account (VA)"
   - "RAB" -> "Rencana Anggaran Biaya (RAB)"
   - "SK" -> "Surat Keputusan (SK)"
   - "LoA" -> "Letter of Acceptance (LoA)"
   - "BoA / BoC" -> "Book of Abstract (BoA) / Book of Content (BoC)"
   - "WCU" -> "World Class University (WCU)"
   - "CMANIS" -> "CMANIS"
   - "EDAS" -> "EDAS"
   - "Keynote" -> "Keynote Speaker"

3. PERTAHANKAN ENTITAS DAN NAMA PROSES ASLI (ZERO SEMANTIC DRIFT):
   - Wajib pertahankan nama proses persis seperti aslinya: "Submit to Present", "Camera Ready", "Review Paper", "Registrasi & Pembayaran", dll.
   - Wajib pertahankan nama peran/aktor spesifik: "Author", "Reviewer", "TPC Chair", "Bendahara", "Sekretaris", "Track Chair", "Webmaster", "Ketua Panitia", dll.
   - Wajib pertahankan nama sistem dan platform: "EDAS", "CMANIS", "IEEE Xplore", "BNI", "ICICoS", dll.
   - JANGAN menambahkan asumsi baru, jangan memperluas ke topik di luar yang ditanyakan pengguna, dan jangan mengubah pertanyaan yang sudah baku menjadi kalimat berbelit-belit.
   - Jika pertanyaan mentah SUDAH berupa kalimat baku dan formal yang jelas, pertahankan pertanyaan tersebut tanpa perubahan substantif.

4. FORMAT KELUARAN:
   - Keluarkan HANYA satu kalimat pertanyaan yang telah diperbaiki/dinormalisasi.
   - Jangan menambahkan tanda petik, awalan seperti "Pertanyaan hasil revisi:", atau teks penjelasan tambahan.

---
CONTOH 1:
Pertanyaan Mentah: gmn byr kl bni beda?
Pertanyaan Hasil Normalisasi: Bagaimana prosedur pembayaran apabila transfer dilakukan melalui bank selain BNI?

CONTOH 2:
Pertanyaan Mentah: spj keynote lampiran apa aja?
Pertanyaan Hasil Normalisasi: Apa saja dokumen lampiran yang wajib disertakan untuk SPJ Honor Keynote Speaker?

CONTOH 3:
Pertanyaan Mentah: Dokumen apa yang harus diunggah Author pada tahap Submit to Present?
Pertanyaan Hasil Normalisasi: Dokumen apa yang harus diunggah Author pada tahap Submit to Present?
---

Pertanyaan Mentah: {raw_query}
Pertanyaan Hasil Normalisasi:"""


class QueryUnderstandingAgent:
    """Agent that cleans, formalizes, and enriches queries before vector retrieval."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm or get_llm()
        self.prompt = PromptTemplate.from_template(QUERY_REWRITE_PROMPT)
        self.output_parser = StrOutputParser()
        self._chain = self.prompt | self.llm | self.output_parser

    def rewrite_query(self, raw_query: str) -> str:
        """
        Transform informal/typo-laden query into formal procedural inquiry.
        Falls back to raw query if transformation encounters any issue.
        """
        cleaned_raw = raw_query.strip()
        if not cleaned_raw:
            return ""

        try:
            rewritten = self._chain.invoke({"raw_query": cleaned_raw}).strip()
            # Clean up accidental quotes or formatting returned by LLM
            rewritten = rewritten.strip("\"'").strip()
            return rewritten if rewritten else cleaned_raw
        except Exception:
            # Graceful fallback: return original query if agent encounters an issue
            return cleaned_raw
