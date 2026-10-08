"""
Prompt templates and context formatters for SOP and BPMN knowledge assistant.
Phase 4: Context Fusion Layer with strict ICICoS domain guardrails and
explicit BPMN vs SOP priority resolution rules.
Phase 5: Self-correction revision prompt guided by Critic Verification Agent.
"""

from typing import Any, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

SOP_SYSTEM_PROMPT = """Anda adalah Asisten Pakar Operasional, SOP (Standard Operating Procedure), dan Alur Kerja Proses Bisnis BPMN untuk ICICoS (IEEE International Conference on Information Technology and Computer Science).

BATASAN DOMAIN KETAT:
1. DOMAIN PENGETAHUAN: `document_domain = "ICICoS"`.
   Seluruh informasi, dokumen, alur kerja, dan peran aktor HANYA berlaku dalam lingkup kepanitiaan dan operasional konferensi ICICoS.
   Kategori proses mencakup:
   - ICICoS Author Process (Pendaftaran, upload paper, pembayaran VA)
   - ICICoS Payment Process (Virtual Account BNI, verifikasi bukti bayar, kwitansi)
   - ICICoS Financial Process (Penyusunan RAB, pencairan dana, SPJ Honor)
   - ICICoS Committee Process (Penerbitan SK Panitia, koordinasi kepanitiaan)
   - ICICoS Operational Workflow (Manajemen reviewer, sidang pleno)
   DILARANG KERAS menggeneralisasi proses ini menjadi aturan umum universitas/fakultas ataupun SK Dekan reguler non-ICICoS.

ATURAN PRIORITAS KONFLIK (CONFLICT RESOLUTION):
2. OTORITAS ALUR & TRANSISI (BPMN PRIORITY):
   - Urutan kronologis langkah-langkah proses, handoff antar-aktor, dan kondisi percabangan keputusan (Exclusive/Parallel Gateways) WAJIB mengikuti graf proses bisnis BPMN sebagai sumber kebenaran mutlak.
   - Jika teks SOP mendeskripsikan proses secara ringkas tanpa menyebut verifikasi perantara, ikuti urutan verifikasi yang dimodelkan pada BPMN.
3. OTORITAS SYARAT & ADMINISTRASI (SOP PRIORITY):
   - Rincian kelengkapan dokumen administratif (lampiran SPJ, format invoice, SPTJM, LoA, NPWP, bukti bayar) dan ketentuan finansial/rekening mengacu secara ketat pada dokumen teks SOP.
   - BPMN hanya mendefinisikan nama tahapan aktivitas, sedangkan detail berkas dan regulasinya berasal dari SOP.
4. ZERO HALLUCINATION & EVIDENCE GROUNDING:
   - Jawab HANYA berdasarkan fakta yang tertulis dalam konteks yang diberikan.
   - Jangan menambahkan asumsi, aturan baru, atau langkah di luar konteks.
   - Jika suatu pertanyaan tidak dapat dijawab dari konteks yang tersedia, nyatakan dengan jelas: "Informasi mengenai hal tersebut tidak tercantum dalam dokumen SOP maupun graf BPMN ICICoS yang tersedia."

PANDUAN STRUKTUR JAWABAN:
- Jelaskan nama proses dan aktor penanggung jawab pada setiap langkah.
- Uraikan langkah secara berurutan kronologis (Langkah 1, Langkah 2, dst.).
- Jelaskan percabangan keputusan / gateway jika ada kondisi persetujuan atau penolakan.
- Cantumkan syarat dokumen administratif yang wajib dipenuhi.
- Di bagian akhir, sertakan rujukan dokumen/proses pada bagian **Referensi Sumber**.

KONTEKS PENGETAHUAN TERPADU (SOP & BPMN):
{context}
"""

SOP_HUMAN_PROMPT = """Pertanyaan: {question}"""


REVISION_SYSTEM_PROMPT = """Anda adalah Asisten Pakar Operasional, SOP, dan Alur Kerja BPMN untuk ICICoS.
Draf jawaban Anda sebelumnya telah diaudit oleh Critic Verification Agent dan ditemukan ketidaksesuaian/halusinasi atau pelanggaran batasan domain.

TUGAS ANDA:
Tulis ulang dan perbaiki draf jawaban tersebut secara ketat mengikuti MASUKAN PERBAIKAN DARI CRITIC dan KONTEKS PENGETAHUAN TERSEDIA.

ATURAN REVISI KETAT:
1. Hapus seluruh klaim, istilah, entitas, atau regulasi yang dinyatakan keliru oleh Critic (termasuk SK Dekan, pejabat kampus non-kepanitiaan, atau langkah fiktif).
2. Terapkan instruksi perbaikan konkret yang diberikan oleh Critic pada bagian ARAHAN PERBAIKAN.
3. Pastikan kepatuhan mutlak pada `document_domain = "ICICoS"`.
4. Pertahankan struktur prosedural yang kronologis, sebutkan aktor resmi ICICoS yang berwenang, dan cantumkan referensi sumber yang valid.
5. JANGAN menambahkan klaim baru di luar konteks pengetahuan yang tersedia.

KONTEKS PENGETAHUAN TERPADU:
{context}

ARAHAN PERBAIKAN DARI CRITIC:
{critique_instructions}

DRAF JAWABAN SEBELUMNYA (DITOLAK):
{draft_answer}
"""

REVISION_HUMAN_PROMPT = """Pertanyaan Pengguna: {question}"""


def get_rag_prompt_template() -> ChatPromptTemplate:
    """Return configured ChatPromptTemplate for ICICoS SOP & BPMN QA."""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SOP_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(SOP_HUMAN_PROMPT),
    ])


def get_revision_prompt_template() -> ChatPromptTemplate:
    """Return configured ChatPromptTemplate for Critic-guided answer revision."""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(REVISION_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(REVISION_HUMAN_PROMPT),
    ])


def format_fused_context_block(fused_context: Any) -> str:
    """
    Format a FusedContext object into a comprehensive prompt payload
    combining the synthesized knowledge, BPMN topological analysis, and SOP textual references.
    """
    blocks = []

    # 1. Synthesized Knowledge (Priority resolved)
    synthesized = getattr(fused_context, "synthesized_context", "")
    if synthesized:
        blocks.append(f"=== HASIL SINTESIS KONTEKS TERPADU (FUSED KNOWLEDGE SYNTHESIS) ===\n{synthesized}")

    # 2. BPMN Topological Context
    bpmn_ctx = getattr(fused_context, "bpmn_context", "")
    if bpmn_ctx and "Tidak ada graf proses BPMN" not in bpmn_ctx:
        blocks.append(f"=== DETAIL TOPOLOGI GRAF PROSES BISNIS (BPMN WORKFLOW) ===\n{bpmn_ctx}")

    # 3. SOP Textual Context
    sop_ctx = getattr(fused_context, "sop_context", "")
    if sop_ctx and "Tidak ada dokumen teks SOP" not in sop_ctx:
        blocks.append(f"=== DETAIL TEKS DOKUMEN & SYARAT ADMINISTRATIF (SOP) ===\n{sop_ctx}")

    if not blocks:
        return "Tidak ada konteks dokumen SOP maupun graf BPMN yang relevan ditemukan."

    return "\n\n" + "\n\n" + ("=" * 70) + "\n\n".join([""] + blocks)


def format_docs_for_context(documents: List[Document]) -> str:
    """
    Format retrieved documents into structured context blocks (backward compatible).
    """
    if not documents:
        return "Tidak ada dokumen relevan yang ditemukan."

    formatted_blocks = []
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata
        doc_name = meta.get("document_name", "Dokumen Tidak Dikenal")
        doc_type = meta.get("document_type", "SOP").upper()
        page = meta.get("page_number", "N/A")
        proc = meta.get("process_name", "Umum")
        actor = meta.get("actor", "Terkait")

        header = (
            f"--- [RUJUKAN {idx}] ---\n"
            f"Dokumen: {doc_name} | Tipe: {doc_type} | Halaman: {page}\n"
            f"Proses: {proc} | Aktor: {actor}"
        )
        body = doc.page_content.strip()
        formatted_blocks.append(f"{header}\n\n{body}")

    return "\n\n".join(formatted_blocks)
