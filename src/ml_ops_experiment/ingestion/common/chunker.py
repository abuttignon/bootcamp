import hashlib

from ml_ops_experiment.ingestion.common.data_models import NormalizedDocument, ProcessedChunk


class StructureAwareChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: NormalizedDocument) -> list[ProcessedChunk]:
        chunks: list[ProcessedChunk] = []
        chunk_index = 0

        for segment in document["segments"]:
            text = " ".join(segment["text"].split())
            if not text:
                continue

            for start, end, piece in self._split_text(text):
                chunk_id = self._chunk_id(document["doc_id"], segment["section_path"], start, end, piece)
                chunks.append(
                    ProcessedChunk(
                        chunk_id=chunk_id,
                        doc_id=document["doc_id"],
                        chunk_index=chunk_index,
                        text=piece,
                        source_path=document["source_path"],
                        source_type=document["source_type"],
                        folder_context=document["folder_context"],
                        section_path=segment["section_path"],
                        page_start=segment["page"],
                        page_end=segment["page"],
                        chapter_hints=document["chapter_hints"],
                        token_estimate=max(1, len(piece) // 4),
                    )
                )
                chunk_index += 1

        return chunks

    def _split_text(self, text: str) -> list[tuple[int, int, str]]:
        if len(text) <= self.chunk_size:
            return [(0, len(text), text)]

        windows: list[tuple[int, int, str]] = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                windows.append((start, end, piece))
            if end >= len(text):
                break
            start += step

        return windows

    @staticmethod
    def _chunk_id(
            doc_id: str,
            section_path: list[str],
            start: int,
            end: int,
            text: str,
    ) -> str:
        raw = f"{doc_id}|{'/'.join(section_path)}|{start}:{end}|{text[:80]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

