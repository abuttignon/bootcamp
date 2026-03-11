from pydantic import BaseModel, Field


class TextSegment(BaseModel):
    text: str
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None


class NormalizedDocument(BaseModel):
    doc_id: str
    title: str
    source_path: str
    source_type: str
    folder_context: list[str] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    chapter_hints: list[str] = Field(default_factory=list)
    page_count: int | None = None
    segments: list[TextSegment] = Field(default_factory=list)


class ProcessedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    source_path: str
    source_type: str
    folder_context: list[str] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    chapter_hints: list[str] = Field(default_factory=list)
    token_estimate: int = 0