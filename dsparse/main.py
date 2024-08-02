from dsparse.semantic_sectioning import get_sections
from dsparse.semantic_chunking import get_chunks_from_segments


def parse_and_chunk(file_path: str, llm_file_parsing: bool = False, semantic_sectioning: bool = True, \
                    chunking_method: str = "semantic", file_parsing_model: str = "gpt-4o-mini", \
                    semantic_sectioning_and_chunking_model: str = "gpt-4o-mini", chunk_size: int = 800, \
                    min_length_for_chunking: int = 2000) -> dict:

    # Parse the file

    # Get the sections if semantic sectioning is enabled

    # Get the chunks from the sections
    pass