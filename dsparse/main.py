from llm_file_parsing import parse_file_llm
from file_parsing import parse_file
from dsparse.semantic_sectioning import get_sections
from dsParse.dsparse.chunking import get_chunks_from_sections, get_sections_and_chunks_naive


def parse_and_chunk(file_path: str, file_or_url: str = "file", use_llm_file_parsing: bool = False, \
                    use_semantic_sectioning: bool = True, chunking_method: str = "semantic", \
                    file_parsing_model: str = "gpt-4o-mini", semantic_sectioning_and_chunking_model: str = "gpt-4o-mini", \
                    chunk_size: int = 800, min_length_for_chunking: int = 2000) -> dict:
                    

    # Parse the file
    if use_llm_file_parsing:
        document_text = parse_file_llm(file_path, file_or_url, file_parsing_model)
    else:
        document_text = parse_file(file_path, file_or_url)

    # Get the sections if semantic sectioning is enabled
    if use_semantic_sectioning:
        sections, document_lines = get_sections(document_text, max_characters=20000, model=semantic_sectioning_and_chunking_model)
        # Get the chunks from the sections
        formatted_sections = get_chunks_from_sections(sections, document_lines, semantic_sectioning_and_chunking_model, chunk_size, min_length_for_chunking, chunking_method)
    else:
        # There will just be a single section for the entire document
        formatted_sections = get_sections_and_chunks_naive(document_text, chunk_size, 0)
    
    return formatted_sections

