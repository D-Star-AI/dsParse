from pydantic import BaseModel, Field
from typing import List, Dict, Any
from anthropic import Anthropic
from openai import OpenAI
import instructor
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunk(BaseModel):
    start_index: int = Field(description="line number where the section begins (inclusive)")

class StructuredChunk(BaseModel):
    """obtains meaningful sections, each centered around a single concept/topic"""
    chunks: List[Chunk] = Field(description="a list of chunks of the document")


system_prompt = """
Read the document below and extract a StructuredChunk object from it where each chunk of the document is centered around a single concept/topic. Whenever possible, your chunks should match up with the natural sections of the document (i.e. Introduction, Conclusion, References, etc.). Chunks can vary in length, but should generally be anywhere from a few sentences to a paragraph.
If there are no natural sections in the document, you should create chunks that are in the range of 2000 characters each.
Each line of the document is marked with its line number in square brackets (e.g. [1], [2], [3], etc). Use the line numbers to indicate a chunk start.
The start line numbers will be treated as inclusive. For example, if the first line of a chunk is line 5, the start_index should be 5. Your goal is to find the starting line number of a given chunk, where a chunk is a group of lines that are thematically related.
The first chunk must start at the first line number of the document ({start_line} in this case). The chunks MUST cover the entire document.
Note: the document provided to you may just be an excerpt from a larger document, rather than a complete document. Therefore, you can't always assume, for example, that the first line of the document is the beginning of the Introduction section and the last line is the end of the Conclusion section (if those section are even present).
YOU MUST CREATE BETWEEN {min_chunks} and {max_chunks} CHUNKS.
"""


def get_target_num_chunks(text: str, max_characters: int = 2000) -> List[int]:
    """ This function will return the number of chunks that the text should be divided into """
    expected_num_chunks = len(text) // max_characters + 1

    if expected_num_chunks < 2:
        return 1, 2
    
    return expected_num_chunks - 1, expected_num_chunks + 1

def check_for_fallback_chunking_usage(min_num_chunks, max_num_chunks, chunks):
    # If the number of chunks is well off the expected, then we will use the basic character chunking
    if len(chunks) < min_num_chunks // 2 or len(chunks) > max_num_chunks * 2:
        return True
    else:
        return False

def get_chunk_text(start_index: int, end_index: int, document_lines: List[str]) -> str:
    chunk_text = ""
    # Using end_index+1 because the end_index is inclusive, but the range function is exclusive
    for i in range(start_index, end_index+1):
        chunk_text += f"{document_lines[i]}\n"
    return chunk_text


def get_structured_chunks(document_with_line_numbers: str, start_line: int, min_chunks: int, max_chunks: int, llm_provider: str, model: str) -> StructuredChunk:

    """
    Note: This function relies on Instructor, which only supports certain model providers. That's why this function doesn't use the LLM abstract base class that is used elsewhere in the project.
    """

    if llm_provider == "anthropic":
        client = instructor.from_anthropic(Anthropic())
        return client.chat.completions.create(
            model=model,
            response_model=StructuredChunk,
            max_tokens=4000,
            temperature=0.0,
            system=system_prompt.format(start_line=start_line, min_chunks=min_chunks, max_chunks=max_chunks),
            messages=[
                {
                    "role": "user",
                    "content": document_with_line_numbers,
                },
            ],
        )
    elif llm_provider == "openai":
        client = instructor.from_openai(OpenAI())
        return client.chat.completions.create(
            model=model,
            response_model=StructuredChunk,
            max_tokens=4000,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.format(start_line=start_line, min_chunks=min_chunks, max_chunks=max_chunks),
                },
                {
                    "role": "user",
                    "content": document_with_line_numbers,
                },
            ],
        )
    else:
        raise ValueError("Invalid provider. Must be either 'anthropic' or 'openai'.")
    

def get_chunks_from_segments(segments: List[Dict[str, Any]], document_lines: List[str], llm_provider: str, model: str, chunk_size: int = 800, min_length_for_chunking: int = 2000,) -> List[Dict[str, Any]]:

    all_chunk_dicts = []
    chunk_index = 0
    for i,segment in enumerate(segments):

        start_index = segment["start"]
        end_index = segment["end"]
        
        # Annotate the document lines with line numbers
        document_with_line_numbers = ""
        for i in range(start_index, end_index+1):
            document_with_line_numbers += f"[{i}] {document_lines[i]}\n"
        
        # If the length of the content is less than 2000 characters, then we will use the entire section as one chunk
        if (len(document_with_line_numbers) < min_length_for_chunking):
            # The entire section will be one chunk
            chunk_text = get_chunk_text(start_index, end_index, document_lines)
            all_chunk_dicts.append({
                "chunk_text": chunk_text,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
            continue

        min_num_chunks, max_num_chunks = get_target_num_chunks(document_with_line_numbers)
        structured_chunks = get_structured_chunks(document_with_line_numbers, start_index, min_num_chunks, max_num_chunks, llm_provider, model)
        new_chunks = structured_chunks.chunks
        use_fallback = check_for_fallback_chunking_usage(min_num_chunks, max_num_chunks, new_chunks)

        if use_fallback:
            # Fallback to basic character chunking
            document_text = ""
            for i in range(start_index, end_index+1):
                document_text += f"{document_lines[i]}\n"
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, length_function=len)
            texts = text_splitter.create_documents([document_text])
            new_chunks = [text.page_content for text in texts]
            chunk_dicts = []
            for chunk in new_chunks:
                chunk_dicts.append({
                    "chunk_text": chunk,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
            all_chunk_dicts.extend(chunk_dicts)
        
        else:

            # We need to get the chunk content from the document_lines
            for i,chunk in enumerate(new_chunks):
                if i == len(new_chunks) - 1:
                    end_index = len(document_lines) - 1
                else:
                    end_index = new_chunks[i+1].start_index - 1
                chunk_text = get_chunk_text(chunk.start_index, end_index, document_lines)
                all_chunk_dicts.append({
                    "chunk_text": chunk_text,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
    
    return all_chunk_dicts