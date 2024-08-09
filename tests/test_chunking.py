import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from dsParse.dsparse.chunking import get_target_num_chunks, check_for_fallback_chunking_usage, get_chunk_text


class TestGetTargetNumChunks(unittest.TestCase):

    def test__single_chunk(self):
        # Create a random string less than the max characters
        max_characters = 200
        text = "This is a random string"
        min_num_chunks, max_num_chunks = get_target_num_chunks(text, max_characters)
        assert min_num_chunks == 1
        assert max_num_chunks == 2
    
    def test__multiple_chunks(self):
        # Create a random string more than the max characters
        max_characters = 20
        text = "This is a random string of a lot of characters" # 46 characters, so should be split into 3 chunks
        min_num_chunks, max_num_chunks = get_target_num_chunks(text, max_characters)
        assert min_num_chunks == 2
        assert max_num_chunks == 4


class TestCheckForFallbackChunkingUsage(unittest.TestCase):

    def test__fallback_chunking_usage_true(self):
        min_num_chunks = 1
        max_num_chunks = 2
        chunks = [
            {"start": 0, "end": 1},
            {"start": 2, "end": 3},
            {"start": 4, "end": 5},
            {"start": 6, "end": 7},
            {"start": 8, "end": 9},
            {"start": 10, "end": 11},
        ]
        result = check_for_fallback_chunking_usage(min_num_chunks, max_num_chunks, chunks)
        assert result == True

    def test__fallback_chunking_usage_false(self):
        min_num_chunks = 2
        max_num_chunks = 4
        chunks = [
            {"start": 0, "end": 1},
            {"start": 2, "end": 3},
            {"start": 4, "end": 5},
        ]
        result = check_for_fallback_chunking_usage(min_num_chunks, max_num_chunks, chunks)
        assert result == False


class TestGetChunkText(unittest.TestCase):

    def test__get_chunk_text(self):
        document_lines = [
            "This is a random string of text.",
            "More test text.",
            "Even more test text.",
            "Text that should be ignored."
        ]
        result = get_chunk_text(0, 2, document_lines)
        expected_result = "This is a random string of text.\nMore test text.\nEven more test text."
        # We have to strip the new line character from the end of the result
        assert result.rstrip('\n') == expected_result


# Run all tests
if __name__ == '__main__':
    unittest.main()