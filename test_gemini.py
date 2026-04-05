from clients import get_gemini


def test_generative_model():
    gemini = get_gemini()
    response, usage = gemini.generate("Reply with exactly: hello world")
    print(f"Generate OK: {response}")
    print(f"Usage: {usage}")
    assert isinstance(response, str) and len(response) > 0
    assert usage["model"] == "gemma-4-26b-a4b-it"
    assert "input_tokens" in usage
    assert "output_tokens" in usage


def test_embedding_model():
    gemini = get_gemini()
    vector = gemini.embed("This is a test sentence.")
    print(f"Embed OK: {len(vector[0])} dims, first value: {vector[0][0]:.6f}")
    assert len(vector[0]) == 1536
    assert all(isinstance(v, float) for v in vector[0])


if __name__ == "__main__":
    test_generative_model()
    test_embedding_model()
    print("All Gemini tests passed.")