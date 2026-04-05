from clients import get_gemini


def test_generative_model():
    gemini = get_gemini()
    response = gemini.generate("Reply with exactly: hello world")
    print("Generate OK:", response)
    assert isinstance(response, str) and len(response) > 0


def test_embedding_model():
    gemini = get_gemini()
    vector = gemini.embed("This is a test sentence.")
    print(f"Embed OK: {len(vector)} dims, first value: {vector[0]:.6f}")
    assert len(vector) == 3072
    assert all(isinstance(v, float) for v in vector)


if __name__ == "__main__":
    test_generative_model()
    test_embedding_model()
    print("All Gemini tests passed.")