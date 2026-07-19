
import ollama

EMBEDDING_MODEL = "embeddinggemma"
LANGUAGE_MODEL = "gemma3:4b"


TOP_K = 3
MIN_SIMILARITY = 0.6

## Embedding

def load_chunks(path:str) -> list[str]:

    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

def build_index(chunks: list[str]) -> list[tuple[str, list[float]]]:

    # ollama.embed accepts a list, so we're embedding the whole dataset into ONE request
    print(f"Embedding {len(chunks)} chunks...")
    response = ollama.embed(model=EMBEDDING_MODEL, input=chunks)


    #We want a list of chunk/embedding pairs
    return list(zip(chunks, response["embeddings"]))


def cosine_similarity(a: list[float], b: list[float])-> float:
    dot = sum (x * y for x, y in zip(a, b)) # numerator A * B
    norm_a = sum(x * x for x in a) ** 0.5 # denominator A^2
    norm_b = sum(x * x for x in b) ** 0.5 # denominator B^2
    return dot / (norm_a * norm_b)


def retrieve(query:str,
            index: list[tuple[str, list[float]]],
            top_k:int=TOP_K) -> list[tuple[str, float]]:
    # Embedding the query with the same model as the index, embed() returns a list of embeddings, hence the [0]
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]

    #This scan is a brute-force, we're scoring every stored chunk against the query
    scored_chunks = [(chunk, cosine_similarity(query_embedding, embedding)) for chunk, embedding in index]

    #Highest Similatity First
    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)

    #Keep the top-K but LASO dropping anything under the floor for similarity
    #Because without the floor, we'd also get k chunks even when none of them are relevant
    return [(chunk, score) for chunk, score in scored_chunks if score > MIN_SIMILARITY]



#### NOW for text generation

SYSTEM_TEMPLATE = """You are a helpful assistant, answer ONLY using the context below. if the context is insufficient, say you don't know, do not invent information.

Context:
{context}"""

def chat_loop(index: list [tuple[str, list[float]]]) -> None:
    history: list[dict[str, str]] = []

    print("Ready now, go ahead and ask a question.\n")
    print("Ctrl+C to quit\n")
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        retrieved = retrieve(query, index)
        if retrieved:
            # show the scores
            print("\n Retrieved chunks:")
            for chunk, score in retrieved:
                print(f"  {chunk} (score: {score:.2f})")
            context = "\n".join(chunk for chunk, _ in retrieved)

        else:
            print("\n No relevant chunks found, using empty context.")
            context = "(no relevant context was found)"

        # Message order: System prompt (with fresh context), then the running history, then the new question

        messages = (
            [{"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)}]
            + history
            + [{"role": "user", "content": query}]
        )

        # Stream True yeals the reply token by token as it generates

        print ("\nAssistant:", end="", flush=True)
        answer = ""
        for part in ollama.chat(model=LANGUAGE_MODEL,
                                messages=messages,
                                stream=True):
            piece = part["message"]["content"]
            answer += piece
            print(piece, end="", flush=True)
        print ("\n")

        history +=[

                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},

        ]

if __name__ == "__main__":
    chunks = load_chunks("cat-facts.txt")
    index = build_index(chunks)
    chat_loop(index)

