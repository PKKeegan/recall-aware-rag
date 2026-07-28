from src.common.vector_store import load_vector_index

print("Loading vector index...")

index = load_vector_index()

print("Success!")
print(type(index))