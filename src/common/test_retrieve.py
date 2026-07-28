from src.common.retrieve import retrieve

query = "What is family planning?"

nodes = retrieve(query)

print(f"Retrieved {len(nodes)} chunks\n")

for i, node in enumerate(nodes, start=1):
    print("=" * 60)
    print(f"Chunk {i}")
    print("=" * 60)
    print(node.text[:500])
    print()
    