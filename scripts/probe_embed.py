import embed_anything
import inspect

print("=== Module: embed_anything ===")
print(dir(embed_anything))

if hasattr(embed_anything, 'EmbeddingModel'):
    print("\n=== Class: EmbeddingModel ===")
    print(dir(embed_anything.EmbeddingModel))
    
    if hasattr(embed_anything.EmbeddingModel, 'from_pretrained_hf'):
        print("\n=== Method: from_pretrained_hf ===")
        # Print docstring if available
        print(embed_anything.EmbeddingModel.from_pretrained_hf.__doc__)
        
        # Try to inspect signature
        try:
            sig = inspect.signature(embed_anything.EmbeddingModel.from_pretrained_hf)
            print(f"Signature: {sig}")
        except ValueError:
            print("Signature not available (C extension)")

if hasattr(embed_anything, 'WhichModel'):
    print("\n=== Enum: WhichModel ===")
    print(dir(embed_anything.WhichModel))
    # Print values
    for x in dir(embed_anything.WhichModel):
        if not x.startswith('_'):
            print(f"  {x}")
