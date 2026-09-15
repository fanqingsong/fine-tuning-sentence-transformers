from sentence_transformers import SentenceTransformer, util

original_model = SentenceTransformer("all-mpnet-base-v2")
fine_tuned_model = SentenceTransformer("tuned_models/fine_tuned_model")

# Same-sentence cosine between the two models is a weak signal.
# Fine-tuning often keeps a sentence close to its original vector while
# changing *relative* distances between related and unrelated intents.
print("=== Same sentence: original vs fine-tuned (can stay near 1.0) ===")
probe_sentences = [
    "I can't log in to my account.",
    "How do I reset my password?",
    "My order hasn't arrived yet.",
    "How can I change my password?",
]
for sentence in probe_sentences:
    original_embedding = original_model.encode(sentence, convert_to_tensor=True)
    fine_tuned_embedding = fine_tuned_model.encode(sentence, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(original_embedding, fine_tuned_embedding)
    max_abs_diff = (original_embedding - fine_tuned_embedding).abs().max().item()
    print(
        f"'{sentence}'  cosine={cosine_similarity.item():.6f}  "
        f"max_abs_diff={max_abs_diff:.6e}"
    )

pairs = [
    ("related / login", "I can't log in to my account.", "Unable to access my account."),
    ("related / password", "How do I reset my password?", "I forgot my password. What do I do?"),
    ("related / order", "My order hasn't arrived yet.", "I haven't received my package."),
    ("unrelated / login vs order", "I can't log in to my account.", "My order hasn't arrived yet."),
    ("unrelated / password vs order", "How can I change my password?", "My package is delayed."),
    ("unrelated / login vs payment", "Unable to access my account.", "I need help with the payment process."),
]


def pair_cosine(model, a, b):
    embeddings = model.encode([a, b], convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()


print("\n=== Pair cosine similarity (this is what fine-tuning should change) ===")
print(f"{'pair':<28} {'original':>10} {'fine-tuned':>12} {'delta':>10}")
for label, a, b in pairs:
    before = pair_cosine(original_model, a, b)
    after = pair_cosine(fine_tuned_model, a, b)
    print(f"{label:<28} {before:>10.4f} {after:>12.4f} {after - before:>+10.4f}")
