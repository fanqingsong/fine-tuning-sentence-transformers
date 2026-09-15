"""Compare original vs fine-tuned models on input/output pairs.

Input is the query (e.g. a user utterance). Output is the text you expect
to match it (paraphrase, FAQ answer, or candidate document).

Examples:
  python 03_compare_input_output.py \\
      --input "I can't log in to my account." \\
      --output "Unable to access my account."

  python 03_compare_input_output.py \\
      --input "How do I reset my password?" \\
      --output "I forgot my password. What do I do?" \\
      --input "How do I reset my password?" \\
      --output "My order hasn't arrived yet."

  python 03_compare_input_output.py --rank \\
      --input "I can't log in to my account." \\
      --output "Unable to access my account."
"""

import argparse

from sentence_transformers import SentenceTransformer, util

ORIGINAL_MODEL_NAME = "all-mpnet-base-v2"
FINE_TUNED_MODEL_PATH = "tuned_models/fine_tuned_model"

# Candidate pool used by --rank (same domain as 01_tuning_your_model.py).
CANDIDATES = [
    "Unable to access my account.",
    "Login keeps failing on the website.",
    "The sign-in page shows an error.",
    "I forgot my password. What do I do?",
    "Send me a password recovery email.",
    "How can I change my password?",
    "What is the process for password change?",
    "I want to update my login credentials.",
    "I haven't received my package.",
    "The delivery is late.",
    "Where is my shipment?",
    "Tracking still says in transit.",
    "The checkout page rejects my card.",
    "Can I use a debit card at checkout?",
    "Please cancel this purchase and refund me.",
    "I want to switch to the premium plan.",
    "Update the delivery address for my order.",
    "How do I sign out of my account?",
    "I need help with the payment process.",
    "My order hasn't arrived yet.",
]


def cosine(model, query, document):
    embeddings = model.encode([query, document], convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()


def rank(model, query, candidates, top_k):
    query_emb = model.encode(query, convert_to_tensor=True)
    cand_emb = model.encode(candidates, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, cand_emb)[0]
    top = scores.topk(min(top_k, len(candidates)))
    return [
        (candidates[idx], scores[idx].item())
        for idx in top.indices.tolist()
    ]


def rank_of(model, query, document, candidates):
    pool = list(dict.fromkeys([*candidates, document]))
    query_emb = model.encode(query, convert_to_tensor=True)
    pool_emb = model.encode(pool, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, pool_emb)[0]
    order = scores.argsort(descending=True).tolist()
    doc_index = pool.index(document)
    position = order.index(doc_index) + 1
    return position, scores[doc_index].item(), len(pool)


def print_pair_table(original_model, fine_tuned_model, pairs):
    print("=== Input vs output cosine similarity ===")
    print(
        f"{'#':<3} {'original':>10} {'fine-tuned':>12} {'delta':>10}  input -> output"
    )
    for i, (query, document) in enumerate(pairs, start=1):
        before = cosine(original_model, query, document)
        after = cosine(fine_tuned_model, query, document)
        print(
            f"{i:<3} {before:>10.4f} {after:>12.4f} {after - before:>+10.4f}  "
            f"{query!r} -> {document!r}"
        )


def print_rank_section(original_model, fine_tuned_model, query, expected, top_k):
    print(f"\n=== Retrieval ranking for input ===")
    print(f"input: {query!r}")
    if expected:
        print(f"expected output: {expected!r}")
        before_pos, before_score, n = rank_of(
            original_model, query, expected, CANDIDATES
        )
        after_pos, after_score, _ = rank_of(
            fine_tuned_model, query, expected, CANDIDATES
        )
        print(
            f"expected rank: original #{before_pos}/{n} (sim={before_score:.4f})  "
            f"fine-tuned #{after_pos}/{n} (sim={after_score:.4f})  "
            f"rank_delta={before_pos - after_pos:+d}"
        )

    orig_hits = rank(original_model, query, CANDIDATES, top_k)
    ft_hits = rank(fine_tuned_model, query, CANDIDATES, top_k)

    print(f"\n{'rank':<6} {'original':<48} {'sim':>8}    {'fine-tuned':<48} {'sim':>8}")
    for i in range(top_k):
        o_text, o_score = orig_hits[i] if i < len(orig_hits) else ("", float("nan"))
        f_text, f_score = ft_hits[i] if i < len(ft_hits) else ("", float("nan"))
        o_mark = " *" if expected and o_text == expected else ""
        f_mark = " *" if expected and f_text == expected else ""
        print(
            f"{i + 1:<6} {o_text[:48]:<48} {o_score:>8.4f}{o_mark:<2}  "
            f"{f_text[:48]:<48} {f_score:>8.4f}{f_mark}"
        )
    if expected:
        print("(* expected output)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare fine-tuning effect on input/output pairs."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Query text. Repeat with --output for multiple pairs.",
    )
    parser.add_argument(
        "--output",
        action="append",
        default=[],
        help="Expected matching text for the corresponding --input.",
    )
    parser.add_argument(
        "--rank",
        action="store_true",
        help="Also rank candidates for each input (retrieval-style check).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many ranked candidates to print with --rank (default: 5).",
    )
    parser.add_argument(
        "--original-model",
        default=ORIGINAL_MODEL_NAME,
        help=f"Base model name or path (default: {ORIGINAL_MODEL_NAME}).",
    )
    parser.add_argument(
        "--fine-tuned-model",
        default=FINE_TUNED_MODEL_PATH,
        help=f"Fine-tuned model path (default: {FINE_TUNED_MODEL_PATH}).",
    )
    return parser.parse_args()


def default_pairs():
    return [
        (
            "I can't log in to my account.",
            "Unable to access my account.",
        ),
        (
            "How do I reset my password?",
            "I forgot my password. What do I do?",
        ),
        (
            "My order hasn't arrived yet.",
            "I haven't received my package.",
        ),
        (
            "I can't log in to my account.",
            "My order hasn't arrived yet.",
        ),
        (
            "How do I reset my password?",
            "Where is my shipment?",
        ),
    ]


def main():
    args = parse_args()

    if args.input and args.output and len(args.input) != len(args.output):
        raise SystemExit(
            f"Got {len(args.input)} --input and {len(args.output)} --output; "
            "they must be paired 1:1 (or omit --output and use --rank)."
        )
    if args.output and not args.input:
        raise SystemExit("--output requires at least one --input.")
    if args.input and not args.output and not args.rank:
        raise SystemExit("Provide --output for each --input, or pass --rank.")

    print(f"Loading original model: {args.original_model}")
    original_model = SentenceTransformer(args.original_model)
    print(f"Loading fine-tuned model: {args.fine_tuned_model}")
    fine_tuned_model = SentenceTransformer(args.fine_tuned_model)

    if args.input and args.output:
        pairs = list(zip(args.input, args.output))
    elif args.input:
        pairs = []
    else:
        print("No --input/--output given; using built-in demo pairs.\n")
        pairs = default_pairs()

    if pairs:
        print_pair_table(original_model, fine_tuned_model, pairs)

    if args.rank:
        queries = args.input or [query for query, _ in default_pairs()]
        expected_by_query = {}
        for query, document in pairs:
            expected_by_query.setdefault(query, document)
        seen = set()
        for query in queries:
            if query in seen:
                continue
            seen.add(query)
            print_rank_section(
                original_model,
                fine_tuned_model,
                query,
                expected_by_query.get(query),
                args.top_k,
            )


if __name__ == "__main__":
    main()
