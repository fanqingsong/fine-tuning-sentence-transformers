import csv
import os
from pathlib import Path

import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import SentenceEvaluator

from device import resolve_device


class LossTracker:
    def __init__(self):
        self._values = []

    def add(self, value):
        self._values.append(value)

    def flush_mean(self):
        if not self._values:
            return None
        mean = sum(self._values) / len(self._values)
        self._values.clear()
        return mean


class TrackedTripletLoss(losses.TripletLoss):
    def __init__(self, model, tracker):
        super().__init__(model=model)
        self.tracker = tracker

    def forward(self, sentence_features, labels):
        loss = super().forward(sentence_features, labels)
        self.tracker.add(float(loss.detach().cpu().item()))
        return loss


class TripletMarginEvaluator(SentenceEvaluator):
    """Hard-triplet metrics: accuracy plus how far positives/negatives are apart."""

    def __init__(self, examples, name="triplet_margin", loss_tracker=None):
        self.anchors = [example.texts[0] for example in examples]
        self.positives = [example.texts[1] for example in examples]
        self.negatives = [example.texts[2] for example in examples]
        self.name = name
        self.loss_tracker = loss_tracker
        self.csv_headers = [
            "epoch",
            "steps",
            "accuracy_cosine",
            "mean_sim_positive",
            "mean_sim_negative",
            "mean_margin",
            "mean_train_loss",
        ]

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        embeddings = model.encode(
            self.anchors + self.positives + self.negatives,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        n = len(self.anchors)
        anchor, positive, negative = embeddings[:n], embeddings[n : 2 * n], embeddings[2 * n :]
        sim_pos = F.cosine_similarity(anchor, positive)
        sim_neg = F.cosine_similarity(anchor, negative)
        margin = sim_pos - sim_neg

        accuracy = float((sim_pos > sim_neg).float().mean().cpu())
        mean_pos = float(sim_pos.mean().cpu())
        mean_neg = float(sim_neg.mean().cpu())
        mean_margin = float(margin.mean().cpu())
        mean_loss = self.loss_tracker.flush_mean() if self.loss_tracker else None
        loss_csv = "" if mean_loss is None else f"{mean_loss:.6f}"

        print(
            f"TripletMarginEvaluator ({self.name}) epoch={epoch} steps={steps}: "
            f"accuracy={accuracy:.4f}  sim_pos={mean_pos:.4f}  "
            f"sim_neg={mean_neg:.4f}  margin={mean_margin:.4f}  "
            f"train_loss={loss_csv or 'n/a'}"
        )

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, f"{self.name}_results.csv")
            write_header = not os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as handle:
                writer = csv.writer(handle)
                if write_header:
                    writer.writerow(self.csv_headers)
                writer.writerow(
                    [epoch, steps, accuracy, mean_pos, mean_neg, mean_margin, loss_csv]
                )

        return mean_margin


device = resolve_device()
model = SentenceTransformer("all-mpnet-base-v2", device=device)

# Triplets: [anchor, positive (same intent), negative (different intent)]
train_examples = [
    InputExample(texts=["I can't log in to my account.", "Unable to access my account.", "I need help with the payment process."]),
    InputExample(texts=["I can't log in to my account.", "Login keeps failing on the website.", "Where is my package?"]),
    InputExample(texts=["Unable to access my account.", "The sign-in page shows an error.", "How do I change my shipping address?"]),
    InputExample(texts=["How do I reset my password?", "I forgot my password. What do I do?", "How can I upgrade my account?"]),
    InputExample(texts=["How do I reset my password?", "Send me a password recovery email.", "My order hasn't arrived yet."]),
    InputExample(texts=["I forgot my password. What do I do?", "How can I change my password?", "I want a refund for my order."]),
    InputExample(texts=["How can I change my password?", "What is the process for password change?", "My package is delayed."]),
    InputExample(texts=["How can I change my password?", "I want to update my login credentials.", "How do I pay with a credit card?"]),
    InputExample(texts=["My order hasn't arrived yet.", "I haven't received my package.", "I can't find the logout button."]),
    InputExample(texts=["My order hasn't arrived yet.", "The delivery is late.", "I can't log in to my account."]),
    InputExample(texts=["I haven't received my package.", "Where is my shipment?", "How do I reset my password?"]),
    InputExample(texts=["My package is delayed.", "Tracking still says in transit.", "Unable to access my account."]),
    InputExample(texts=["I need help with the payment process.", "The checkout page rejects my card.", "I forgot my password. What do I do?"]),
    InputExample(texts=["How do I pay with a credit card?", "Can I use a debit card at checkout?", "I haven't received my package."]),
    InputExample(texts=["I want a refund for my order.", "Please cancel this purchase and refund me.", "How can I change my password?"]),
    InputExample(texts=["How can I upgrade my account?", "I want to switch to the premium plan.", "My order hasn't arrived yet."]),
    InputExample(texts=["How do I change my shipping address?", "Update the delivery address for my order.", "I can't log in to my account."]),
    InputExample(texts=["I can't find the logout button.", "How do I sign out of my account?", "I need help with the payment process."]),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

# Near-miss negatives (related intents) plus held-out wording so binary
# accuracy is less likely to sit at 1.0 from step 0.
evaluator_triplets = [
    InputExample(texts=["Can't sign into my profile.", "My account login is broken.", "I need a password recovery link."]),
    InputExample(texts=["I want to update my password.", "Help me change the password on file.", "Send a reset email, I forgot it."]),
    InputExample(texts=["The parcel still hasn't shown up.", "Still waiting for my delivery.", "Please update where it should be shipped."]),
    InputExample(texts=["Checkout won't take my card.", "Payment failed at the last step.", "I need this order refunded."]),
    InputExample(texts=["Where is the sign-out option?", "Log me out of the website.", "I can't log in to my account."]),
    InputExample(texts=["Move me to the paid plan.", "Upgrade this account to premium.", "Reset my login password."]),
    InputExample(texts=["Tracking still says in transit.", "The delivery is late.", "Unable to access my account."]),
    InputExample(texts=["How do I pay with a debit card?", "Can I use a credit card at checkout?", "Please cancel this purchase and refund me."]),
]

loss_tracker = LossTracker()
train_loss = TrackedTripletLoss(model=model, tracker=loss_tracker)
evaluator = TripletMarginEvaluator(
    evaluator_triplets, name="triplet_margin", loss_tracker=loss_tracker
)

epochs = 8
# 1 step per batch. Keep warmup well below the total number of updates,
# otherwise the learning rate stays near zero for the whole run.
steps_per_epoch = max(1, len(train_dataloader))
warmup_steps = max(1, int(0.1 * steps_per_epoch * epochs))

print(f"Training examples: {len(train_examples)}")
print(f"Eval triplets: {len(evaluator_triplets)}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Epochs: {epochs}")
print(f"Warmup steps: {warmup_steps}")

output_path = "tuned_models/fine_tuned_model"
eval_csv = Path(output_path) / "triplet_margin_results.csv"
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    evaluator=evaluator,
    # 0 = once per epoch (steps=-1). Matching steps_per_epoch would duplicate rows.
    evaluation_steps=0,
    optimizer_params={"lr": 2e-5},
    output_path=output_path,
    save_best_model=True,
    show_progress_bar=True,
)

print(f"Saved fine-tuned model to {output_path}")
print(f"Evaluator CSV: {eval_csv}")
