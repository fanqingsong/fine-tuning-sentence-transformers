from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator


model = SentenceTransformer("all-mpnet-base-v2")

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

evaluator_triplets = [
    InputExample(texts=["How can I change my password?", "What is the process for password change?", "My package is delayed."]),
    InputExample(texts=["I can't log in to my account.", "Unable to access my account.", "I want to change my shipping address."]),
    InputExample(texts=["My order hasn't arrived yet.", "I haven't received my package.", "How do I reset my password?"]),
]
evaluator = TripletEvaluator.from_input_examples(evaluator_triplets, name="my_evaluator")

train_loss = losses.TripletLoss(model=model)

epochs = 8
# 1 step per batch. Keep warmup well below the total number of updates,
# otherwise the learning rate stays near zero for the whole run.
steps_per_epoch = max(1, len(train_dataloader))
warmup_steps = max(1, int(0.1 * steps_per_epoch * epochs))

print(f"Training examples: {len(train_examples)}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Epochs: {epochs}")
print(f"Warmup steps: {warmup_steps}")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    evaluator=evaluator,
    evaluation_steps=steps_per_epoch,
    optimizer_params={"lr": 2e-5},
)

model.save("tuned_models/fine_tuned_model")
print("Saved fine-tuned model to tuned_models/fine_tuned_model")
