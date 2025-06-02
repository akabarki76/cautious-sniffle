from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize the tokenizer with BPE model
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# Train the tokenizer with special tokens
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["path/to/your/dataset.txt"], trainer=trainer)

# Save the trained tokenizer
tokenizer.save("tokenizer.json")
