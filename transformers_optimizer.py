from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from RecAdam import RecAdam, anneal_function


model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small') 

for i in model.named_parameters():
    print(i)
# rec = RecAdam()



# trainer = Trainer(
#     model = model,
#     optimizers = [(RecAdam,params)]
# )



# print(type(trainer))

