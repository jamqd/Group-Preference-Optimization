

from transformers import PreTrainedTokenizerBase
from data.constants import ALPHABET
import random


from typing import Any, Dict, List, Optional, Union
from data.utils import get_alpaca_prompt, get_options_str
from dataclasses import dataclass
import torch

@dataclass
class RMAnthropicDataCollator:
    tokenizer: PreTrainedTokenizerBase
    prompt_format: str = "alpaca"
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        tokenized_data = {
            "input_ids_chosen" : [],
            "attention_mask_chosen" : [],
            "input_ids_rejected" : [],
            "attention_mask_rejected" : [],
        }

        processed_examples = []
        for ex in examples:
            processed_ex = self.process_example(ex["question"], 
                                                ex["selections"], 
                                                ex["options"])
            processed_examples.extend(processed_ex)

            # print(processed_ex)
            # tokenized_ex = self.tokenizer(processed_ex, 
            #                   padding=True, 
            #                   truncation=True, 
            #                   return_tensors="pt")
            
            # tokenized_data["input_ids_chosen"].append(tokenized_ex["input_ids"][0])
            # tokenized_data["attention_mask_chosen"].append(tokenized_ex["attention_mask"][0])

            # tokenized_data["input_ids_rejected"].append(tokenized_ex["input_ids"][1])
            # tokenized_data["attention_mask_rejected"].append(tokenized_ex["attention_mask"][1])

            
            
            # processed_examples.append(processed_ex)

            
        tokenized_data = self.tokenizer(processed_examples, 
                              padding=True, 
                              truncation=True, 
                              return_tensors="pt")
        
        chosen_idx = torch.arange(0, len(tokenized_data["input_ids"]), 2)
        tokenized_data["input_ids_chosen"] = (tokenized_data["input_ids"][chosen_idx])
        tokenized_data["attention_mask_chosen"] = (tokenized_data["attention_mask"][chosen_idx])
        
        rejected_index = torch.arange(1, len(tokenized_data["input_ids"]), 2)
        tokenized_data["input_ids_rejected"] = (tokenized_data["input_ids"][rejected_index])
        tokenized_data["attention_mask_rejected"] = (tokenized_data["attention_mask"][rejected_index])

        # tokenized_data["labels"] = tokenized_data["input_ids"].clone()
       
        return tokenized_data
        
    
    def process_example(self, question, selections, options):
        sampled_responses = random.choices(
            range(len(options)), 
            weights=selections,
            k=2
        )

        if selections[sampled_responses[0]] > selections[sampled_responses[1]]:
            chosen_response = ALPHABET[sampled_responses[0]]
            rejected_response = ALPHABET[sampled_responses[1]]
        else:
            chosen_response = ALPHABET[sampled_responses[1]]
            rejected_response = ALPHABET[sampled_responses[0]]


        instruction = "Answer the following question by picking from the \
            given options"
        input_text = "{question}\n\nOptions:{options}".format(
            question=question, options=get_options_str(options))

        prompt = get_alpaca_prompt(instruction=instruction, 
                                   input_text=input_text)

        chosen_seq = prompt + chosen_response
        rejected_seq = prompt + rejected_response
        return [chosen_seq, rejected_seq]