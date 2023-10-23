from data.constants import ALPACA_PROMPT, ALPACA_PROMPT_NO_INPUT, ALPHABET, LLAMA2_PROMPT, LLAMA2_SYSTEM_PROMPT
import random
def get_alpaca_prompt(instruction, input_text=None):
    if input_text is None:
        return ALPACA_PROMPT_NO_INPUT.format(instruction=instruction)
    else:
        return ALPACA_PROMPT.format(instruction=instruction, input=input_text)
    

def get_options_str(options):
    option_str = ""
    for i, opt in enumerate(options):
        option_str += f"\n{ALPHABET[i]}. {opt}"
    return option_str

def get_llama2_prompt(user_message, system_prompt=None):
    prompt = LLAMA2_PROMPT
    if system_prompt is None:
        prompt = prompt.replace("{{ system_prompt }}", LLAMA2_SYSTEM_PROMPT)
    else:
        prompt = prompt.replace("{{ system_prompt }}", system_prompt)

    prompt = prompt.replace("{{ user_message }}", user_message)
    
    return prompt