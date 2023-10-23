from enum import Enum

GROUP_NAMES = ['Northeast', 'South', 'Female', 'Male', 'College graduate/some postgrad',
             'Less than high school', 'Atheist', 'Hindu', 'Jewish', 'Muslim',
             'Protestant', 'Democrat', 'Republican', '$100,000 or more',
             'Less than $30,000', 'Conservative', 'Liberal', 'Moderate', 'Asian',
             'Black', 'Hispanic', 'White']

COUNTRIES = [
    'Nigeria', 
    'Egypt', 
    'India (Current national sample)', 
    'China', 
    'Japan', 
    'Germany', 
    'France', 
    'Spain', 
    'United States', 
    'Canada', 
    'Brazil', 
    'Argentina', 
    'Australia', 
    'New Zealand'
]

ALL_COUNTRIES = [
    'Albania',
    'Andorra',
    'Angola',
    'Angola (Non-national sample)',
    'Argentina',
    'Armenia',
    'Australia',
    'Austria',
    'Azerbaijan',
    'Bangladesh',
    'Bangladesh (Non-national sample)',
    'Belarus',
    'Belgium',
    'Bolivia',
    'Bolivia (Non-national sample)',
    'Bosnia Herzegovina',
    'Brazil',
    'Brazil (Non-national sample)',
    'Britain',
    'Bulgaria',
    'Burkina Faso',
    'Canada',
    'Chile',
    'China',
    'China (Non-national sample)',
    'Colombia',
    'Colombia (Non-national sample)',
    'Croatia',
    'Cyprus',
    'Czech Rep.',
    'Czechia',
    'Denmark',
    'Ecuador',
    'Egypt',
    'Egypt (Non-national sample)',
    'El Salvador',
    'Estonia',
    'Ethiopia',
    'Ethiopia (Non-national sample)',
    'Finland',
    'France',
    'Georgia',
    'Germany',
    'Ghana',
    'Great Britain',
    'Greece',
    'Guatemala',
    'Guatemala (Non-national sample)',
    'Honduras',
    'Honduras (Non-national sample)',
    'Hong Kong SAR',
    'Hungary',
    'Iceland',
    'India (Current national sample)',
    'India (Non-national sample)',
    'India (Old national sample)',
    'Indonesia',
    'Indonesia (Non-national sample)',
    'Iran',
    'Iraq',
    'Israel',
    'Italy',
    'Ivory Coast',
    'Ivory Coast (Non-national sample)',
    'Japan',
    'Jordan',
    'Jordan (Non-national sample)',
    'Kazakhstan',
    'Kenya',
    'Kuwait',
    'Kyrgyzstan',
    'Latvia',
    'Lebanon',
    'Libya',
    'Lithuania',
    'Macau SAR',
    'Malaysia',
    'Maldives',
    'Mali',
    'Mali (Non-national sample)',
    'Mexico',
    'Mongolia',
    'Montenegro',
    'Morocco',
    'Morocco (Non-national sample)',
    'Myanmar',
    'Netherlands',
    'New Zealand',
    'Nicaragua',
    'Nigeria',
    'Nigeria (Non-national sample)',
    'North Macedonia',
    'Northern Ireland',
    'Norway',
    'Pakistan',
    'Pakistan (Non-national sample)',
    'Palest. ter.',
    'Peru',
    'Philippines',
    'Philippines (Non-national sample)',
    'Poland',
    'Poland (Non-national sample)',
    'Portugal',
    'Puerto Rico',
    'Romania',
    'Russia',
    'Russia (Non-national sample)',
    'S. Africa',
    'S. Africa (Non-national sample)',
    'S. Korea',
    'Senegal',
    'Senegal (Non-national sample)',
    'Serbia',
    'Singapore',
    'Slovakia',
    'Slovenia',
    'South Korea',
    'Spain',
    'Sweden',
    'Switzerland',
    'Taiwan',
    'Taiwan ROC',
    'Tajikistan',
    'Tanzania',
    'Tanzania (Non-national sample)',
    'Thailand',
    'Tunisia',
    'Turkey',
    'Uganda',
    'Ukraine',
    'United States',
    'Uruguay',
    'Uzbekistan',
    'Venezuela',
    'Venezuela (Non-national sample)',
    'Vietnam',
    'Vietnam (Non-national sample)',
    'Zimbabwe'
]

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

ALPACA_PROMPT_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


ALPHABET = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
    "L", "M", "N", "O","P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z"
]

LLAMA2_PROMPT  = """<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>
{{ user_message }} [/INST] """

LLAMA2_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""



class TaskType(str, Enum):
    SFT = "SFT"
    RM = "RM"
    RL = "RL"


class PromptFormatType(str, Enum):
    ALPACA = "alpaca"
    LLAMA2 = "llama2"