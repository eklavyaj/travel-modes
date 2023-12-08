import langchain
import transformers
import torch
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain, LLMChain
import time
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CACHE_DIR = '/home/eklavya/Code/.cache'
TOKEN = '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'

def get_model(model_id):
    
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=TOKEN,
        cache_dir=CACHE_DIR
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        token=TOKEN,
        cache_dir=CACHE_DIR
    )

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        trust_remote_code=True,
        device_map={"": 0},
        max_new_tokens = 300,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=generate_text, 
                              model_kwargs={'temperature':0.00})
    
    return llm

def get_prompt():
    
    examples =  [
        {
            'input': 'The subway is delayed yet again. This city just can not run on time. Apparently there is a water leak near Times Square. Feel so angry.',
            'output':
            """
            1. Subway
            2. Negative
            3. The subway was delayed due to a water leak near Times Square.
            """
        },
        {
            'input': "It's such a beautiful day. Looks like I am going to go on a walk in Central Park.",
            'output':
            """
            1. Unknown
            2. Neutral
            3. Doesn't talk about travel modes
            """
        },
        {
            'input': "MTA has done a good job maintaining the bus service schedule this summer. I have been at work on time everyday.",
            'output':
            """
            1. Bus
            2. Positive
            3. The bus service was well maintained throughout the summer.
            """
        },
    ]
    
    prompt_template_cot = """[INST]
    Tweet: {input}

    Question:
    Only answer the following questions in order as bullet points.
        1. Select the mode of travel: Subway, Bus, Bike, Taxi, Car, Unknown
        2. Select the sentiment: Positive, Neutral, Negative
        3. Explain your reasoning behind the selected sentiment in less than 20 words.

    [/INST]auto
    Answer:
    {output}
    """

    example_prompt = PromptTemplate(template=prompt_template_cot, input_variables=['input', 'output'])
    
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="""
    Tweet: {input}

    Question:
    Only answer the following questions in order as bullet points.
        1. Select the mode of travel: Subway, Bus, Bike, Taxi, Car, Unknown
        2. Select the sentiment: Positive, Neutral, Negative
        3. Explain your reasoning behind the selected sentiment in less than 20 words.

    Answer:
    """,
        input_variables=['input']
    )
    
    return prompt

def llm_chain_fn(llm, prompt):
    return LLMChain(llm=llm, prompt=prompt)


def do_qa(answer_list):

    if len(answer_list) < 3:
        return [None, None, None]

    else:
        if answer_list[0].strip().lower() not in ['subway', 'bus', 'bike', 'taxi', 'car', 'unknown']:
            return ['Unknown', None, None]

        if answer_list[1].strip().lower() not in ['positive', 'negative', 'neutral']:
            return [answer_list[0], None, "Can't determine sentiment"]

    return answer_list


def process_answer(answer):
    possible_answers = [' '.join(x.strip().split(" ")[1:]) for x in answer.split("\n") if x.strip().startswith(tuple(["1. ", "2. ", "3. "]))]
    final_answer = possible_answers[:3]

    final_answer = do_qa(final_answer)
    return final_answer

def process_batch(results):
    processed_results = []
    for i, result in enumerate(results):
        processed_results.append([result['idx'], result['input'], process_answer(result['text'])])

    return processed_results


def get_results(llm_chain, src_path, dst_path, batch_size = 50):
    

    try:
        df = pd.read_csv(dst_path)
        print("-- Output exists -- Beginning Batch Processing")
        # i = tmp.dropna(subset=['travel_mode']).shape[0]
        # flag = True
    except:
        df = pd.read_csv(src_path)

        print("-- Read data -- Begninning Batch Processing")
        df['travel_mode'] = None
        df['sentiment'] = None
        df['reasoning'] = None
    
    tmp = df[df['travel_mode'].isna()]
    
    print(f"-- {len(tmp)} Number of Rows to be processed")
    inputs = []
    
    for i, x in zip(tmp.index, tmp['processed_txt'].values):
        inputs.append({'input': x, 'idx': i})

    inputs = np.array(inputs)
    
    times = []
    results = []
    
    i = tmp.index[0]
    while i < len(tmp.index):

        print(f" -- Starting Batch {i//batch_size + 1}")
        up = min(i + batch_size, len(tmp.index))
        
        batch_idx = tmp.index[i:up]
        batch = list(inputs[batch_idx])
        
        start = time.time()
        temp = llm_chain.batch(batch)
        times.append(time.time() - start)

        result = process_batch(temp)
        results.extend(result)
            
        i = i + batch_size
        print(f" -- Completed Batch {i//batch_size}: {times[-1]}")
        
    for res in results:
        df.iloc[res[0], -3:] = res[-1]
    df.to_csv(dst_path, index=False)
    
    
    
# ----------------------------- MISTRAL 7B ---------------------

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model_name = 'mistral_7b_instruct'

if not os.path.exists("/home/eklavya/Code/Results/" + model_name):
    os.mkdir(os.path.join("/home/eklavya/Code/Results/", model_name))
    

# ----------------------------- LLAMA2 7B ---------------------

# model_id = "meta-llama/Llama-2-7b-chat-hf"
# model_name = 'llama2_7b_chat'

# if not os.path.exists(os.path.join("../Results/", model_name)):
#     os.mkdir(os.path.join("../Results/", model_name))
        

llm = get_model(model_id=model_id)
prompt = get_prompt()
llm_chain = llm_chain_fn(llm=llm, prompt=prompt)

print("Model Initialized")

src_path1 = '/home/eklavya/Code/Data/processed_2000_2999.csv'
dst_path1 = f'/home/eklavya/Code/Results/{model_name}/results_2000_2999.csv'

src_path2 = '/home/eklavya/Code/Data/processed_3000_3999.csv'
dst_path2 = f'/home/eklavya/Code/Results/{model_name}/results_3000_3999.csv'

get_results(llm_chain=llm_chain, src_path=src_path1, dst_path=dst_path1, batch_size=100)
get_results(llm_chain=llm_chain, src_path=src_path2, dst_path=dst_path2, batch_size=50)