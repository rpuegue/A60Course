from transformers import AutoModelForCausalLM, AutoTokenizer

def demo(prompt_text):
    tokenizer = AutoTokenizer.from_pretrained('TrainedModels/Model_10_epochs')
    model = AutoModelForCausalLM.from_pretrained('TrainedModels/Model_10_epochs')
    special_token = '<|endoftext|>'
    
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=len(encoded_prompt[0])+800,
        temperature=0.9,
        top_k=100,
        top_p=0.9,
        repetition_penalty=1,
        do_sample=True,
        num_return_sequences=1,
    )
    
    result = tokenizer.decode(output_sequences[0])
    result = result[:result.index(special_token)]
    print('------------------')
    print(f'Generated Output')
    print('------------------')
    print(result)