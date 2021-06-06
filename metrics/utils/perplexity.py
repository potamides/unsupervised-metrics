from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import tensor

def lm_perplexity(hyps, device):
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")        

    scores = list()
    model.eval()
    for hyp in hyps:
        tokenize_input = tokenizer.tokenize(hyp)
        
        if len(tokenize_input) <= 1:
            scores.append(0)
        else:
            if len(tokenize_input) > 1024:
                tokenize_input = tokenize_input[:1024]
                
            input_ids = tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
            score = model(input_ids, labels=input_ids)[0]
            scores.append(-score.item())

    return scores
