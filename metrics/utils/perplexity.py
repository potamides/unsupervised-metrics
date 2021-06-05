from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import tensor

def lm_perplexity(hyps, batch_size, device):
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")        

    scores = list()
    model.eval()
    for batch_start in range(0, len(hyps), batch_size):
        batch_hyps = hyps[batch_start:batch_start+batch_size]    
        
        tokenize_input = tokenizer.tokenize(batch_hyps[0])
        
        if len(tokenize_input) <=1:
            scores.append(0)
        else:
            if len(tokenize_input) > 1024:
                tokenize_input = tokenize_input[:1024]
                
            input_ids = tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
            score = model(input_ids, labels=input_ids)[0]
            scores.append(-score.item())
    return scores
