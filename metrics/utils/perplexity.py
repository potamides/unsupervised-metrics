from transformers import AutoModelWithLMHead, AutoTokenizer
from torch import tensor

def lm_perplexity(hyps, device, name="gpt2"):
    # Some models need a special tokenizer, like chinese gpt2, see here:
    # https://huggingface.co/ckiplab/gpt2-base-chinese
    model_name, tokenizer_name = (name, name) if isinstance(name, str) else name

    model = AutoModelWithLMHead.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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
