import torch

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = (
        text
        .replace("<obj>", " <obj> ")
        .replace("<subj>", " <subj> ")
        .replace("<triplet>", " <triplet> ")
        .replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .strip()
    )
    for i in range(100):
        text = text.replace(f"<extra_id_{i}>", "")
    current = 'x'
    for token in text.split():
        token = token.strip()
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject.strip() and relation.strip() and object_.strip():
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    triplets = [item for item in triplets if item["head"] and item["type"] and item["tail"]]
    return triplets


def _predict(model, tokenizer, text):
    gen_kwargs = {
        "max_length": 128,
        "early_stopping": False,
        "length_penalty": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 3,
    }
    
    model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt')
    batch_size = model_inputs['input_ids'].shape[0]
    decoder_inputs = torch.tensor([[0, 250100] for _ in range(batch_size)])
    generated_tokens = model.generate(
        model_inputs["input_ids"].to(model.device),
        attention_mask=model_inputs["attention_mask"].to(model.device),
        decoder_input_ids=decoder_inputs.to(model.device),
        **gen_kwargs,
    )
    
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    return decoded_preds
