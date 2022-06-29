def preprocessing(input_text, tokenizer):
    '''
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
      - input_ids: list of token ids
      - token_type_ids: list of token type ids
      - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
    '''
    return tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=32,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt')
