from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
text_encoder = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

batch_sentences = ["fk u", "i love u"]
tokenized_text = tokenizer(batch_sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(tokenized_text)

a = text_encoder.get_input_embeddings()
print(a(tokenized_text['input_ids']).shape)

text_features = text_encoder(**tokenized_text, output_hidden_states=True)

text_features[2]
for i in range(len(text_features[2])):
    print(text_features[2][i].shape)
print()