### Konstantinos
1. Pre-process BookSum - keep 2000 words from text and 500 words from summary
   1. Create dataset (negative examples - oversampling)
   2. Remove stop-words
2. Two encoder model - text->encode, summary->encode, text+summary (concat) - BERT, LongT5, Longformer
4. Handle long texts

### Vasilis
1. Pre-process PubMed  - keep 2000 words from text and 500 words from summary
   1. Create dataset (negative examples - oversampling)
   2. Remove stop-words
2. Single input model - <CLS>text<SEP>summary<SEP> - BERT, GPT, LongT5, Longformer
4. Handle long texts
