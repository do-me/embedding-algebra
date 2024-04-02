# Embedding Algebra

## Background 
With the rise of Word2Vec it's reduction to the formula `King - Man + Woman = Queen` fueled common falsehoods about embedding algebra.
The idea is that you can add and substract vectors to obtain a new embedding reflecting the semantic change, like `King - Man = Royal` or `Woman + Royal = Queen`. The bigger picture is right that the vector operations reflect the semantic changes but only to a certain degree. 

Apparently there are certain exceptions to the rule and some analogies work better (=as a human would expect) than others. 
Reading this [Medium Article](https://blog.esciencecenter.nl/king-man-woman-king-9a7fd2935a85) from a couple of years ago I thought I'd give it a go with current SOTA models.

## Findings 
**tl;dr** confirmed again: `King - Man + Woman = Queen` is pretty much never true!

### General Findings
- As the Medium article claims, it's always `King` that is most similar. `Queen` doesn't even come second always! 
- What's most interesting is that negative embeddings have the biggest impact, to the word `Man` will always rank last. 
- Unsurprisingly, the instruction has a high impact on these results, like in the case of [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) where with instruction (that is not mentioned in the repo) it performs better than without.
- `King - Man` leads to `King` too, but second always comes `Royal` ✔️
- `Queen - Woman` leads to `Queen` too, but second comes `Prince` ✖️
- Averaging doesn't change much, `Woman + Royal` and `(Woman + Royal) / 2` roughly lead to the same results

### Gender Bias
I expected to see a gender bias when testing for `King + Queen`, like that `King` is more similar to the resulting embedding than `Queen` due to a bias in the training data (like more mentions of kings in our history books than queens) but apparently that doesn't hold. Instead, it **highly depends on the model**:
- `mixedbread-ai/mxbai-embed-large-v1`:
```
Cosine similarity between 'queen' and analogy vector: 0.9102759957313538
Cosine similarity between 'king'  and analogy vector: 0.909360408782959
```
- `BAAI/bge-base-en-v1.5`
```
Cosine similarity between 'king'  and analogy vector: 0.9067744016647339
Cosine similarity between 'queen' and analogy vector: 0.9067744016647339
```
So while `BAAI/bge-base-en-v1.5` takes a mathematical approach that the summed vector has the same distance to all of its summands, that's not the case for `mixedbread-ai/mxbai-embed-large-v1`.

## Scripts
See the notebook in this repo to reproduce the results with any model and any equation. I included all three, `Euclidian Distance`, `Dot Product` and `Cosine Similarity` but keep in mind that most models have a preferred distance metric (often cosine distance). These are the [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) results for `King - Man + Woman`:

```
Dot Product:
Dot product between 'king' and analogy vector: 206.02333068847656
Dot product between 'woman' and analogy vector: 179.44287109375
Dot product between 'princess' and analogy vector: 177.8961181640625
Dot product between 'queen' and analogy vector: 177.6486053466797
Dot product between 'castle' and analogy vector: 116.86325073242188
Dot product between 'prince' and analogy vector: 113.52368927001953
Dot product between 'horse' and analogy vector: 113.3372802734375
Dot product between 'person' and analogy vector: 110.6568374633789
Dot product between 'apple' and analogy vector: 107.81037139892578
Dot product between 'banana' and analogy vector: 103.31510925292969
Dot product between 'basketball' and analogy vector: 101.27586364746094
Dot product between 'clown' and analogy vector: 97.28660583496094
Dot product between 'football' and analogy vector: 96.44972229003906
Dot product between 'man' and analogy vector: 47.41835021972656

--------------------------------------------------------------------------------

Cosine Similarity:
Cosine similarity between 'king' and analogy vector: 0.7420865297317505
Cosine similarity between 'woman' and analogy vector: 0.6679535508155823
Cosine similarity between 'queen' and analogy vector: 0.6367943286895752
Cosine similarity between 'princess' and analogy vector: 0.6064033508300781
Cosine similarity between 'person' and analogy vector: 0.4240642786026001
Cosine similarity between 'castle' and analogy vector: 0.41255974769592285
Cosine similarity between 'horse' and analogy vector: 0.39906030893325806
Cosine similarity between 'prince' and analogy vector: 0.3888675272464752
Cosine similarity between 'apple' and analogy vector: 0.3804171681404114
Cosine similarity between 'banana' and analogy vector: 0.3553932309150696
Cosine similarity between 'basketball' and analogy vector: 0.3550359904766083
Cosine similarity between 'football' and analogy vector: 0.3462764620780945
Cosine similarity between 'clown' and analogy vector: 0.3232797086238861
Cosine similarity between 'man' and analogy vector: 0.17841237783432007

--------------------------------------------------------------------------------

Euclidean Distance (sorted by smallest distance, which indicates highest similarity):
Euclidean distance between 'king' and analogy vector: 12.40994930267334
Euclidean distance between 'woman' and analogy vector: 13.87999439239502
Euclidean distance between 'queen' and analogy vector: 14.593585968017578
Euclidean distance between 'princess' and analogy vector: 15.389604568481445
Euclidean distance between 'person' and analogy vector: 17.837038040161133
Euclidean distance between 'castle' and analogy vector: 18.484573364257812
Euclidean distance between 'horse' and analogy vector: 18.707866668701172
Euclidean distance between 'apple' and analogy vector: 18.974042892456055
Euclidean distance between 'prince' and analogy vector: 19.055479049682617
Euclidean distance between 'football' and analogy vector: 19.355772018432617
Euclidean distance between 'basketball' and analogy vector: 19.39596176147461
Euclidean distance between 'banana' and analogy vector: 19.529788970947266
Euclidean distance between 'clown' and analogy vector: 20.282350540161133
Euclidean distance between 'man' and analogy vector: 21.264333724975586
```

## PRs
Highly appreciated, maybe some automization would be good the create a nicely formatted markdown table to be included in this readme listing the behavior of the most used embedding models. Would this even be something for MTEB?




