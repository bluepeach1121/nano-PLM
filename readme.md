# nano-PLM

This repo was my project to make a Masked Protein language Model with little LLM assistance. The rules are very similar to Stanford CS 336 and the model structure is comparable to modded-nanogpt.
The the LLM is meant to:
- Ensure I had the prequisite knowlegde before I started coding, by asking questions, and providing appropriate literature.
- Guide the coding direction by suggesting options to accomplish a task and previous implemetations.
- Point out mistakes and the steps to a solutions only Ive written a function or a class. 
- Point out best practices e.g. type hints, debugging etc.

The only fully LLM written code were the test codes (which are in a private colab file to ensure that my written code was working).

GPU ---> 1 A100

Dataset ----> 11000 samples from uniref50 which is about 19.5 million.

Implementation BreakDown:

- In `tokeniser.py`, I used a fixed protein residue tokenizer rather than BPE (given that the vocab size isnt much). The vocabulary contains `[PAD]`, `[MASK]`, `[CLS]`, the 20 canonical amino acids, and `X`. Ambiguous/rare residue symbols `B/Z/J/U/O/X` are mapped to `X`. No invalid characters are found.
- This idea is from the simple_gpt_train file in modded-nanogpt, where they didnt use the pytorch dataloader class, instead they made it into some sort of cache format. In `preprocess.py`, UniRef50 FASTA records are streamed one protein at a time and written into a processed cache. The token IDs are stored in `tokens.bin` as `uint8`, with `offsets.npy` and `lengths.npy` storing where each protein starts and how long it is. The entire file was downloaded (about 8 gb) but not processed. The number to be process is specified in the files (its 11000 for now). The entire thing is about 1.1 million sequences.
- `ProteinCache` memory-maps `tokens.bin` with `np.memmap`, then uses `offsets.npy` and `lengths.npy` to recover individual protein token sequences on demand.
- For training batches in `data.py`, long protein sequences are randomly cropped to fit the 512-token context window.
- For evaluation batches in `data.py`, long proteins are converted into deterministic sliding windows with stride 256. The eval masking is also deterministic because the mask RNG is seeded using the eval seed, protein index, and window start.
- In `model.py`, The transformer is a bidirectional masked protein Transformer with LLaMA-like components. Output is MLM logits of shape `[batch, seq_len, vocab_size]`.
- The run used a batch size of 512, context length 512, 3400 training steps, evaluation every 500 steps. Future improvements would mainly be around making the entire thing train faster. Right now, the frst implemetation takes about 50 minutes on 1 A100.