##### PART 1
from collections import Counter

#we define the special tokens

SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[CLS]" ]

CANONICAL_RESIDUES = list("ACDEFGHIKLMNPQRSTVWY")

TURN_INTO_X = {
    "B": "X",
    "Z": "X",
    "J": "X",
    "U": "X",
    "O": "X",
    "X": "X",
}

RESIDUE_TOKENS = CANONICAL_RESIDUES + ["X"]

VOCAB = SPECIAL_TOKENS + RESIDUE_TOKENS

token_to_id = {tok: i for i, tok in enumerate(VOCAB)}
id_to_token = {i: tok for tok, i in token_to_id.items()}

PAD_ID = token_to_id["[PAD]"]
MASK_ID = token_to_id["[MASK]"]
CLS_ID = token_to_id["[CLS]"]

#print(token_to_id)
#print(id_to_token)
#print("vocab length:::", len(VOCAB))

VALID_RAW_CHARS = set(CANONICAL_RESIDUES) | set(TURN_INTO_X.keys())
#print(VALID_RAW_CHARS)

def clean_sequence(seq):
    seq = seq.upper()
    seq ="".join(seq.split())

    invalid = sorted(set(seq) - VALID_RAW_CHARS)

    if invalid:
        raise ValueError(f"invalid sequence characters: {invalid}")

    return seq


def count_raw_residues(seqs):
    counts = Counter()
    for seq in seqs:
        clean = clean_sequence(seq)
        counts.update(clean)
    return counts

###PART 2

class ProteinTokeniser:
    def __init__(self):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.pad_id = PAD_ID
        self.mask_id = MASK_ID
        self.cls_id = CLS_ID
        self.vocab = VOCAB
        self.vocab_size = len(VOCAB)

    def encode(self, seq, add_cls=True):
        seq = clean_sequence(seq)
        #seq.split splits by whitespace
        seq_list = list(seq)

        ids = []
        if add_cls:
            ids.append(self.cls_id)

        for residue in seq_list:
            #.get(python)
            #If it exists, return the mapped value.
            #If it does not exist, return residue itself.
            mapped_residue = TURN_INTO_X.get(residue, residue)
            token_id = self.token_to_id[mapped_residue]
            ids.append(token_id)

        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            token = self.id_to_token[i]
            tokens.append(token)

        return "".join(tokens)

#tokeniser = ProteinTokeniser()
#ids = tokeniser.encode("ACDEFBZJUOX")
#print(ids)
#print(tokeniser.decode(ids))
#print(tokeniser.encode("ACD"))
#print(tokeniser.encode("ACD", add_cls=False))
