import torch
class proteintokenizer:
    def __init__(self):
        super().__init__()
        AA_set = sorted(['A','G','V','L','I','P','Y','F','W','S','T','C','M','N','Q','D','E','K','R','H','-','!','&','?','|','b'])
        self.stoi = {AA_set[i]: i for i in range(len(AA_set))} 
        self.itos = {i: AA_set[i] for i in range(len(AA_set))}
        self.pad_id = self.stoi['b']
        self.cls_id = self.stoi['&']
        self.sep_id = self.stoi['|']
        self.idx_to_char = [self.itos[i] for i in range(len(AA_set))]
        self.idx_tensor = torch.tensor(list(self.stoi.values()), dtype=torch.long)
        self.char_tensor = torch.tensor([ord(c) for c in self.idx_to_char], dtype=torch.int8)

    def encode(self,seq):
        return torch.tensor([self.stoi.get(s) for s in seq])
    def decode(self, tensor):

        is_batch = tensor.dim() == 2
        if not is_batch:
            tensor = tensor.unsqueeze(0)  

        batch_size, seq_len = tensor.shape
        results = []

        for i in range(batch_size):
            seq_ids = tensor[i]
            mask = seq_ids != self.pad_id
            valid_ids = seq_ids[mask]

            protein_seq = ''.join([self.idx_to_char[id] for id in valid_ids.tolist()])
            results.append(protein_seq)

        return results if is_batch else results[0]
    def bert_encode(self,seq):
        seq = [self.stoi['&']] +seq + [self.stoi['|']]
        return torch.tensor([self.stoi.get(s) for s in seq])

    def peptokendecode_pre(self):
        ressymb_to_resindex = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,}
        flowtoaa ={}
        for k,v in ressymb_to_resindex.items():
            flowtoaa[v] = self.stoi[k]
        max_src_index = max(ressymb_to_resindex.values())
        self.mapping_tensor = torch.zeros(max_src_index + 1, dtype=torch.long)
        for src_idx, tgt_idx in flowtoaa.items():
            self.mapping_tensor[src_idx] = tgt_idx
    def peptokendecode(self,aa):
        if isinstance(self.mapping_tensor,torch.Tensor):
            continue
        else:
            self.peptokendecode_pre()
        return self.mapping_tensor[aa]