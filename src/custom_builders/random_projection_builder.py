import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from src.model.projection_builder_base import ProjectionBuilderBase

class RandomProjectionBuilder(ProjectionBuilderBase):
    def run(self, output_dir):
        num_layers = len(self.layers)
        n_kv = self.model.config.num_key_value_heads

        buf_H   = [[[] for _ in range(n_kv)] for _ in range(num_layers)]
        buf_Hp  = [[[] for _ in range(n_kv)] for _ in range(num_layers)]
        buf_Hn  = [[[] for _ in range(n_kv)] for _ in range(num_layers)]

        pbar = tqdm(total=self.max_samples, desc="Extracting Keys", unit="ex")
        count = 0
        for ex in self.iter_examples():
            for ctx, rel_q, ans, irr_q in self.get_triplets(ex):
                if self.chat:
                    text_H = self.tokenizer.apply_chat_template([{"role":"user","content":f"Context: {ctx}"}], tokenize=False)
                    text_Hp= self.tokenizer.apply_chat_template([{"role":"user","content":f"Question: {rel_q}\nContext: {ctx}"}], tokenize=False)
                    text_Hn= self.tokenizer.apply_chat_template([{"role":"user","content":f"Question: {irr_q}\nContext: {ctx}"}], tokenize=False)
                else:
                    text_H, text_Hp, text_Hn = f"Context: {ctx} ", f"Question: {rel_q}\nContext: {ctx}", f"Question: {irr_q}\nContext: {ctx}"

                idx_H  = self.span_token_indices(self.tokenizer, text_H,  ans)
                idx_Hp = self.span_token_indices(self.tokenizer, text_Hp, ans)
                idx_Hn = self.span_token_indices(self.tokenizer, text_Hn, ans)
                if not (idx_H and idx_Hp and idx_Hn):
                    continue
                assert len(idx_H) == len(idx_Hp) == len(idx_Hn)

                keys_H  = self.extract_keys(self.model, self.tokenizer, text_H,  idx_H,  self.layers, self.feature)
                keys_Hp = self.extract_keys(self.model, self.tokenizer, text_Hp, idx_Hp, self.layers, self.feature)
                keys_Hn = self.extract_keys(self.model, self.tokenizer, text_Hn, idx_Hn, self.layers, self.feature)

                for idx_flat, (k_H, k_Hp, k_Hn) in enumerate(zip(keys_H, keys_Hp, keys_Hn)):
                    L = idx_flat // n_kv
                    h = idx_flat %  n_kv
                    buf_H[L][h].append(k_H)
                    buf_Hp[L][h].append(k_Hp)
                    buf_Hn[L][h].append(k_Hn)
            count += 1
            pbar.update(1)
            if count >= self.max_samples:
                break
        pbar.close()

        pos_list, neg_list = [], []
        applied, skipped = [], []
        norm_diffs = np.zeros((num_layers, n_kv))

        for L in tqdm(range(num_layers), desc="Computing Random Projectors", unit="layer"):
            Pp_heads, Pn_heads = [], []
            for h in range(n_kv):
                H_mat  = torch.cat(buf_H[L][h],  0).double().to(self.device)
                Hp_mat = torch.cat(buf_Hp[L][h], 0).double().to(self.device)
                Hn_mat = torch.cat(buf_Hn[L][h], 0).double().to(self.device)

                norm_value = (torch.norm(Hp_mat - Hn_mat) / len(Hp_mat)).item()
                norm_diffs[L, h] = norm_value

                d = H_mat.size(-1)
                if norm_value < self.min_diff:
                    skipped.append((self.layers[L], h, norm_value))
                    Pp = torch.zeros((d, d), dtype=torch.float, device=self.device)
                    Pn = torch.zeros((d, d), dtype=torch.float, device=self.device)
                else:
                    applied.append((self.layers[L], h, norm_value))
                    Qp, _ = torch.linalg.qr(torch.randn(d, d, device=self.device))
                    Qn, _ = torch.linalg.qr(torch.randn(d, d, device=self.device))
                    Pp = Qp @ Qp.T
                    Pn = Qn @ Qn.T

                Pp_heads.append(Pp.float())
                Pn_heads.append(Pn.float())

            pos_list.append(torch.stack(Pp_heads, dim=0))
            neg_list.append(torch.stack(Pn_heads, dim=0))

        pos_proj = torch.stack(pos_list, dim=0)
        neg_proj = torch.stack(neg_list, dim=0)

        os.makedirs(output_dir, exist_ok=True)
        torch.save({'layers': self.layers, 'proj': pos_proj.cpu()},
                   os.path.join(output_dir, f"{self.model_path.split('/')[-1]}_pos_proj_random.pt"))
        torch.save({'layers': self.layers, 'proj': neg_proj.cpu()},
                   os.path.join(output_dir, f"{self.model_path.split('/')[-1]}_neg_proj_random.pt"))

        print("\nRandom Projection Summary:")
        print(f" ✔ Applied projection: {len(applied)}")
        print(f" ✖ Skipped (identity): {len(skipped)}")
        print(f"Saved positive projectors to {output_dir}, {tuple(pos_proj.shape)}")
        print(f"Saved negative projectors to {output_dir}, {tuple(neg_proj.shape)}")


    def iter_examples(self):
        with open(self.data_path) as f:
            for i, line in enumerate(f):
                if i >= self.max_samples:
                    break
                yield json.loads(line)

    def get_triplets(self, ex: dict) -> list[tuple[str,str,str,str]]:
        return [
            (ex['context_1'], ex['question_1'], ex['answer_1'], ex['question_2']),
            (ex['context_2'], ex['question_2'], ex['answer_2'], ex['question_1'])
        ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--layers', default='all')
    parser.add_argument('--top_pct', type=float, default=0.9)
    parser.add_argument('--feature', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--min_diff', type=float, default=2)
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    builder = RandomProjectionBuilder(
        model_path=args.model,
        data_path=args.data,
        layers=args.layers,
        top_pct=args.top_pct,
        feature=args.feature,
        max_samples=args.max_samples,
        min_diff=args.min_diff,
        chat=args.chat
    )
    builder.run(args.output_dir)