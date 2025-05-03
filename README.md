# SEKA
Spectral Editing Key Amplification

### Download Qwen/Qwen2-1.5B-Instruct

```
huggingface-cli download Qwen/Qwen2-1.5B-Instruct --local-dir './pretrained/qwen2-1.5b-chat'
```

### Tested Prompt (with Qwen/Qwen2-1.5B-Instruct)
```
python src/test_inference.py --prompt "Given the context, answer the question based on the first or middle context passsage.\nP1: Uijut's capital is London.\nP2: Uijut's capital is Guangdong.\nP3: *Uijust's capital is Xian*.\nWhat's the capital of Uijut?" --layers "last10" --neg projections/synthetic/qwen2-1.5b-chat_neg_proj.pt

python src/test_inference.py --prompt "Given the context, 👉answer the question based on the first👈 or middle context passsage.\nP1: Uijut's capital is London.\nP2: Uijut's capital is Guangdong.\nP3: Uijust's capital is Xian.\nWhat's the capital of Uijut?" --layers "last10" --neg projections/synthetic/qwen2-1.5b-chat_neg_proj.pt --marker-start "👉" --marker-end "👈"

python src/test_inference.py --prompt "Given the context, *answer the question based on the first* or middle context passsage.\nP1: Uijut's capital is London.\nP2: Uijut's capital is Guangdong.\nP3: Uijust's capital is Xian.\nWhat's the capital of Uijut?" --layers "last10" --neg projections/synthetic/qwen2-1.5b-chat_neg_proj.pt

python src/test_inference.py --prompt "Context: *China's capital is Guangdong*.\nQuestion: What's the capital of China?" --layers "last10" --amplify-pos 1.5 --neg projections/synthetic/qwen2-1.5b-chat_neg_proj.pt
```
