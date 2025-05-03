# SEKA
Spectral Editing Key Amplification

### Download Qwen2-1.5B-Instruct and Llama-3.2-1B-Instruct

```
huggingface-cli download Qwen/Qwen2-1.5B-Instruct --local-dir './pretrained/qwen2-1.5b-chat'
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir './pretrained/llama3.2-1b-chat'
```

### Tested Prompt (with Qwen/Qwen2-1.5B-Instruct)
```
python src/test_inference.py --prompt "Given the context, answer the question based on the first or middle context passsage directly.\nP1: Uijut's capital is London.\nP2: Uijut's capital is Guangdong.\nP3: *Uijust's capital is Xian*.\nWhat's the capital of Uijut?" --layers "last10" --neg projections/synthetic/qwen2-1.5b-chat_neg_proj.pt --chat
```

```
python src/test_inference.py --prompt "Given the context, 👉answer the question based on the first👈 or middle context passsage directly.\nP1: Uijut's capital is London.\nP2: Uijut's capital is Guangdong.\nP3: Uijust's capital is Xian.\nWhat's the capital of Uijut?" --layers "last10" --neg projections/synthetic/qwen2-1.5b-chat_neg_proj.pt --marker-start "👉" --marker-end "👈" --chat
```

```
python src/test_inference.py --prompt "Given the context, *answer the question based on the first* or middle context passsage directly.\nP1: Uijut's capital is London.\nP2: Uijut's capital is Guangdong.\nP3: Uijust's capital is Xian.\nWhat's the capital of Uijut?" --layers "last10" --neg projections/synthetic/qwen2-1.5b-chat_neg_proj.pt --chat
```

```
python src/test_inference.py --prompt "Context: *China's capital is Guangdong*.\nQuestion: What's the capital of China?" --layers "last10" --neg projections/synthetic/qwen2-1.5b-chat_neg_proj.pt --chat
```

### Tested Prompt (with Llama-3.2-1B-Instruct)
```
python src/test_inference.py --prompt "Given the context, answer the question directly. P1: Uijut's capital is London. P2: Uijut's capital is Guangdong. P3: *Uijust's capital is Xian*. Question: What's the capital of Uijut?" --layers "last7" --neg projections/synthetic/llama3.2-1b-chat_neg_proj.pt --pos projections/synthetic/llama3.2-1b-chat_pos_proj.pt --model pretrained/llama3.2-1b-chat --chat --amplify-pos 1.0 --amplify-neg 0.0
```

```
python src/test_inference.py --prompt "Given the context, 👉answer the question based on the first👈 or middle context passsage directly. P1: Uijut's capital is London. P2: Uijut's capital is Guangdong. P3: Uijust's capital is Xian. Question: What's the capital of Uijut?" --layers "last7" --neg projections/synthetic/llama3.2-1b-chat_neg_proj.pt --pos projections/synthetic/llama3.2-1b-chat_pos_proj.pt --marker-start "👉" --marker-end "👈" --model pretrained/llama3.2-1b-chat --chat --amplify-pos 1.0 --amplify-neg 1.0
```

```
python src/test_inference.py --prompt "Given the context, *answer the question based on the first* or middle context passsage directly. P1: Uijut's capital is London. P2: Uijut's capital is Guangdong. P3: Uijust's capital is Xian. Question: What's the capital of Uijut?" --layers "last7" --neg projections/synthetic/llama3.2-1b-chat_neg_proj.pt --pos projections/synthetic/llama3.2-1b-chat_pos_proj.pt --model pretrained/llama3.2-1b-chat --chat --amplify-pos 1.0 --amplify-neg 1.0
```

```
python src/test_inference.py --prompt "Context: *China's capital is Guangdong*.\nQuestion: What's the capital of China?" --layers "last7" --neg projections/synthetic/llama3.2-1b-chat_neg_proj.pt --pos projections/synthetic/llama3.2-1b-chat_pos_proj.pt --model pretrained/llama3.2-1b-chat --chat --amplify-pos 1.0 --amplify-neg 1.0
```