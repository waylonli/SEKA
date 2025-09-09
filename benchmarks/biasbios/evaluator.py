import json 
from itertools import combinations
import spacy

def get_all_substring_index(string, substring):
    index = [i for i in range(len(string)) if string.startswith(substring, i)]
    return index

class InstructionEvaluator(object):
    """
    The Evaluator for instruction following tasks: JSON Formatting and Pronouns Changing. 

    Args:
        task (`str`): The name of evaluation task (`json` or `pronchange`). 
        prompt_idx (`int`): The index of the prompte template. See `DEFAULT_PROMPT`. 

    Returns: 
        InstructionEvaluator
    """
    INSTRUCTION_TASK_NAMES = [
        "json", "pronchange", 
    ]
    DEFAULT_PROMPT = {
        "json": [
            '''
            Answer the occupation of {} and generate the answer as json format. Here is an example: {{"name": , "occupation": }}. Now generate the answer:
            ''',
            '''
            Generate the occupation of {} as json format: 
            ''',
            '''
            Answer the occupation of {} and generate the answer as json format: 
            ''',
            '''
            **Answer the occupation of {} and generate the answer as json format. Here is an example: {{"name": , "occupation": }}. Now generate the answer:**
            ''',
            '''
            "Answer the occupation of {} and generate the answer as json format. Here is an example: {{"name": , "occupation": }}. Now generate the answer:"
            ''',
            '''
            Answer the occupation of {} and generate the answer as json format. Here is an example: {{"name": , "occupation": }}. 
            ''',
        ],
        "pronchange": [
            '''
            For the aforementioned text, substitute 'she' and 'he' with 'they' and generate the occupation of {} after changing pronouns:
            ''',
            '''
            For the aforementioned descriptions, replace 'she' and 'he' with 'they' in the aformentioned text and generate the new text after replacing the pronouns:
            ''',
            '''
            Change 'she' and 'he' with 'they' and answer the occupation of {} after replacing the pronouns: 
            ''',
            '''
            **For the aforementioned text, substitute 'she' and 'he' with 'they' and generate the occupation of {} after changing pronouns:**
            ''',
            '''
            "For the aforementioned text, substitute 'she' and 'he' with 'they' and generate the occupation of {} after changing pronouns:"
            ''',
        ], 
    }
    JSON_ALIKE_CHARS = ["{", "}", ":", ","]
    PRON_CHANGE_TARGET = ["she", "he"] 
    PRON_ALL_CHANGE_TARGET = ["she", "he", "her", "him", "hers", "his", "herself", "himself"] 
    PRON_REPLACEMENTS = ["they", "them", "their", "theirs", "themselves"]

    def __init__(
        self, 
        task:str,
        prompt_idx: int|list|None = None, 
    ):
        self.task = task
        assert task in self.INSTRUCTION_TASK_NAMES 

        if isinstance(prompt_idx, int): 
            self.prompt_temple = [self.DEFAULT_PROMPT[task][prompt_idx]]
            self.prompt_idx = [prompt_idx]
        elif isinstance(prompt_idx, list): 
            self.prompt_temple = [self.DEFAULT_PROMPT[task][i] for i in prompt_idx]
            self.prompt_idx = prompt_idx 
        else:
            self.prompt_temple = self.DEFAULT_PROMPT[task]
            self.prompt_idx = list(range(len(self.DEFAULT_PROMPT[task])))
        
        self.prompt_temple = [prompt.strip() for prompt in self.prompt_temple] 
        self.num_prompt = len(self.prompt_temple) 
        self.nlp = spacy.load("en_core_web_sm")
        
    def parapare_prompt_inputs(
            self, 
            contexts: list, 
            entities: list, 
            other_text_inputs: tuple[list], 
        ):
        """Apply the prompt templates and generation input strings for models."""
        prompts = [
            self.apply_prompt_templete(context, entity, prompt, idx)
                for context,entity in zip(contexts, entities) for prompt,idx in zip(self.prompt_temple, self.prompt_idx)
        ]
        instructions = [
            prompt.format(entity) for entity in entities for prompt in self.prompt_temple 
        ]
        new_entities = [
            entity for entity in entities for i in range(self.num_prompt)
        ]
        new_other_inputs = []
        for text_input in other_text_inputs:
            new_text_input = [
                val for val in text_input for i in range(len(self.prompt_temple))
            ]
            new_other_inputs.append(new_text_input)
        return prompts, instructions, new_entities, tuple(new_other_inputs)
    
    def evaluate_sample(self, generation, target=None, original_context=None, **kwargs):
        result = {}
        if self.task == "json":
            result["json_EM"] = self.evaluate_json_EM(generation)
            if target is not None:
                result['json_pred_acc'] = self.evaluate_json_pred(generation, target)
            for requrie_num in range(2, len(self.JSON_ALIKE_CHARS)+1):
                result[f"json_alike_{requrie_num}"] = self.evaluate_json_alike(generation, requrie_num)
            return result 
        elif self.task == "pronchange":
            # Enhanced pronoun evaluation with weighted lexical overlap
            pron_results = self.evaluate_pronounce_change_weighted(generation, original_context)
            result.update(pron_results)
        return result 
    
    def apply_prompt_templete(self, context, entity, prompt_templete, prompt_idx):
        """Apply the prompt template"""
        prompt = context+'\n'+prompt_templete.format(entity)+'\n' 
        if self.task == "json" and prompt_idx in [5]:
            prompt = prompt_templete.format(entity)+'\n'+context+'\n'+"Now generate the answer:"
        else:
            prompt = context+'\n'+prompt_templete.format(entity)+'\n' 
        return prompt
    
    def evaluate_json_alike(self, generation, require_num):
        """
        Evaluate how generations are alike JSON format
        by checking whether generations include '{', ':', ',', '}'. 
        """
        all_combs = list(combinations(self.JSON_ALIKE_CHARS, require_num))
        for chars in all_combs:
            if all([char in generation for char in chars]):
                return True 
        return False 

    def evaluate_json_EM(self, generation):
        """
        Evaluate whether generations exactly match JSON format 
        by `json.loads` function. 
        """
        generation = generation.strip()
        if "{" not in generation and "}" not in generation:
            return False
        
        string_begs = get_all_substring_index(generation, "{")
        string_ends = get_all_substring_index(generation, "}")
        # Try to load every substring starting from '{' and end with '}'
        for idx_beg in string_begs:
            for idx_end in string_ends:
                substring = generation[idx_beg:idx_end+1]
                try:
                    json.loads(substring)
                    return True 
                except ValueError:
                    continue
        return False 
    
    def evaluate_json_pred(self, generation, target):
        """
        Evaluate whether generated JSON outputs correctly include 
        the traget label by checking its `values` after `json.loads`
        """
        generation = generation.strip()
        if "{" not in generation and "}" not in generation:
            return False
        
        string_begs = get_all_substring_index(generation, "{")
        string_ends = get_all_substring_index(generation, "}")
        for idx_beg in string_begs:
            for idx_end in string_ends:
                substring = generation[idx_beg:idx_end+1]
                try:
                    pred = json.loads(substring)
                    pred_values = [p.lower() if isinstance(p, str) else p for p in pred.values()]
                    if target.lower() in pred_values:
                        return True 
                    for value in pred_values:
                        if isinstance(value, str) and target.lower() in value:
                            return True
                except ValueError:
                    continue
        return False 
    
    def evaluate_pronounce_change_weighted(self, generation, original_context):
        """
        Weighted lexical overlap evaluation for pronoun changes
        
        Score = (pronoun_conversion_rate × content_overlap) / total_content_tokens
        where pronoun_conversion_rate = successfully_converted / total_target_pronouns
        """
        if not original_context:
            # Fallback to original method if no context
            print("No original context provided. Falling back to legacy evaluation.")
            return self.evaluate_pronounce_change_legacy(generation)
        import re

        generation = re.sub(
            r"For the aforementioned text,\s*substitute[\s\S]*?after changing pronouns:\s*",
            "",
            generation,
            flags=re.IGNORECASE | re.DOTALL
        )
        
        # Tokenise both texts
        orig_doc = self.nlp(original_context.lower())
        gen_doc = self.nlp(generation.lower())
        
        orig_tokens = [token.text for token in orig_doc if not token.is_space]
        gen_tokens = [token.text for token in gen_doc if not token.is_space]
        
        # Count target pronouns in original (with frequency)
        from collections import Counter
        target_pronouns_basic_count = Counter([t for t in orig_tokens if t in self.PRON_CHANGE_TARGET])
        target_pronouns_all_count = Counter([t for t in orig_tokens if t in self.PRON_ALL_CHANGE_TARGET])
        
        # Count remaining old pronouns in generation
        remaining_pronouns_basic_count = Counter([t for t in gen_tokens if t in self.PRON_CHANGE_TARGET])
        remaining_pronouns_all_count = Counter([t for t in gen_tokens if t in self.PRON_ALL_CHANGE_TARGET])
        
        # Extract content tokens (excluding all pronouns)
        orig_content = [t for t in orig_tokens if t not in self.PRON_ALL_CHANGE_TARGET + self.PRON_REPLACEMENTS]
        gen_content = [t for t in gen_tokens if t not in self.PRON_ALL_CHANGE_TARGET + self.PRON_REPLACEMENTS]
        
        # Calculate pronoun conversion rates
        total_basic_pronouns = sum(target_pronouns_basic_count.values())
        if total_basic_pronouns == 0:
            pronoun_weight_basic = 1.0  # No pronouns to change
        else:
            remaining_basic = sum(remaining_pronouns_basic_count.values())
            converted_basic = total_basic_pronouns - remaining_basic
            pronoun_weight_basic = max(0.0, converted_basic / total_basic_pronouns)
        
        total_all_pronouns = sum(target_pronouns_all_count.values())
        if total_all_pronouns == 0:
            pronoun_weight_all = 1.0  # No pronouns to change
        else:
            remaining_all = sum(remaining_pronouns_all_count.values())
            converted_all = total_all_pronouns - remaining_all
            pronoun_weight_all = max(0.0, converted_all / total_all_pronouns)
        
        # Calculate content overlap
        content_overlap = len(set(orig_content) & set(gen_content))
        total_content_tokens = len(orig_content)
        
        # Weighted scores
        if total_content_tokens == 0:
            # Handle edge case where original has no content tokens
            score_basic = pronoun_weight_basic
            score_all = pronoun_weight_all
        else:
            score_basic = (pronoun_weight_basic * content_overlap) / total_content_tokens
            score_all = (pronoun_weight_all * content_overlap) / total_content_tokens

        return {
            "pron_weighted_basic": score_basic,
            "pron_weighted_all": score_all,
            "pron_sub_acc": score_basic,  # Backward compatibility
            "pron_all_acc": score_all,   # Backward compatibility
        }
    
    def _calculate_f1(self, precision, recall):
        """Calculate F1 score from precision and recall - kept for potential future use"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate_pronounce_change_legacy(self, generation, original_context=None):
        """
        Legacy evaluation method for backward compatibility
        """
        generation_lower = generation.lower()
        all_tokens = [token.text for token in self.nlp(generation_lower)]
        
        # Original pronoun checks
        pron_sub_raw = True
        pron_all_raw = True
        for word in all_tokens:
            if word in self.PRON_CHANGE_TARGET:
                pron_sub_raw = False
            if word in self.PRON_ALL_CHANGE_TARGET:
                pron_all_raw = False
        
        # Content preservation check
        if len(generation.strip()) == 0:
            return {"pron_sub_acc": 0.0, "pron_all_acc": 0.0}
        
        if original_context is None:
            # Fallback to length-based heuristic if context not provided
            min_length = 20
            content_preservation = min(1.0, len(generation.strip()) / min_length)
        else:
            # Proper content preservation using original context
            orig_tokens = set([token.text.lower() for token in self.nlp(original_context) 
                            if token.text.lower() not in self.PRON_ALL_CHANGE_TARGET])
            gen_tokens = set([token.text for token in self.nlp(generation_lower) 
                            if token.text not in self.PRON_ALL_CHANGE_TARGET + ['they', 'them', 'their', 'theirs', 'themselves']])
            
            if len(orig_tokens) == 0:
                content_preservation = 1.0
            else:
                content_preservation = len(orig_tokens & gen_tokens) / len(orig_tokens)
        
        # Combined scores
        pron_sub_combined = float(pron_sub_raw) * content_preservation
        pron_all_combined = float(pron_all_raw) * content_preservation
        
        return {"pron_sub_acc": pron_sub_combined, "pron_all_acc": pron_all_combined}
        
    def aggregate_evaluation_results(self, samples):
        """Aggergate the evaluation results"""
        result_key = list(samples[0].instruction_evaluation.keys())
        grouped_results = {}
        for sample in samples:
            if sample.id not in grouped_results:
                grouped_results[sample.id] = {key:[] for key in result_key}
            for key in sample.instruction_evaluation:
                grouped_results[sample.id][key].append(sample.instruction_evaluation[key])
        
        all_result = {key:[] for key in result_key}
        for _, sample_eval in grouped_results.items():
            for key in sample_eval:
                all_result[key].append(any(sample_eval[key])) 

        return {key:sum(value)/len(value) for key,value in all_result.items()}
    
    def is_pred_true_sample_result(self, sample):
        if self.task == "json":
            return sample.instruction_evaluation["json_EM"]
        elif self.task == "pronchange":
            # Use weighted score if available, fallback to legacy
            if "pron_weighted_basic" in sample.instruction_evaluation:
                return sample.instruction_evaluation["pron_weighted_basic"]
            else:
                return sample.instruction_evaluation["pron_sub_acc"]

    def get_generation_label(self, context, entity, target):
        if self.task == "json":
            label = '{"name":"%s", "occupation":"%s"}'%(entity, target)
        elif self.task == "pronchange":
            label = context.lower().replace('she', 'they').replace(
                'he', 'they').replace('him', 'them').replace('her', 'them')
        return label 

    def prepare_fewshot_examples(
            self, fewshot_samples, example_sep = "\n###\n", text_target_sep = " ", 
        ):
        """Prepare few-shot demonstration"""
        labeled_example = [] 
        for example in fewshot_samples:
            context = example['context']
            entity = example['entity']
            target = example['target_mediated'] 

            label = self.get_generation_label(context, entity, target) 
            labeled_example.append(context+text_target_sep+label)
        
        fewshot_examples = example_sep.join(labeled_example) + example_sep
        return fewshot_examples