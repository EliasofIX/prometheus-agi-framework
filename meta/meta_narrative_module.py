# meta/meta_narrative_module.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MetaNarrativeModule:
    """
    Generates natural language explanations and self-reflections.
    """
    def __init__(self, knowledge_graph, llm_model_name="gpt2"):
        self.kg = knowledge_graph
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.model.eval()

    def explain_action(self, action_desc, context):
        causal_path = self.kg.trace_causal_path(action_desc)
        prompt = (f"Explain why the action '{action_desc}' was taken in context '{context}'. "
                  f"Causal path: {causal_path}\nExplanation:")
        return self._generate_text(prompt)

    def reflect_on_failure(self, observed_outcome, expected_outcome):
        if observed_outcome != expected_outcome:
            prompt = (f"Expected: {expected_outcome} but got: {observed_outcome}. "
                      "Explain the discrepancy and propose improvements.\nReflection:")
            reflection = self._generate_text(prompt)
            self.kg.update_concept_properties("System_Self", {"last_reflection": reflection})
            return reflection
        return "Outcome as expected; no reflection needed."

    def _generate_text(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_length=max_length, do_sample=False)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

