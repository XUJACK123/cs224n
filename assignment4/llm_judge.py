import re
import json
import os
from typing import List, Dict

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query, QueryResponse
from client.query import query_model


# You may find these constants useful for structuring the judge's output.
MODEL_E_PREFERED_TAG = "<MODEL_E_BETTER>"
MODEL_F_PREFERED_TAG = "<MODEL_F_BETTER>"
NO_PREFERENCE_FOUND_TAG = "<NO_PREFERENCE_FOUND>"


def load_alpaca_data() -> List[Dict[str, str]]:

    dataset = []
    with open("./data/alpaca_eval_first_30.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            dataset.append(example)

    return dataset

def llm_judge_template(query: str, response_E: str, response_F: str) -> str:
    """
    Construct a prompt for an LLM judge to evaluate two model responses.

    Args:
        query: the question given to the two models (from AlpacaEval)
        response_E: output from model E on query
        response_F: output from model F on query
    Returns:
        Prompt for the LLM judge.
    
    Consider: The judge is an LLM that will output free-form text. How will you 
    design the prompt so that you can reliably determine which response it preferred?
    Your llm_judge_template and extract_llm_judge_preference should work together.
    """
    prompt = f"""You are a helpful, neutral, and precise AI assistant acting as an expert judge.
Your task is to evaluate the quality of two responses (Response E and Response F) given to the user prompt below.
[User Prompt]{query}
[Response E]{response_E}
[Response F]{response_F}
[output]
- output "[[E]]" if Response E is better
- output "[[F]]" if Response F is better
"""
    return prompt

def extract_llm_judge_preference(judge_output: str) -> str:
    """
    Extract the judge's preference from its output.

    Args:
        judge_output: the string sampled from the LLM judge.
    Returns:
        A string representing which response the judge preferred.
    
    This function should work in tandem with your llm_judge_template design.
    What if the judge's output is malformed or ambiguous?
    """
    match = re.search(r"\[\[(E|F)\]\]", judge_output)
    if match:
        return match.group(1)
    return "UNKNOWN"
def run_llm_judge_eval():
    """
    Run the LLM-as-a-judge evaluation comparing models E and F on AlpacaEval data.
    Use model Z as the judge.
    
    For each AlpacaEval instruction, you'll need responses from both models E and F,
    then have the judge compare them.
    
    Remember to save your results (model responses + judge outputs) - you will 
    need them for Parts C and D.
    """
    data = load_alpaca_data()
    results = []
    e_wins = 0
    f_wins = 0
    for item in data:
        query = item.get("instruction", item.get("query"))
        response_E = query_model("E", Query(turns=[{"user": query}]))
        response_F = query_model("F", Query(turns=[{"user": query}]))
        judge_prompt = llm_judge_template(query, response_E.text, response_F.text)
        judge_output = query_model("Z", Query(turns=[{"user": judge_prompt}]))
        preference = extract_llm_judge_preference(judge_output.text)
        if preference == "E":
            e_wins += 1
        elif preference == "F":
            f_wins += 1
        results.append(
            {
                "query": query,
                "response_E": response_E.text,
                "response_F": response_F.text,
                "judge_output": judge_output.text,
                "preference": preference,
            }
        )
    with open("data/llm_judge_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_valid = e_wins + f_wins
    win_rate_E = (e_wins / total_valid * 100) if total_valid > 0 else 0.0
    print(
        f"Evaluation complete! Model E Win Rate: {win_rate_E:.2f}% ({e_wins}/{total_valid})"
    )

    return results
    
def plot_model_output_lengths() -> None:
    """
    For Part D: Plot histograms of response lengths for preferred vs. not-preferred outputs.
    """
    with open("data/llm_judge_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    preferred_lengths = []
    not_preferred_lengths = []
    for item in results:
        pref = item.get("preference")
        len_E = len(item["response_E"])
        len_F = len(item["response_F"])
        if pref == "E":
            preferred_lengths.append(len_E)
            not_preferred_lengths.append(len_F)
        elif pref == "F":
            preferred_lengths.append(len_F)
            not_preferred_lengths.append(len_E)
        plt.figure(figsize=(9, 5))
    plt.hist(
        preferred_lengths,
        bins=10,
        alpha=0.6,
        label="Preferred Output Length",
        color="skyblue",
        edgecolor="black",
    )
    plt.hist(
        not_preferred_lengths,
        bins=10,
        alpha=0.6,
        label="Not Preferred Output Length",
        color="salmon",
        edgecolor="black",
    )

    plt.xlabel("Output Length (Characters)")
    plt.ylabel("Frequency")
    plt.title(
        "Response Length Distributions: Preferred vs. Not Preferred Outputs"
    )
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("output_lengths_histogram.png")
    plt.show()
    

if __name__=="__main__":

    load_dotenv()

    ## Uncomment to run your code
    #run_llm_judge_eval()
    #plot_model_output_lengths()
