import requests
from typer import prompt

class OllamaEvaluator:
    def __init__(self, model="llama2"):
        self.model = model

    def describe_chart(self, info, dots):
        dots_sample = dots.head(20).to_dict(
            orient="records")  # send only a sample for brevity
    #     prompt = (
    #         f"Describe the following event log summary and dotted chart data:\n"
    #     f"Summary:\n{summary}\n"
    #     f"clusters:\n{dots_sample}\n"
    #     f"Data points:\n{dots}\n"
    #     "What patterns or insights can you find?"
    # )
        prompt = (
            f"Describe the following process variants and their frequencies:\n"
            + "\n".join([f"{i+1}. {variant}: {count} cases" for i, (variant, count) in enumerate(info.items())])
        )
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt}
        )
        result = response.json()
        return result.get("response", "")
