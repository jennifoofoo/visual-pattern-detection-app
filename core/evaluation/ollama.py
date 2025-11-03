import requests
import json

class OllamaEvaluator:
    def __init__(self, model="llama2"):
        self.model = model
    
    def describe_chart(self, top_variants, df_base):
        prompt = f"""
        Analyze this process mining data and provide insights:
        
        Top process variants:
        {top_variants.to_string()}
        
        Total cases: {df_base['case_id'].nunique()}
        Total events: {len(df_base)}
        
        Please provide a short analysis of the process patterns.
        """
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model, 
                "prompt": prompt,
                "stream": False  # Request non-streaming response
            }
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                return result.get("response", "No response received")
            except json.JSONDecodeError:
                # If still getting streaming response, handle it manually
                lines = response.text.strip().split('\n')
                full_response = ""
                for line in lines:
                    if line.strip():
                        try:
                            json_obj = json.loads(line)
                            if "response" in json_obj:
                                full_response += json_obj["response"]
                        except json.JSONDecodeError:
                            continue
                return full_response if full_response else "Failed to parse response"
        else:
            return f"Error: HTTP {response.status_code} - {response.text}"