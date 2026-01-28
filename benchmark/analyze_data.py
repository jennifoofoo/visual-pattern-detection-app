import sys
import os
import pandas as pd

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

from core.data_processing.loader import load_xes_log

def analyze():
    xes_path = os.path.join(current_dir, "eventdata", "DomesticDeclarations.xes")
    if not os.path.exists(xes_path):
        print(f"Error: {xes_path} not found")
        return
        
    print(f"Loading {xes_path}...")
    df = load_xes_log(xes_path)
    
    events = len(df)
    cases = df['case_id'].nunique()
    activities = df['activity'].nunique()
    
    counts = df.groupby('case_id').size()
    max_len = counts.max()
    avg_len = counts.mean()
    
    print(f"Events: {events}")
    print(f"Cases: {cases}")
    print(f"Activities: {activities}")
    print(f"Avg sequence length: {avg_len:.1f}")
    print(f"Max sequence length: {max_len}")
    
    # Check for many repetitive activities in a single case
    # PrefixSpan can explode if there are many identical items
    print("\nCase with max length events sample:")
    max_case_id = counts.idxmax()
    case_df = df[df['case_id'] == max_case_id]
    print(case_df['activity'].value_counts().head(5))

if __name__ == "__main__":
    analyze()
