import pandas as pd
import json
from typing import List


def split_negotiations(input_file: str, output_file: str = None) -> List[str]:
    df = pd.read_csv(input_file)
    logs = df.to_dict('records')
    
    negotiations = []
    cycle_boundaries = []
    
    for i, log in enumerate(logs):
        if log['round'] == 0 and i > 0:
            if logs[i-1]['round'] > 0:
                cycle_boundaries.append(i)
    
    cycle_boundaries.append(len(logs))
    
    for i, boundary in enumerate(cycle_boundaries):
        if i == 0:
            cycle_logs = logs[:boundary]
        else:
            cycle_logs = logs[cycle_boundaries[i-1]:boundary]
        
        if not cycle_logs:
            continue
            
        current_cycle = i + 1
        
        negotiation_groups = {}
        
        for log in cycle_logs:
            key = (log['retailer'], log['supplier'])
            if key not in negotiation_groups:
                negotiation_groups[key] = []
            negotiation_groups[key].append(log)
        
        for (retailer, supplier), logs_for_negotiation in negotiation_groups.items():
            negotiation_str = f"Cycle {current_cycle} - Retailer: {retailer}, Supplier: {supplier}\n"
            negotiation_str += "Negotiation Logs:\n"
            
            for log in logs_for_negotiation:
                negotiation_str += f"  Round {log['round']}, {log['speaker']}: {log['message']}\n"
            
            negotiation_str += f"\nTotal rounds: {max(log['round'] for log in logs_for_negotiation) + 1}\n"
            negotiation_str += f"Total messages: {len(logs_for_negotiation)}\n"
            negotiation_str += "-" * 80 + "\n"
            
            negotiations.append(negotiation_str)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(negotiations, f, indent=2)
    
    return negotiations


if __name__ == "__main__":
    negotiations = split_negotiations('data/rawNegotiations/95simNEGOTIATIONSgrp2.csv', 'negotiations_output.json')
