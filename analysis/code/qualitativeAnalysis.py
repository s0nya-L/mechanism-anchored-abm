import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from dotenv import load_dotenv
import time
import os

load_dotenv()
client = OpenAI(api_key=os.getenv('API_KEY'), base_url="https://api.siliconflow.cn/")

CONSTRUCTS = {
    'Common Knowledge': [
        'We all know that', 'As we agreed', 'It is clear that', 'Obviously', 'Everyone understands',
        'Shared understanding', 'Mutual awareness', 'Common ground', 'We both know', 'It is evident that'
    ],
    'Loss Aversion': [
        'can\'t afford to lose', 'avoiding negative outcomes', 'preventing losses', 'minimize losses',
        'worst case scenario', 'risk of losing', 'downside risk', 'protect against loss', 'avoid failure', 'prevent damage'
    ],
    'Bounded Rationality': [
        'good enough solution', 'simplest option', 'easiest approach', 'practical solution', 'reasonable compromise',
        'manageable solution', 'straightforward approach', 'basic option', 'minimal complexity', 'adequate solution'
    ],
    'Schelling Points': [
        'meet halfway', 'usual approach', 'standard practice', 'natural choice', 'obvious solution',
        'default option', 'conventional method', 'typical approach', 'common practice', 'focal point'
    ]
}

def parse_negotiations(json_file):
    with open(json_file, 'r') as f:
        negotiations_raw = json.load(f)
    
    negotiations = []
    negotiation_id = 1
    
    for negotiation_str in negotiations_raw:
        lines = negotiation_str.strip().split('\n')
        header = lines[0]
        
        cycle = int(header.split(' - ')[0].replace('Cycle ', ''))
        retailer = header.split('Retailer: ')[1].split(', ')[0].strip()
        supplier = header.split('Supplier: ')[1].strip()
        
        messages = []
        for line in lines[3:]:
            if line.startswith('  Round'):
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    round_info = parts[0].replace('  Round ', '')
                    message = parts[1]
                    round_num = int(round_info.split(', ')[0])
                    speaker = round_info.split(', ')[1]
                    messages.append({'round': round_num, 'speaker': speaker, 'message': message})
        
        negotiations.append({
            'negotiation_id': f"N{negotiation_id:03d}",
            'cycle': cycle,
            'retailer': retailer,
            'supplier': supplier,
            'messages': messages,
            'full_text': negotiation_str
        })
        negotiation_id += 1
    
    return negotiations

def create_prompt(negotiation_text):
    return f"""Analyze the following negotiation text and identify instances of the specified theoretical constructs. 

Theoretical Constructs and their indicators:

1. Common Knowledge (Aumann, 1976):
   - Indicators: Explicit statements that assume shared understanding, repeated confirmations, or tactics that rely on mutual awareness
   - Examples: "We all know that...", "As we agreed...", "It is clear that..."

2. Loss Aversion (Kahneman & Tversky, 1979):
   - Indicators: Emphasis on avoiding losses rather than gaining benefits, framing offers in terms of preventing negative outcomes
   - Examples: "We can't afford to lose...", "avoiding negative outcomes", "minimize losses"

3. Bounded Rationality (Simon, 1955):
   - Indicators: Simplified decision-making, reliance on heuristics, ignoring complex options, or settling for "good enough" solutions
   - Examples: "Let's just pick the easiest option...", "good enough solution", "simplest approach"

4. Schelling Points (Schelling, 1960):
   - Indicators: Use of focal points or obvious solutions without explicit coordination, referencing natural defaults or conventions
   - Examples: "Let's meet halfway...", "The usual approach works best...", "standard practice"

Negotiation Text:
\"\"\"
{negotiation_text}
\"\"\"

For each construct found, provide:
1. The construct name
2. The specific text that demonstrates the construct
3. A brief explanation of why this text demonstrates the construct

Return your response in the following JSON format:
{{
  "Common Knowledge": [
    {{
      "text": "specific text",
      "explanation": "why this demonstrates common knowledge"
    }}
  ],
  "Loss Aversion": [
    {{
      "text": "specific text", 
      "explanation": "why this demonstrates loss aversion"
    }}
  ],
  "Bounded Rationality": [
    {{
      "text": "specific text",
      "explanation": "why this demonstrates bounded rationality"
    }}
  ],
  "Schelling Points": [
    {{
      "text": "specific text",
      "explanation": "why this demonstrates schelling points"
    }}
  ]
}}

If a construct is not found, return an empty array for that construct."""

def detect_constructs(negotiation_text):
    prompt = create_prompt(negotiation_text)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3-30B-A3B-Instruct-2507",
                messages=[
                    {"role": "system", "content": "You are an expert in behavioral economics and negotiation analysis."},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.1,
                max_tokens=1500
            )
            response_text = response.choices[0].message.content
            print(response_text)
            import re

            # Try to parse the response directly as JSON first
            try:
                result = json.loads(response_text.strip())
                # Convert to simple list format for compatibility
                return {construct: [item['text'] for item in instances] for construct, instances in result.items()}
            except json.JSONDecodeError as e:
                print(f"Direct JSON parsing failed: {e}")
                # Fall back to regex extraction
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        # Convert to simple list format for compatibility
                        return {construct: [item['text'] for item in instances] for construct, instances in result.items()}
                    except json.JSONDecodeError as e2:
                        print(f"Regex-extracted JSON parsing failed: {e2}")
                        print("Returning empty construct lists for this negotiation")
                        return {construct: [] for construct in CONSTRUCTS}
                else:
                    print("No JSON found in response")
                    return {construct: [] for construct in CONSTRUCTS}
        except Exception as e:
            print(f"API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("Max retries reached, returning empty")
                return {construct: [] for construct in CONSTRUCTS}

def analyze_negotiations(negotiations, csv_file="data/construct_frequency.csv"):
    # Check if file exists to determine if we need header
    file_exists = os.path.isfile(csv_file)
    
    for negotiation in negotiations:
        print("Negotiation ID:", negotiation['negotiation_id'])
        construct_analysis = detect_constructs(negotiation['full_text'])
        frequencies = {construct: len(instances) for construct, instances in construct_analysis.items()}
        
        result = {
            'negotiation_id': negotiation['negotiation_id'],
            'cycle': negotiation['cycle'],
            'retailer': negotiation['retailer'],
            'supplier': negotiation['supplier'],
            'total_messages': len(negotiation['messages']),
            'total_rounds': max([msg['round'] for msg in negotiation['messages']]) + 1 if negotiation['messages'] else 0,
            **frequencies,
            'total_constructs': sum(frequencies.values())
        }
        
        # Append to CSV
        df_result = pd.DataFrame([result])
        df_result.to_csv(csv_file, mode='a', header=not file_exists, index=False)
        file_exists = True  # After first write, header is written
        
        time.sleep(0.5)

def create_cooccurrence_matrix(results):
    cooccurrence_matrix = pd.DataFrame(0, index=CONSTRUCTS, columns=CONSTRUCTS)
    
    for result in results:
        present_constructs = [construct for construct in CONSTRUCTS if result[construct] > 0]
        
        for i, construct1 in enumerate(present_constructs):
            for construct2 in present_constructs[i:]:
                if construct1 == construct2:
                    cooccurrence_matrix.loc[construct1, construct2] += result[construct1]
                else:
                    cooccurrence_matrix.loc[construct1, construct2] += 1
                    cooccurrence_matrix.loc[construct2, construct1] += 1
    
    return cooccurrence_matrix

def create_heatmap(results, cooccurrence_matrix):
    df_results = pd.DataFrame(results)
    df_results['negotiation_label'] = df_results['negotiation_id'] + '\n(C' + df_results['cycle'].astype(str) + ')'

    heatmap_data = df_results[['negotiation_label'] + list(CONSTRUCTS.keys())].set_index('negotiation_label')

    # Original heatmap (tall, but readable)
    fig, ax = plt.subplots(figsize=(12, len(results)*0.3))  # Adjust height based on number of negotiations
    sns.heatmap(heatmap_data, annot=False, cmap='coolwarm', center=0, linewidths=0.5, cbar_kws={'label': 'Frequency'}, ax=ax)
    ax.set_title('Frequency of Theoretical Constructs in Negotiations', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Theoretical Constructs', fontsize=12)
    ax.set_ylabel('Negotiations (Grouped by Cycle)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=6)  # Smaller font for readability
    plt.tight_layout()
    plt.savefig("analysis/plots/construct_heatmap_original.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # Grouped by cycle heatmap
    cycle_averages = df_results.groupby('cycle')[list(CONSTRUCTS.keys())].mean()
    cycle_averages.index = 'Cycle ' + cycle_averages.index.astype(str)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cycle_averages, annot=True, cmap='coolwarm', center=0, linewidths=0.5, fmt='.2f', cbar_kws={'label': 'Average Frequency'}, ax=ax)
    ax.set_title('Average Frequency of Theoretical Constructs by Cycle', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Theoretical Constructs', fontsize=12)
    ax.set_ylabel('Cycle', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.savefig("analysis/plots/construct_heatmap_by_cycle.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # Clustered heatmap
    g = sns.clustermap(heatmap_data, cmap='coolwarm', center=0, linewidths=0.5, cbar_kws={'label': 'Frequency'},
                       figsize=(12, len(results)*0.3), dendrogram_ratio=(0.1, 0.1))
    g.ax_heatmap.set_title('Clustered Heatmap of Theoretical Constructs in Negotiations', fontsize=14, fontweight='bold', pad=20)
    g.ax_heatmap.set_xlabel('Theoretical Constructs', fontsize=12)
    g.ax_heatmap.set_ylabel('Negotiations (Clustered)', fontsize=12)
    g.ax_heatmap.tick_params(axis='both', which='major', labelsize=6)
    plt.tight_layout()
    plt.savefig("analysis/plots/construct_heatmap_clustered.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # Co-occurrence matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cooccurrence_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5, fmt='d', cbar_kws={'label': 'Co-occurrence Count'}, ax=ax)
    ax.set_title('Co-occurrence Matrix of Theoretical Constructs', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Theoretical Constructs', fontsize=12)
    ax.set_ylabel('Theoretical Constructs', fontsize=12)
    plt.tight_layout()
    plt.savefig("analysis/plots/cooccurrence_matrix.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Read the full results from CSV (assuming it's already generated)
    df = pd.read_csv("data/construct_frequency.csv")
    results = df.to_dict('records')

    cooccurrence_matrix = create_cooccurrence_matrix(results)
    cooccurrence_matrix.to_csv("data/cooccurrence_matrix.csv")

    create_heatmap(results, cooccurrence_matrix)

    summary_stats = {
        'total_negotiations': len(results),
        'avg_constructs_per_negotiation': df['total_constructs'].mean(),
        'max_constructs_in_single_negotiation': df['total_constructs'].max()
    }

    for construct in CONSTRUCTS:
        summary_stats[f'{construct}_frequency'] = df[construct].sum()
        summary_stats[f'{construct}_presence_rate'] = (df[construct] > 0).mean()

    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv("data/construct_summary_statistics.csv", index=False)

if __name__ == "__main__":
    main()
