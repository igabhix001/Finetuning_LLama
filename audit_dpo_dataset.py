"""
DPO Dataset Quality Audit Script
=================================
Analyzes existing DPO dataset for critical issues:
1. Name consistency (are names extracted correctly from charts?)
2. Date accuracy (do dates come from actual dasha periods?)
3. Metadata leakage (any debug info in responses?)
4. Past/future confusion (are past events in past tense?)
5. Safety issues (medical diagnoses, death predictions?)
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

def load_dpo_pairs(filepath):
    """Load DPO pairs from JSONL"""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs

def extract_name_from_prompt(prompt):
    """Extract chart name from prompt YAML"""
    match = re.search(r'name:\s*([^\n]+)', prompt)
    if match:
        return match.group(1).strip()
    return None

def extract_names_from_response(response):
    """Extract all names mentioned in response (e.g., 'Anjali Desai ji')"""
    # Pattern: Name followed by 'ji' or standalone Indian names
    pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+ji'
    matches = re.findall(pattern, response)
    return matches

def check_metadata_leakage(response):
    """Check if response contains internal metadata"""
    metadata_patterns = [
        r'rulesused:',
        r'timingmethod:',
        r'planetsinvolved:',
        r'housessignified:',
        r'outcome:',
        r'financialimpact:',
        r'careeradvancement:',
        r'KPCAR\d+',
        r'KPTIM\d+',
        r'KPPRO\d+',
    ]
    
    issues = []
    for pattern in metadata_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            issues.append(pattern)
    return issues

def check_date_patterns(response):
    """Extract date patterns to check for repetition"""
    # Pattern: Month Year to Month Year (e.g., "February 2026 to July 2026")
    pattern = r'([A-Z][a-z]+\s+\d{4})\s+to\s+([A-Z][a-z]+\s+\d{4})'
    matches = re.findall(pattern, response)
    return matches

def check_safety_issues(response):
    """Check for medical diagnoses or death predictions"""
    safety_issues = []
    
    # Medical diagnoses
    disease_patterns = [
        r'\bcancer\b',
        r'\bdiabetes\b',
        r'\bheart\s+disease\b',
        r'\bkidney\s+disease\b',
        r'\bliver\s+disease\b',
        r'\btumor\b',
        r'\bstroke\b',
    ]
    
    for pattern in disease_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            safety_issues.append(f"medical_diagnosis: {pattern}")
    
    # Death predictions (should be redirected, not predicted)
    if re.search(r'you will die|death.*(?:in|around|during)\s+\d{4}', response, re.IGNORECASE):
        safety_issues.append("death_prediction")
    
    return safety_issues

def check_past_future_confusion(prompt, response):
    """Check if past-tense questions get future-tense answers"""
    # Past tense question patterns
    past_patterns = [
        r'what happened',
        r'did i',
        r'when did',
        r'was there',
        r'were there',
        r'in \d{4}',  # "in 2020"
    ]
    
    is_past_question = any(re.search(p, prompt, re.IGNORECASE) for p in past_patterns)
    
    if is_past_question:
        # Check if response uses future dates (2025+)
        future_years = re.findall(r'\b(202[5-9]|203\d)\b', response)
        if future_years:
            return True, future_years
    
    return False, []

def audit_dataset(filepath):
    """Main audit function"""
    print(f"🔍 Auditing DPO dataset: {filepath}")
    print("=" * 80)
    
    pairs = load_dpo_pairs(filepath)
    print(f"📊 Total pairs: {len(pairs)}\n")
    
    # Issue counters
    name_mismatches = []
    metadata_leaks = []
    date_patterns = Counter()
    safety_issues = []
    past_future_confusions = []
    
    for idx, pair in enumerate(pairs):
        prompt = pair.get('prompt', '')
        chosen = pair.get('chosen', '')
        rejected = pair.get('rejected', '')
        
        # Extract chart name from prompt
        chart_name = extract_name_from_prompt(prompt)
        
        # Check chosen response
        if chosen:
            # Name consistency
            response_names = extract_names_from_response(chosen)
            if chart_name and response_names:
                for resp_name in response_names:
                    if resp_name.lower() != chart_name.lower():
                        name_mismatches.append({
                            'pair_idx': idx,
                            'chart_name': chart_name,
                            'response_name': resp_name,
                            'response_preview': chosen[:200]
                        })
            
            # Metadata leakage
            metadata = check_metadata_leakage(chosen)
            if metadata:
                metadata_leaks.append({
                    'pair_idx': idx,
                    'metadata_found': metadata,
                    'response_preview': chosen[:200]
                })
            
            # Date patterns
            dates = check_date_patterns(chosen)
            for date_range in dates:
                date_patterns[f"{date_range[0]} to {date_range[1]}"] += 1
            
            # Safety issues
            safety = check_safety_issues(chosen)
            if safety:
                safety_issues.append({
                    'pair_idx': idx,
                    'issues': safety,
                    'response_preview': chosen[:200]
                })
            
            # Past/future confusion
            is_confused, future_years = check_past_future_confusion(prompt, chosen)
            if is_confused:
                past_future_confusions.append({
                    'pair_idx': idx,
                    'future_years': future_years,
                    'prompt_preview': prompt[:200],
                    'response_preview': chosen[:200]
                })
    
    # Print results
    print("\n" + "=" * 80)
    print("📋 AUDIT RESULTS")
    print("=" * 80)
    
    print(f"\n🔴 CRITICAL ISSUES:")
    print(f"  • Name mismatches: {len(name_mismatches)}")
    print(f"  • Metadata leakage: {len(metadata_leaks)}")
    print(f"  • Safety issues: {len(safety_issues)}")
    print(f"  • Past/future confusion: {len(past_future_confusions)}")
    
    print(f"\n⚠️  DATE PATTERN ANALYSIS:")
    print(f"  • Unique date ranges: {len(date_patterns)}")
    print(f"  • Most common date ranges:")
    for date_range, count in date_patterns.most_common(10):
        print(f"    - '{date_range}': {count} times")
    
    # Detailed reports
    if name_mismatches:
        print(f"\n\n{'=' * 80}")
        print(f"🔴 NAME MISMATCH DETAILS (showing first 10):")
        print(f"{'=' * 80}")
        for issue in name_mismatches[:10]:
            print(f"\nPair #{issue['pair_idx']}:")
            print(f"  Chart name: {issue['chart_name']}")
            print(f"  Response name: {issue['response_name']}")
            print(f"  Preview: {issue['response_preview']}...")
    
    if metadata_leaks:
        print(f"\n\n{'=' * 80}")
        print(f"🔴 METADATA LEAKAGE DETAILS (showing first 5):")
        print(f"{'=' * 80}")
        for issue in metadata_leaks[:5]:
            print(f"\nPair #{issue['pair_idx']}:")
            print(f"  Metadata found: {issue['metadata_found']}")
            print(f"  Preview: {issue['response_preview']}...")
    
    if safety_issues:
        print(f"\n\n{'=' * 80}")
        print(f"🔴 SAFETY ISSUES DETAILS (showing all):")
        print(f"{'=' * 80}")
        for issue in safety_issues:
            print(f"\nPair #{issue['pair_idx']}:")
            print(f"  Issues: {issue['issues']}")
            print(f"  Preview: {issue['response_preview']}...")
    
    if past_future_confusions:
        print(f"\n\n{'=' * 80}")
        print(f"🔴 PAST/FUTURE CONFUSION DETAILS (showing first 5):")
        print(f"{'=' * 80}")
        for issue in past_future_confusions[:5]:
            print(f"\nPair #{issue['pair_idx']}:")
            print(f"  Future years in response: {issue['future_years']}")
            print(f"  Prompt: {issue['prompt_preview']}...")
            print(f"  Response: {issue['response_preview']}...")
    
    # Summary
    print(f"\n\n{'=' * 80}")
    print(f"📊 SUMMARY")
    print(f"{'=' * 80}")
    
    total_issues = len(name_mismatches) + len(metadata_leaks) + len(safety_issues) + len(past_future_confusions)
    issue_rate = (total_issues / len(pairs)) * 100 if pairs else 0
    
    print(f"Total pairs: {len(pairs)}")
    print(f"Total issues: {total_issues}")
    print(f"Issue rate: {issue_rate:.1f}%")
    
    if issue_rate > 10:
        print(f"\n❌ VERDICT: Dataset quality is POOR (>{10}% issue rate)")
        print(f"   Recommendation: Generate new dataset with fixes")
    elif issue_rate > 5:
        print(f"\n⚠️  VERDICT: Dataset quality is ACCEPTABLE ({issue_rate:.1f}% issue rate)")
        print(f"   Recommendation: Filter bad pairs and supplement with new data")
    else:
        print(f"\n✅ VERDICT: Dataset quality is GOOD (<5% issue rate)")
        print(f"   Recommendation: Minor cleanup, ready for training")
    
    return {
        'total_pairs': len(pairs),
        'name_mismatches': name_mismatches,
        'metadata_leaks': metadata_leaks,
        'safety_issues': safety_issues,
        'past_future_confusions': past_future_confusions,
        'date_patterns': date_patterns,
        'issue_rate': issue_rate
    }

if __name__ == "__main__":
    dpo_file = Path("data/dpo/dpo_pairs.jsonl")
    
    if not dpo_file.exists():
        print(f"❌ File not found: {dpo_file}")
        print(f"   Please run from project root or specify correct path")
        exit(1)
    
    results = audit_dataset(dpo_file)
    
    # Save results
    output_file = Path("data/dpo/audit_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert Counter to dict for JSON serialization
        results['date_patterns'] = dict(results['date_patterns'])
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
