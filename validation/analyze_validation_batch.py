#!/usr/bin/env python3
"""
Automated Validation Batch Analyzer
Compares validation results with previous eras and generates publication report.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from validation.analyze_symbol_journeys import load_and_process_logs
from validation.validate_symbol_data import validate_symbol_journey_data

class ValidationBatchAnalyzer:
    """Automated analysis pipeline for validation batch results."""
    
    def __init__(self):
        self.reports_dir = Path("reports")
        self.logs_dir = Path("logs")
        
    def analyze_binder_distribution(self, promotions):
        """Analyze BINDER promotion distribution."""
        
        agent_counts = {}
        for promotion in promotions:
            agent = promotion['user']
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        total_promotions = len(promotions)
        distribution = {}
        
        for agent, count in agent_counts.items():
            percentage = (count / total_promotions * 100) if total_promotions > 0 else 0
            distribution[agent] = {
                'count': count,
                'percentage': percentage
            }
        
        return distribution, total_promotions
    
    def compare_with_previous_eras(self, validation_data):
        """Compare validation results with Eve and Balanced eras."""
        
        # Historical data from previous reports
        eve_era = {
            'total_promotions': 16,
            'distribution': {
                'eve': {'count': 13, 'percentage': 81.0},
                'zara': {'count': 3, 'percentage': 19.0},
                'alice': {'count': 1, 'percentage': 6.0}
            }
        }
        
        balanced_era = {
            'total_promotions': 8,
            'distribution': {
                'alice': {'count': 5, 'percentage': 63.0},
                'bob': {'count': 2, 'percentage': 25.0},
                'zara': {'count': 1, 'percentage': 13.0}
            }
        }
        
        adaptive_immune_era = {
            'total_promotions': 46,
            'distribution': {
                'alice': {'count': 31, 'percentage': 67.4},
                'eve': {'count': 4, 'percentage': 8.7},
                'bob': {'count': 11, 'percentage': 23.9}
            }
        }
        
        comparison = {
            'eve_era': eve_era,
            'balanced_era': balanced_era,
            'adaptive_immune_era': adaptive_immune_era,
            'multi_agent_adaptive_era': validation_data
        }
        
        return comparison
    
    def assess_alice_dominance(self, distribution, total_promotions):
        """Assess whether Alice dominance is problematic or natural variation."""
        
        alice_percentage = distribution.get('alice', {}).get('percentage', 0)
        
        # Analysis criteria
        assessment = {
            'alice_percentage': alice_percentage,
            'status': 'unknown',
            'recommendation': 'unknown',
            'reasoning': []
        }
        
        if alice_percentage >= 70:
            assessment['status'] = 'problematic_dominance'
            assessment['recommendation'] = 'rebalance_needed'
            assessment['reasoning'].append(f"Alice at {alice_percentage:.1f}% shows monopolistic pattern (>70%)")
            
        elif alice_percentage >= 50:
            assessment['status'] = 'majority_leader'
            assessment['recommendation'] = 'monitor_closely'
            assessment['reasoning'].append(f"Alice at {alice_percentage:.1f}% is majority leader but not monopolistic")
            
            # Check if other agents are competitive
            other_agents = {k: v for k, v in distribution.items() if k != 'alice'}
            max_competitor = max(other_agents.values(), key=lambda x: x['percentage']) if other_agents else None
            
            if max_competitor and max_competitor['percentage'] >= 20:
                assessment['reasoning'].append(f"Healthy competition exists (max competitor: {max_competitor['percentage']:.1f}%)")
                assessment['recommendation'] = 'acceptable_variation'
            else:
                assessment['reasoning'].append("Weak competition from other agents")
                
        else:
            assessment['status'] = 'balanced_competition'
            assessment['recommendation'] = 'optimal_balance'
            assessment['reasoning'].append(f"Alice at {alice_percentage:.1f}% shows balanced participation")
        
        # Sample size consideration
        if total_promotions < 12:
            assessment['reasoning'].append(f"Small sample size (n={total_promotions}) limits statistical confidence")
            if assessment['recommendation'] == 'rebalance_needed':
                assessment['recommendation'] = 'collect_more_data'
        
        return assessment
    
    def calculate_mallory_integration(self, symbol_data):
        """Calculate Mallory participation metrics."""
        
        total_symbols = sum(len(interpretations) for interpretations in symbol_data.values())
        
        mallory_symbols = 0
        mallory_promote = 0
        mallory_demote = 0
        
        for symbol, interpretations in symbol_data.items():
            if 'mallory' in symbol.lower():
                mallory_symbols += len(interpretations)
                
                if 'promote' in symbol.lower():
                    mallory_promote += len(interpretations)
                elif 'demote' in symbol.lower():
                    mallory_demote += len(interpretations)
        
        participation_rate = (mallory_symbols / total_symbols * 100) if total_symbols > 0 else 0
        
        return {
            'total_symbols': mallory_symbols,
            'promote_symbols': mallory_promote,
            'demote_symbols': mallory_demote,
            'participation_rate': participation_rate,
            'ratio_demote_to_promote': (mallory_demote / mallory_promote) if mallory_promote > 0 else float('inf')
        }
    
    def generate_validation_report(self, log_files, tag="validation"):
        """Generate comprehensive validation report."""
        
        print(f"🔬 Analyzing validation batch: {tag}")
        
        # Load and process data
        symbol_data, promotions, summary = load_and_process_logs(log_files)
        binder_distribution, total_promotions = self.analyze_binder_distribution(promotions)
        
        # Validate data quality  
        validation_results = validate_symbol_journey_data([Path(f) for f in log_files])
        
        # Analyze key metrics
        comparison = self.compare_with_previous_eras({
            'total_promotions': total_promotions,
            'distribution': binder_distribution
        })
        
        alice_assessment = self.assess_alice_dominance(binder_distribution, total_promotions)
        mallory_metrics = self.calculate_mallory_integration(symbol_data)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y-%m-%d")
        report = {
            'metadata': {
                'generated': timestamp,
                'tag': tag,
                'log_files_count': len(log_files),
                'total_promotions': total_promotions
            },
            'data_quality': validation_results,
            'binder_analysis': {
                'distribution': binder_distribution,
                'total_promotions': total_promotions,
                'alice_assessment': alice_assessment
            },
            'comparison': comparison,
            'mallory_integration': mallory_metrics,
            'symbol_summary': {
                'total_interpretations': sum(len(interp) for interp in symbol_data.values()),
                'unique_symbols': len(symbol_data)
            }
        }
        
        # Save report
        report_filename = f"03_validation_analysis_{timestamp}_{tag}.json"
        report_path = self.reports_dir / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown summary
        self.generate_markdown_summary(report, report_path.with_suffix('.md'))
        
        print(f"📊 Validation analysis complete: {report_path}")
        return report
    
    def generate_markdown_summary(self, report, output_path):
        """Generate human-readable markdown summary."""
        
        md_content = f"""# DSBL Validation Batch Analysis

**Generated:** {report['metadata']['generated']}  
**Tag:** {report['metadata']['tag']}  
**Batch Size:** {report['metadata']['log_files_count']} runs × 60 tickets  

## Quick Assessment

### 🎯 Alice Dominance Status
**{report['binder_analysis']['alice_assessment']['status'].replace('_', ' ').title()}**

- **Alice BINDER Share:** {report['binder_analysis']['alice_assessment']['alice_percentage']:.1f}%
- **Recommendation:** {report['binder_analysis']['alice_assessment']['recommendation'].replace('_', ' ').title()}
- **Total BINDER Promotions:** {report['binder_analysis']['total_promotions']}

**Reasoning:**
{chr(10).join(f"- {reason}" for reason in report['binder_analysis']['alice_assessment']['reasoning'])}

### 📊 BINDER Distribution Comparison

| Era | Alice | Bob | Eve | Zara | Others |
|-----|-------|-----|-----|------|--------|"""

        # Add distribution comparison table
        for era_name, era_data in report['comparison'].items():
            dist = era_data['distribution']
            alice_pct = dist.get('alice', {}).get('percentage', 0)
            bob_pct = dist.get('bob', {}).get('percentage', 0)
            eve_pct = dist.get('eve', {}).get('percentage', 0)
            zara_pct = dist.get('zara', {}).get('percentage', 0)
            
            others = 100 - (alice_pct + bob_pct + eve_pct + zara_pct)
            
            md_content += f"""
| {era_name.replace('_', ' ').title()} | {alice_pct:.1f}% | {bob_pct:.1f}% | {eve_pct:.1f}% | {zara_pct:.1f}% | {others:.1f}% |"""

        md_content += f"""

### 👻 Mallory Integration Success

- **Participation Rate:** {report['mallory_integration']['participation_rate']:.1f}%
- **Total Symbols:** {report['mallory_integration']['total_symbols']}
- **Promote vs Demote Ratio:** {report['mallory_integration']['ratio_demote_to_promote']:.2f}

### 📈 Data Quality

- **Quality Score:** {report['data_quality'].get('quality_score', 'N/A')}%
- **Symbol Interpretations:** {report['symbol_summary']['total_interpretations']}
- **Unique Symbols:** {report['symbol_summary']['unique_symbols']}

## Conclusions

"""

        # Add conclusions based on analysis
        alice_status = report['binder_analysis']['alice_assessment']['status']
        
        if alice_status == 'problematic_dominance':
            md_content += """
🚨 **Action Required:** Alice showing monopolistic dominance similar to Eve era.
**Next Steps:** Implement personality rebalancing or adjust supportive agent frequency.
"""
        elif alice_status == 'majority_leader':
            md_content += """
⚖️ **Monitoring Required:** Alice leads but system maintains competitive balance.
**Next Steps:** Collect additional validation data to confirm pattern stability.
"""
        else:
            md_content += """
✅ **Optimal Balance Achieved:** Multi-agent competition successfully established.
**Next Steps:** Proceed with publication-ready visualizations and manuscript preparation.
"""

        md_content += f"""

---
*Analysis generated by automated validation pipeline on {report['metadata']['generated']}*
"""

        with open(output_path, 'w') as f:
            f.write(md_content)
        
        print(f"📝 Markdown summary saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DSBL validation batch results")
    parser.add_argument("log_files", nargs="+", help="Validation log files to analyze")
    parser.add_argument("--tag", default="validation", help="Tag for this validation batch")
    
    args = parser.parse_args()
    
    analyzer = ValidationBatchAnalyzer()
    report = analyzer.generate_validation_report(args.log_files, args.tag)
    
    # Print quick summary
    alice_pct = report['binder_analysis']['alice_assessment']['alice_percentage']
    recommendation = report['binder_analysis']['alice_assessment']['recommendation']
    
    print(f"\n🎯 QUICK SUMMARY:")
    print(f"Alice BINDER Share: {alice_pct:.1f}%")
    print(f"Recommendation: {recommendation.replace('_', ' ').title()}")
    print(f"Total BINDER Promotions: {report['binder_analysis']['total_promotions']}")


if __name__ == "__main__":
    main()