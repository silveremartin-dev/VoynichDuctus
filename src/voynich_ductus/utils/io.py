"""
Benchmark formatting and reporting utilities.
"""

from typing import Dict, Any


class BenchmarkFormatter:
    """
    Renders comparative benchmark diagnostics into Markdown tables and structured summaries.
    """

    @staticmethod
    def to_markdown_table(benchmark_results: Dict[str, Any]) -> str:
        """
        Converts multi-corpus benchmark output into a clean GitHub-flavored markdown table.
        """
        headers = [
            "Corpus / System",
            "Words",
            "H1 Char (bits)",
            "H2 Cond (bits)",
            "H2 Drop %",
            "Hurst (DFA)",
            "Gzip Ratio",
            "Markov Top-1 (O2)",
            "Top Hypothesis"
        ]

        rows = []
        for name, res in benchmark_results.items():
            ent = res["entropy"]
            mem = res["memory"]
            comp = res["compressibility"]
            mark = res["markov_automata"]["order_2"]
            diag = res["diagnostic_diagnosis"]

            rows.append([
                name,
                str(res["token_count"]),
                f"{ent['h1_char_bits']:.2f}",
                f"{ent['h2_cond_char_bits']:.2f}",
                f"{ent['h2_drop_ratio'] * 100:.1f}%",
                f"{mem['hurst_exponent']:.3f}",
                f"{comp['gzip_ratio']:.3f}",
                f"{mark['top1_prediction_accuracy']:.2f}",
                f"**{diag['top_hypothesis']}**"
            ])

        col_widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]

        def format_row(row_items):
            return "| " + " | ".join(item.ljust(col_widths[idx]) for idx, item in enumerate(row_items)) + " |"

        separator = "| " + " | ".join("-" * col_widths[idx] for idx in range(len(headers))) + " |"

        table_str = [format_row(headers), separator]
        for r in rows:
            table_str.append(format_row(r))

        return "\n".join(table_str)
