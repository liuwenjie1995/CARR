"""DeepSeek adapter for the pinned MemoryAgentBench Long-Context Agent."""

import hashlib
import runpy
from functools import lru_cache
from pathlib import Path

UPSTREAM = Path(__file__).with_name("MemoryAgentBench")
UPSTREAM_COMMIT = "538026089d1a8a8eff05121d0db89b388f360eba"
CHUNK_SIZE = 4096
MAX_TOKENS = 10


class LongContext:
    name = "memoryagentbench-long-context"

    def __init__(self, nltk_data: Path):
        import nltk

        nltk.data.path.insert(0, str(nltk_data))
        nltk.data.find("tokenizers/punkt_tab/english/")
        with (UPSTREAM / "utils" / "templates.py").open("rb") as handle:
            self.templates_sha256 = hashlib.sha256(handle.read()).hexdigest()
        self.templates = runpy.run_path(str(UPSTREAM / "utils" / "templates.py"))["get_template"]
        self.scoring = runpy.run_path(str(UPSTREAM / "utils" / "eval_other_utils.py"))

    @lru_cache(maxsize=8)
    def _memorize(self, source: str, context: str) -> str:
        template = self.templates(source, "memorize", "Long_context_agent")
        chunks = self.scoring["chunk_text_into_sentences"](context, chunk_size=CHUNK_SIZE)
        # Upstream inserts wall-clock timestamps. A fixed marker keeps the prefix
        # identical between evaluation runs and does not change fact order.
        return "\n".join(
            template.format(context=chunk, time_stamp="2000-01-01 00:00:00")
            for chunk in chunks
        ).strip()

    def messages(self, sample: dict, history: str) -> list[dict]:
        source = sample["source"]
        get_template = self.templates
        query = get_template(source, "query", "Long_context_agent").format(question=sample["question"])
        return [
            {"role": "system", "content": get_template(source, "system", "Long_context_agent")},
            {"role": "user", "content": self._memorize(source, history) + "\n" + query},
        ]

    def score(self, answer: str, aliases: list[str]) -> bool:
        parsed = self.scoring["parse_output"](answer)
        metric = self.scoring["substring_exact_match_score"]
        max_over_aliases = self.scoring["drqa_metric_max_over_ground_truths"]
        return any(max_over_aliases(metric, text, aliases) for text in [answer, parsed] if text is not None)
