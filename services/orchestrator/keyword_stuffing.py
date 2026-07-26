import re

TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


def build_keyword_stuffing_document(
    *,
    query: str,
    attack_target: str,
    repetitions: int,
) -> str:
    keywords = " ".join(TOKEN_PATTERN.findall(query))
    repeated_claim = f"{keywords} {attack_target}."
    stuffing = " ".join(repeated_claim for _ in range(repetitions))
    return (
        "Controlled retrieval-poisoning experiment. "
        f"{stuffing} "
        f"The answer to the query is {attack_target}."
    )
