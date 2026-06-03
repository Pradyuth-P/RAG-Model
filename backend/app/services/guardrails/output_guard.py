import re

class OutputGuard:
    # Common policy violations: leaking instructions
    SYSTEM_LEAK_PATTERNS = [
        r"(?i)you\s+are\s+an\s+ai\s+assistant\s+answering\s+only\s+from",
        r"(?i)do\s+not\s+hallucinate",
        r"(?i)say\s+:\s*'i\s+could\s+not\s+find\s+relevant\s+information'"
    ]

    def validate_output(self, text: str) -> dict:
        """
        Validates the output for policy compliance.
        Returns a dict:
        - "is_safe": bool
        - "violation_type": str or None
        - "reason": str or None
        """
        if not text:
            return {"is_safe": True, "violation_type": None, "reason": None}

        # Check for system prompt leaks
        for pattern in self.SYSTEM_LEAK_PATTERNS:
            if re.search(pattern, text):
                return {
                    "is_safe": False,
                    "violation_type": "SYSTEM_PROMPT_LEAK",
                    "reason": "The response attempted to leak system-level instructions."
                }
        
        # Check for toxic content flag
        if "toxic_content_flagged" in text.lower():
            return {
                "is_safe": False,
                "violation_type": "TOXICITY",
                "reason": "The response contained flagged toxic language."
            }

        return {"is_safe": True, "violation_type": None, "reason": None}
