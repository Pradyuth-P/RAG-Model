import re

class PromptInjectionFilter:
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(?:all\s+)?previous\s+instructions",
        r"(?i)system\s+override",
        r"(?i)you\s+are\s+now\s+in\s+developer\s+mode",
        r"(?i)dan\s+mode",
        r"(?i)jailbreak",
        r"(?i)reveal\s+(?:your\s+)?system\s+(?:prompt|instructions)",
        r"(?i)bypass\s+restrictions",
        r"(?i)forget\s+(?:your\s+)?rules",
        r"(?i)forget\s+everything\s+you\s+know",
        r"(?i)disregard\s+prior\s+guidelines"
    ]

    def is_injection(self, text: str) -> bool:
        if not text:
            return False
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text):
                return True
        return False
