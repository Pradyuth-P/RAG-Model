import re

class PIIMasker:
    # Email pattern
    EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
    # Phone pattern: matches standard US formats and generic global formats
    PHONE_REGEX = re.compile(r'\b\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    # SSN pattern: matches US social security numbers
    SSN_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

    def mask(self, text: str) -> str:
        if not text:
            return text
        text = self.EMAIL_REGEX.sub("[EMAIL]", text)
        text = self.PHONE_REGEX.sub("[PHONE]", text)
        text = self.SSN_REGEX.sub("[SSN]", text)
        return text
