import re
import unicodedata


AMBIGUOUS_TERMS = {
    "en": {"operator", "interconnection", "subscriber", "service provider"},
    "hy": {"օպերատոր", "օպերատորը", "փոխկապակցում", "փոխկապակցումը",
           "բաժանորդ", "բաժանորդը", "ծառայություններ մատուցող", "ծառայություններ մատուցողը"},
}
META_PHRASES = {
    "en": {
        "who are you", "what are you", "what can you do", "what can i ask you",
        "which law do you use", "what law do you answer from",
    },
    "hy": {
        "ով ես դու", "ինչ ես դու", "ինչ կարող ես անել", "ինչ հարցեր կարող եմ տալ",
        "ինչ հարցեր կարող եմ քեզ տալ", "որ օրենքի հիման վրա ես պատասխանում",
        "որ օրենքն ես օգտագործում",
    },
}
DESCRIPTIONS = {
    "en": "I am an internal legal assistant for questions about the Law of the Republic of Armenia on Electronic Communications. You can ask questions in Armenian or English. Legal answers are based on the law and include article citations.",
    "hy": "Ես ներքին իրավական օգնական եմ՝ Հայաստանի Հանրապետության էլեկտրոնային հաղորդակցության մասին օրենքի վերաբերյալ հարցերի համար։ Կարող եք հարցեր տալ հայերեն կամ անգլերեն։ Իրավական պատասխանները հիմնվում են օրենքի վրա և ներառում են հոդվածների հղումներ։",
}


def question_language(question):
    return "hy" if any("\u0531" <= char <= "\u0586" for char in question) else "en"


def meta_answer(question):
    language = question_language(question)
    normalized = question.casefold().strip()
    # Armenian question and emphasis marks occur inside words.
    normalized = normalized.translate(str.maketrans("", "", "՚՛՜՞"))
    while normalized and unicodedata.category(normalized[0]).startswith("P"):
        normalized = normalized[1:].lstrip()
    while normalized and unicodedata.category(normalized[-1]).startswith("P"):
        normalized = normalized[:-1].rstrip()
    normalized = " ".join(normalized.split())
    return DESCRIPTIONS[language] if normalized in META_PHRASES[language] else None


def retrieval_query(question):
    query = " ".join(question.split())
    language = question_language(query)
    normalized = query.casefold().translate(str.maketrans("", "", "՚՛՜՞"))
    normalized = " ".join("".join(
        " " if unicodedata.category(char).startswith("P") else char for char in normalized
    ).split())
    if len(normalized.split()) > 6:
        return query
    if language == "en":
        match = re.fullmatch(r"(?:(?:who|what) is|define) (?:a |an |the )?(.+)", normalized)
        if match is None:
            match = re.fullmatch(r"what does (?:a |an |the )?(.+) mean", normalized)
    else:
        match = re.fullmatch(r"(?:ով է|ինչ է նշանակում|ինչ է) (.+)", normalized)
    if match is None or match.group(1) not in AMBIGUOUS_TERMS[language]:
        return query
    while query and unicodedata.category(query[-1]).startswith("P"):
        query = query[:-1].rstrip()
    suffix = " in electronic communications?" if language == "en" else " էլեկտրոնային հաղորդակցության ոլորտում։"
    return query + suffix
