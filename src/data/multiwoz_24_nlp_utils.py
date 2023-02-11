"""Reference: https://github.com/smartyfh/MultiWOZ2.4/blob/main/create_data.py"""
import re
from pathlib import Path


def get_replacements(raw_data_path: Path):
    with (raw_data_path / "mapping.pair").open() as fin:
        replacements = []
        for line in fin.readlines():
            tok_from, tok_to = line.replace("\n", "").split("\t")
            replacements.append((" " + tok_from + " ", " " + tok_to + " "))

    return replacements


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match("[0-9]", text[sidx - 1]) and re.match("[0-9]", text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != " ":
            text = text[:sidx] + " " + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != " ":
            text = text[: sidx + 1] + " " + text[sidx + 1 :]
        sidx += 1
    return text


def normalize(text, replacements, clean_value=False):
    digitpat = re.compile("\d+")
    timepat = re.compile("\d{1,2}[:]\d{1,2}")
    pricepat = re.compile("\d{1,3}[.]\d{1,2}")

    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r"^\s*|\s*$", "", text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)
    text = re.sub(r"guesthouse", "guest house", text)

    if clean_value:
        # normalize phone number
        ms = re.findall("\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})", text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == "(":
                    sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], "".join(m))

        # normalize postcode
        ms = re.findall(
            "([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})", text
        )
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub("[,\. ]", "", m) + text[eidx:]

    # weird unicode bug
    text = re.sub("(\u2018|\u2019)", "'", text)

    if clean_value:
        # replace time and and price
        text = re.sub(timepat, " [value_time] ", text)
        text = re.sub(pricepat, " [value_price] ", text)
        # text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(";", ",")
    text = re.sub("$\/", "", text)
    text = text.replace("/", " and ")

    # replace other special characters
    text = text.replace("-", " ")
    text = re.sub('["\<>@\(\)]', "", text)  # remove

    # insert white space before and after tokens:
    for token in ["?", ".", ",", "!"]:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace("'s", text)

    # replace it's, does't, you'd ... etc
    text = re.sub("^'", "", text)
    text = re.sub("'$", "", text)
    text = re.sub("'\s", " ", text)
    text = re.sub("\s'", " ", text)
    for fromx, tox in replacements:
        text = " " + text + " "
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(" +", " ", text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match("^\d+$", tokens[i]) and re.match("\d+$", tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = " ".join(tokens)

    return text
