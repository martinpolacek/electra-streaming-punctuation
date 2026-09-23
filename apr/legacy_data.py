"""Archived sentence preprocessing and block construction (see docs/PROVENANCE.md)."""
import os, re, random, logging
from sklearn.model_selection import train_test_split
logger = logging.getLogger(__name__)

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = r"(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

# Multilingualni character class - diakritika cs, sk, no, swe, de, en + cyrilice
# Povolujeme vsechna unicode pismena (\w) namisto explicitnich znaku
MULTILINGUAL_FILTER = re.compile(r"[^\w.?, ]", re.UNICODE)


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub(r"\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "\u201d" in text: text = text.replace(".\u201d","\u201d.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = re.sub(r"(?<=[^0-9])(\.)",".<stop>", text)
    text = re.sub(r"(?<=[^0-9])(\?)","?<stop>", text)
    text = re.sub(r"(?<=[^0-9])(\!)","!<stop>", text)
    text = text.replace("<prd>",".")
    text = text.replace(":","")
    text = text.replace("#","")
    text = text.replace("$","")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def generate_dataset(folder, validation_split, shortcuts_path=None):
    lines = []
    zkratky = []

    if shortcuts_path and os.path.exists(shortcuts_path):
        with open(shortcuts_path, "r", encoding="utf-8") as f:
            zkratky = f.read().split("\n")
        zkratky = sorted(set(zkratky))

    for file in sorted(os.listdir(folder)):
        if file.endswith(".utf8"):
            with open(os.path.join(folder, file), "r", encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.replace("***", "")
                    line = line.replace("  ", " ")
                    line = line.replace("   ", " ")

                    pattern = r"(\d+)\.\s(godina|godine|godinu|godini|godinom)"
                    replacement = r"\1 \2"
                    line = re.sub(pattern, replacement, line)

                    line = split_into_sentences(line)
                    line = list((map(lambda x: x.lower(), line)))

                    for sent in line:
                        for z in zkratky:
                            if z:
                                sent = sent.replace(" " + z, " " + z[:-1] if z else "")
                        sent = sent.replace("?", " ?")
                        sent = sent.replace(".", " .")
                        sent = sent.replace(",", " ,")
                        sent = sent.replace("!", " .")
                        out = MULTILINGUAL_FILTER.sub('', sent)
                        out = out.replace("  ", " ")

                        if len(out.split()) > 0 and len(out) >= 7:
                            lines.append(out)
            logger.warning("Loaded file: " + str(file))

    X_train, X_test, y_train, y_test = train_test_split(lines, lines, test_size=float(validation_split), random_state=0)
    return y_train, y_test


def generate_random_lenghts(lines):
    # Keep the trainer''s process-wide RNG untouched. Resetting it here made
    # --seed ineffective for later dataset sampling and epoch shuffling.
    rng = random.Random(2)
    len_lines = len(lines)
    count = 0
    rnd_nums = []
    while count < len_lines:
        val = rng.randint(1, 15)
        if count + val <= len_lines:
            rnd_nums.append(val)
            count += val
    return rnd_nums


def connect_sentences(lines, rnd_nums):
    lines_new = []
    curr_idx = 0
    for i in range(len(rnd_nums)):
        subseq = lines[curr_idx:curr_idx + rnd_nums[i]]
        curr_idx += rnd_nums[i]
        subseq = " ".join(subseq)
        n = random.randint(2, 6)
        if n != 3:
            subseq = subseq.rsplit(' ', n)[0]
        lines_new.append(subseq)
    return lines_new


