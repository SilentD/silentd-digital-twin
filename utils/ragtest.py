import spacy

# Paste the literal path to the model's directory folder containing 'config.cfg'
# (Make sure to prefix your string with 'r' to prevent Windows backslash escaping)
_nlp = spacy.load("../.venv/lib/python3.12/site-packages/en_core_web_lg/en_core_web_lg-3.8.0/")
print("done")