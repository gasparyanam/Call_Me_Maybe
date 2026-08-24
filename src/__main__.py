from llm_sdk import Small_LLM_Model
import json

def main() -> None:
    llm = Small_LLM_Model()
    s = "what is sum of 3 and 5?"
    input_id = llm.encode(s).tolist()[0]
    logits = llm.get_logits_from_input_ids(input_id)
    b = max(logits)
    logits = llm.get_logits_from_input_ids(input_id)
    max_l = max(logits)
    for i,j in enumerate(logits):
        if j == max_l:
            print(i)
    print(max_l, flush=True)
    word = llm.get_path_to_vocab_file()
    with open(word, encoding="utf-8") as file:
        token = file.read()
        d = token.split(",")
        o = []
        for i in d:
            o.append(i.split(":"))
        
        print(o)





main()
