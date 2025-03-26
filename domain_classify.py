from langchain_ollama import OllamaLLM
import json
from tqdm import tqdm
import os
from collections import Counter


llm = OllamaLLM(model="gemma2:27b")

json_file = "./FakeSV_DATA_OPENSOURCE/data.json"


with open(json_file, "r", encoding="utf-8") as file:
    data = file.readlines()

domain = ["Society", "Health", "Disaster", "Culture", "Education", "Finance", "Politics", "Science", "Military"]
domain_files = {field: [] for field in domain}
domain_files["Unknown"] = []

for entry in tqdm(data, desc="Processing Entries", unit="entry", colour="cyan"):
        json_obj = json.loads(entry.strip())

        keywords = "keywords{" + json_obj.get("keywords", "") + "}"
        prompt = """
        Here are some fields: Society, Health, Disaster, Culture, Education, Finance, Politics, Science, Military. 
  
        Here are some Domain Choose Samples (For your reference): 
        Society: Cabbage sells for 38.60 RMB.
        Health: Cabbage is made of wax.
        Disaster: Shipwreck in Phuket, Thailand.
        Culture: The United Nations declares Chinese as a global universal language.
        Education: Elementary school students lead the police to catch illegal make-up classes.
        Finance: RMB ceases to be issued.
        Science: Clouds can predict earthquakes in advance.
        Military: US military intercepts Iranian missiles.
        Politics: Zhejiang Province sent 100,000 ducks to Pakistan to eradicate locusts.
        
        You can then compare and analyze the question with the examples above to choose the most suitable field.
        """

        question = f"Question: {keywords}. Please think this Question and choose the best-fitting field about this Question from the options above. Only output the one word from the list: Society, Health, Disaster, Culture, Education, Finance, Politics, Science, Military. You must only output  one word, which is enough; don't output anything else."
        input_prompt = prompt + "\n" + question + "\n" + "Domain:"

        output = llm.invoke(input_prompt).strip()
        matched_domains = [domain_name for domain_name in domain if domain_name.lower() in output.lower()]
        print("Gemma : ", output)
        if matched_domains:
            for matched_domain in matched_domains:
                print("Gemma Choose : ", matched_domain)
                domain_files[matched_domain].append(json_obj)
        else:
            again = "Please carefully consider and choose an appropriate field from (Society, Health, Disaster, Culture, Education, Finance, Politics, Science, Military)"
            output_again = llm.invoke(again).strip()
            again_matched_domains = [domain_name for domain_name in domain if domain_name.lower() in output.lower()]
            if again_matched_domains:
                for again_matched_domain in again_matched_domains:
                    print("Gemma Choose : ", again_matched_domain)
                    domain_files[again_matched_domain].append(json_obj)
            else:
                domain_files["Unknown"].append(json_obj)



os.makedirs("./FakeSV_Domain_output_Gemma", exist_ok=True)
for field, entries in domain_files.items():
    if entries:
        output_filename = f"./FakeSV_Domain_output_Gemma/{field}_classified_New.json"
        with open(output_filename, "w", encoding="utf-8") as output_file:
            for entry in entries:
                json.dump(entry, output_file, ensure_ascii=False, separators=(',', ':'))
                output_file.write("\n")

        print(f"Saved {len(entries)} entries to {output_filename}")
