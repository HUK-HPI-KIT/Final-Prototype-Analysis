from random import randint

from deep_translator import GoogleTranslator

from pathlib import Path
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.corpus import wordnet
import json
from utils import check_paths
from tqdm import tqdm

def get_permutations(text):
    permutations = []
    words = text.split(" ")
    for idx, word in enumerate(words):
        word_type = nltk.pos_tag([word])[0][1]
        synonyms = wordnet.synsets(word)
        for synonym in synonyms:
            for syn_word in synonym.lemmas():
                syn_word_type = nltk.pos_tag([syn_word.name()])[0][1]
                if syn_word_type != word_type:
                    continue
                permutations.append(" ".join(words[:idx] + [syn_word.name()] + words[idx + 1:]))
    return list(set(permutations))

def generate_synthetic_dataset(original_dataset_path, output_path, translator, synthetic_path):
    synthetic_path = Path(synthetic_path)
    original_dataset_path = Path(original_dataset_path)
    check_paths(synthetic_path, original_dataset_path)
    with synthetic_path.open() as synthetic_rule_file:
        rules = json.load(synthetic_rule_file)
    with original_dataset_path.open() as original_dataset_file:
        orig_dataset = json.load(original_dataset_file)
    
    userAnswers = []
    print("generating synthetic data points...")
    for rule in rules:
        question = translator.translate(rule["question"])
        question_permutations = get_permutations(question)
        for answer in rule["answers"]:
            product_idx = list(orig_dataset["products"].keys()).index(answer[1][0])
            print(product_idx)
            recommendation_vector = [0] * len(orig_dataset["products"])
            recommendation_vector[product_idx] = answer[1][1]
            answer[0] = translator.translate(answer[0])
            answer_permutations = get_permutations(answer[0])
            for answer_permutation in answer_permutations:
                for question_permutation in question_permutations:
                    full_text = question_permutation + " " + answer_permutation + "."
                    userAnswers.append([[full_text], recommendation_vector, [product_idx]])
            print("dataset size:", len(userAnswers))
    
    print("merging ", len(userAnswers) // 4, "entries...")
    for _ in tqdm(range(len(userAnswers) // 4)):
        first = randint(0, len(userAnswers) - 1)
        second = randint(0, len(userAnswers) - 1)
        while second == first:
            second = randint(0, len(userAnswers) - 1)
        userAnswers[first][0] += userAnswers[second][0]
        print("ones before merge:", sum(userAnswers[first][1]), "and", sum(userAnswers[second][1]))
        for i in range(len(userAnswers[first][1])):
            if i in userAnswers[second][2]:
                userAnswers[first][1][i] = userAnswers[second][1][i]
        userAnswers[first][2] = list(set(userAnswers[first][2] + userAnswers[second][2]))
        print("ones after merge:", sum(userAnswers[first][1]))
        del userAnswers[second]
    print("final dataset length:", len(userAnswers))
    
    new_dataset = {}
    new_dataset["products"] = orig_dataset["products"]
    new_dataset["userAnswers"] = []
    new_dataset["recommendations"] = []
    for userAnswer in userAnswers:
        new_dataset["userAnswers"].append(userAnswer[0])
        new_dataset["recommendations"].append(userAnswer[1])
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(new_dataset, output_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    
    translator = GoogleTranslator(source='auto', target='en')
    generate_synthetic_dataset("dataset.json", "synth_dataset.json", translator, "synthetic_rules.json")