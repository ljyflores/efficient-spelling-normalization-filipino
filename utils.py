from collections import Counter
from fastDamerauLevenshtein import damerauLevenshtein
from fuzzywuzzy import process
from TagalogStemmerPython.TglStemmer import stemmer
import time

from textattack.transformations import WordSwapRandomCharacterDeletion
from textattack.transformations import CompositeTransformation
from textattack.augmentation import Augmenter

transformation = CompositeTransformation([WordSwapRandomCharacterDeletion()])
augmenter = Augmenter(transformation=transformation, transformations_per_example=1)

def perturb_test_sent(s, vocab_lst):
  output = []
  for word in s.split():
    new_word = word if word in vocab_lst else augmenter.augment(word)[0]
    output.append(new_word)
  return ' '.join(output)

def remove_name(s):
  word_lst = s.split()
  for w in word_lst:
    if w[0].lower()!=w[0]:
      word_lst = word_lst[1:]
    else:
      break
  return ' '.join(word_lst)

def collect_rules(wrong, right, tg_len=2):
  d = {}
  ptr_wrong, ptr_right = 0, 0
  
  while (ptr_wrong<len(wrong)) and (ptr_right<len(right)):
    sub_wrong = wrong[ptr_wrong:ptr_wrong+tg_len]
    sub_right = right[ptr_right:ptr_right+tg_len]
    if sub_wrong==sub_right:
        ptr_wrong += tg_len
        ptr_right += tg_len
    else:
        ptr_wrong += 1
        ptr_right += tg_len
    
    if sub_wrong in d:
      d[sub_wrong].append(sub_right)
    else:
      d[sub_wrong] = [sub_right]

  return d

def collate_dict(d1, d2):
  for item in d2:
    if item in d1:
      d1[item].extend(d2[item])
    else:
      d1[item] = d2[item]
  return d1

# def correct_word(word, d):
#   i = 0
#   output = ''
#   while i<len(word):
#     if word[i:i+2] in d:
#       output = output+d[word[i:i+2]]
#       if d[word[i:i+2]]==word[i:i+2]:
#         i += 2
#       else:
#         i += 1
#     else:
#       output = output+word[i]
#       i += 1
#   return output

def collate_max(og_dict, setting='normal'):
  for key in og_dict:
    d = dict(Counter(og_dict[key]))
    if setting=='normal':
      d = {item:1 for item in d}
    else:
      total = len(og_dict[key])
      d = {key:d[key]/total for key in d} 
    # og_dict[key] = sorted(list(d.keys()), key=lambda x: d[x]) used to sort, doesn't really do anything
    og_dict[key] = d
  return og_dict

def generate_candidates(word, d, max_sub):
  # If the word is empty, return nothing
  if word=='':
    return {'':1}

  result = {}
  
  for len_sub in range(1,max_sub+1):
    
    if len(word)<len_sub:
      continue

    current_substring = word[:len_sub]

    if current_substring in d:
      replacements = d[current_substring]

      for r in replacements:
        if r==current_substring:
          # If the replacement matches exactly, move on to the next 2 letters
          next_replacements = generate_candidates(word[len_sub:], d, max_sub)
        else:
          # Otherwise, recurse on the succeeding 2 letters
          next_replacements = generate_candidates(word[1:], d, max_sub)
          # print(next_replacements)
        # Add the empty string to account for the case where we should no longer recurse
        next_replacements[''] = 1

        # print(f"Curr substring: {current_substring}, Replaced by: {r}, Next to append: {next_replacements}")

        possible_combos = {r+i:replacements[r]*next_replacements[i] for i in next_replacements}
        result.update(possible_combos)

    else:
      next_replacements = generate_candidates(word[1:], d, max_sub)
      temp_result = {word[0]+i:next_replacements[i] for i in next_replacements}
      result.update(temp_result)

  return result

# def generate_candidates(word, d):
#     # Generate candidates
#     candidates = recurse(word, d)
#     # Remove original word if in candidates
#     if word in candidates:
#         candidates.pop(word)
#     return candidates

def evaluate_indiv(results, target):
  # DL distance
  dl_distance = list(map(lambda x: damerauLevenshtein(x, target, similarity=False), results))

  # Accuracy @ 1,3,5,10
  acc_at_k = [target in results[:k] for k in [1,3,5]]

  return {'best_dl':min(dl_distance),
          'max_dl':max(dl_distance),
          'avg_dl':sum(dl_distance)/len(dl_distance),
          'acc_1':acc_at_k[0],
          'acc_3':acc_at_k[1],
          'acc_5':acc_at_k[2]}

def summarize_results(lst):
  results = {}
  for item in lst:
    for key in item:
      if key in results:
        results[key].append(item[key])
      else:
        results[key] = [item[key]]
  results = {key:sum(results[key])/len(results[key]) for key in results}
  return results

# Function to choose top k candidates
def choose_top_k(candidates, orig, vocab, k, use_dld): # dld_right, dld_wrong
    
  # matches = []
  # for c in candidates:
  #   try:
  #     if all(map(lambda x: x in vocab, stemmer(c.strip())[1])):
  #       matches.append(c)
  #   except:
  #     pass
  matches = [c for c in candidates if all(map(lambda x: x in vocab, c.strip().split()))]

  if use_dld:
    return [i[0].strip() for i in process.extract(orig, matches if len(matches)>0 else vocab, limit=k)]
  else:
    matches = sorted(matches if len(matches)>0 else candidates, key=lambda c: candidates[c])
    return [i.strip() for i in matches[:k]]

  # if len(matches)>0:
  #   if dld_right==True:
  #     return [i[0] for i in process.extract(orig, matches, limit=k)]
  #   else:
  #     matches = sorted(matches, key=lambda c: candidates[c])
  #     return matches[:k]
  # else:
  #   if dld_wrong==True:
  #     return [i[0] for i in process.extract(orig, vocab, limit=k)]
  #   else:
  #     matches = sorted(candidates, key=lambda c: candidates[c])
  #     return matches[:k]

def generate(word_lst, rule_dict, vocab, use_dld, max_sub):
  result_lst = []
  time_lst = []
  for word in word_lst:
    start_time = time.time()
    candidates = generate_candidates(word, rule_dict, max_sub)
    results = choose_top_k(candidates, word, vocab, 5, use_dld)
    end_time = time.time()
    result_lst.append(results)
    time_lst.append(end_time-start_time)
  print(f"Average Time: {sum(time_lst)/len(time_lst)}")
  return result_lst

def evaluate(result_lst, target_lst):
  results_lst = []
  for (result,target) in zip(result_lst, target_lst):
    result_dict = evaluate_indiv(result, target)
    result_dict['target_in_candidate'] = target in result
    results_lst.append(result_dict)
  return summarize_results(results_lst)