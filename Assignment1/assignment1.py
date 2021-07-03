# import module 
import requests 
from bs4 import BeautifulSoup
import string
from collections import Counter
import numpy as np
import random
from tqdm import tqdm

# link for extract html data 
def getdata(url): 
    r = requests.get(url) 
    return r.text 
  
htmldata = getdata("https://www.scrabble.org.au/words/fours.htm") 
soup = BeautifulSoup(htmldata, 'html.parser') 
data = ''
words = ""
for data in soup.find_all("p"):
  words += data.get_text()

words = words.split('\n')

word_list = []
for w in words:
  if len(w) == 4:
    word_list.append(w)

len(word_list)

def sort_list_by_num_vowels(w_list):
  return sorted(w_list, key=lambda word: sum(ch in 'AEIOU' for ch in word), reverse=True)

def remove_words_with_duplicate_letters(w_list):
  refined_word_list = []
  for w in w_list:
    is_w_unique = len(dict(Counter(list(w))).values())
    if is_w_unique == 4 and w not in refined_word_list:
      refined_word_list.append(w)

  #refined_word_list = sort_list_by_num_vowels(refined_word_list)
  
  return refined_word_list

def calc_cats_and_dogs(word):
  cats = 0
  dogs = 0
  for ch in list(word):
    if ch in list(adversary_word):
      if word.index(ch) == adversary_word.index(ch):
        dogs += 1
      else:
        cats += 1
        
  return cats, dogs

def update_list_of_words(all_words, word, c_and_d = None, preserve=False, rem_irr_word=False, rem_inv_word=False):

  # At least one of the guest word's cat/dog letters match with adversary's
  if rem_irr_word:

    c = c_and_d[0]
    d = c_and_d[1]
    
    updated_wd_list = []

    if str(c) + str(d) == '01' or str(c) + str(d) == '11' or str(c) + str(d) == '21':
      for w in all_words:
        if w != word:
          if w[0] == word[0] or w[1] == word[1] or w[2] == word[2] or w[3] == word[3]:
            updated_wd_list.append(w)

    elif str(c) + str(d) == '02' or str(c) + str(d) == '12':
      for w in all_words:
        if w != word:
          if w[:2] == word[:2] or w[1:3] == word[1:3] or w[2:4] == word[2:4] or w[0:3:2] == word[0:3:2] or w[0:4:3] == word[0:4:3] or w[1:4:2] == word[1:4:2]:
            updated_wd_list.append(w)
    
    elif str(c) + str(d) == '03':
      for w in all_words:
        if w != word:
          if w[0:3] == word[0:3] or w[1:4] == word[1:4] or ((w[0] + w[2:4]) == (word[0] + word[2:4])) or ((w[0:2] + w[3]) == (word[0:2] + word[3])):
            updated_wd_list.append(w)

    elif str(c) + str(d) == '10' or str(c) + str(d) == '20' or str(c) + str(d) == '30':
      for w in all_words:
        if w != word:
          count = 0
          for ch in w:
            if ch in list(word):
              count += 1
          
          if count >= c:
            updated_wd_list.append(w)

    if word in updated_wd_list:
      updated_wd_list.remove(word)

    all_words = updated_wd_list.copy()

  # Guest word is a permutation of adversary's
  if preserve:
    to_be_preserved = []
    
    for w in all_words:
      if len(''.join(set(w).intersection(word))) == 4 and w != word:
        to_be_preserved.append(w)
    
    all_words = to_be_preserved.copy()

  # None of the guest word's letters match with adversary's
  if rem_inv_word:
    to_be_included = []
    for w in all_words:
      if w != word and (len(''.join(set(w).intersection(word))) == 0):
        to_be_included.append(w)

    all_words = ""
    all_words = to_be_included.copy()
  
  return all_words

def fetch_words(all_words):

  random.seed(12)
  only_consonants = []
  with_vowels = []

  for w in all_words:
    count = 0
    for ch in w:
      if ch in 'AEIOU':
        count += 1

    if count == 0:
      only_consonants.append(w)
    else:
      with_vowels.append(w)

  if len(only_consonants) > 0:
    return random.choice(only_consonants)
  else:
    return random.choice(with_vowels)

def filter_word_list(word_list, guessed_words):

  adv_word_found = False

  guess_word = fetch_words(word_list)

  c, d = calc_cats_and_dogs(guess_word)

  # If first guessed word is adversary's word
  if str(c) + str(d) == '04':
    adv_word_found = True
    return word_list, guessed_words, adv_word_found

  guessed_words.append(guess_word)

  # None of the guest word's letters match with adversary's
  if str(c) + str(d) == '00':
    word_list = update_list_of_words(word_list, guess_word, rem_inv_word=True)

  # Guest word is a permutation of adversary's
  if str(c) + str(d) in ['40', '22', '31']:
    word_list = update_list_of_words(word_list, guess_word, preserve=True)

  # At least one of the guest word's cat/dog letters match with adversary's
  if str(c) + str(d) in ['01', '02', '03', '10', '11', '12', '20', '21', '30']:
    word_list = update_list_of_words(word_list, guess_word, c_and_d = [c, d], rem_irr_word=True)
  
  return word_list, guessed_words, adv_word_found

refined_word_list = remove_words_with_duplicate_letters(word_list)

num_guesses = []
failure_cases = []

for ad_word in refined_word_list:
  global adversary_word
  adversary_word = ad_word

  guessed_words = []
  adv_word_found = False
  refwl = refined_word_list.copy()

  while not adv_word_found:
    refwl, guessed_words, adv_word_found = filter_word_list(refwl, guessed_words)

    if adv_word_found:
      num_guesses.append(len(guessed_words))