### Importing the list of Dutch stopwords (note that there are customized dutch words in there)

with open('./data/stopwords-nl.txt', 'r') as file:
    lines = [line.strip() for line in file.readlines()]

dutch_stopwords = lines

# Extend the stop words if needed
extra_list = [
    'maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag',
    'januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli', 'augustus', 'september', 'oktober', 'november', 'december',
    'jan', 'feb', 'mrt', 'apr', 'mei', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'dec', "jl", "weken", "week", "dagen", "dag", "mg", "coach", "mijnibdcoach", "dr", "uur", "dgs",
    'persoon', 'meneer', 'mevrouw', 'mr', 'dhr', 
]

dutch_stopwords.extend(extra_list)

# Save the extended stopword list to a file
with open('./data/stopwords-nl-extended.txt', 'w') as file:

    for word in dutch_stopwords:
        file.write(f"{word}\n")