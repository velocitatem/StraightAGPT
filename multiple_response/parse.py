

# we have the above options and we want to find the correct one
# The list contains n categories and each category has m options

# turn list into a list of lists of categories
# each category is a list of options


def options_to_categories(options):
    import spacy
    nlp = spacy.load("en_core_web_lg")
    def compare(sentence_a,sentence_b):
        a = nlp(sentence_a)
        b = nlp(sentence_b)
        return a.similarity(b)

    # compare first with 3rd
    categories = {
    }
    for i in range(len(options)):
        for j in range(len(options)):
            if i!=j:
                similarity = compare(options[i],options[j])
                if similarity > 0.9 and i not in categories.keys():
                    categories[i] = [j]
                elif similarity > 0.9 and i in categories.keys():
                    categories[i].append(j)
    pool = [list(set([key, *categories[key]])) for key in categories.keys()]
    # remove duplicate lists
    clean_pool = []
    for i in pool:
        if i not in clean_pool:
            clean_pool.append(i)
    pool = clean_pool
    # convert indices to options
    for i in range(len(pool)):
        for j in range(len(pool[i])):
            pool[i][j] = options[pool[i][j]]

    return pool

def highlight_selected_in_original(original, selection):
    for i in original:
        if i in selection:
            original[original.index(i)] = f"**{i}**"
    return original
