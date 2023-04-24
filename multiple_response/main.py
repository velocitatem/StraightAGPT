import openai
import requests

def evaluate(context, goals):
    url = 'http://localhost:5000/api/v1/ask'
    myobj = {'context': context, 'goals': goals}
    x = requests.post(url, json = myobj)
    x_json = x.json()
    return (x_json['response'], x_json['intermediate_steps'])

def main():
    # load work.txt
    with open("work.txt", "r") as f:
        work = f.read()

    options = [
        'The corresponding sample mean cutoffs are 118.45 and 137.55',
        'The probability of a type II error is 0.0003',
        'The probability of a type II error is 0.3018',
        'The probability of a type II error is 0.6981',
        'The corresponding sample mean cutoffs are 120.45 and 127.56',
        'The corresponding sample mean cutoffs are 119.98 and 136.02',
    ]

    from parse import options_to_categories, highlight_selected_in_original

    categories = options_to_categories(options)
    selected = []
    for category in categories:
        category = '\n'.join([f"{i+1}. {option}" for i, option in enumerate(category)])
        res = evaluate(work, category)[0]
        res = int(''.join([c for c in res if c.isdigit()]))
        print("\t >>>>>>>> ", res)
        selected.append(category[res-1])

    highlighted = highlight_selected_in_original(options, selected)
    for opt in highlighted:
        print(opt)

if __name__ == "__main__":
    main()
