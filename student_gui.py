# @app.route('/api/v1/ask', methods=['post'])
# def api_id():
#     # implement the solve task
#     context = request.json['context']
#     goals = request.json['goals']

#     response = agent.run(template.format(problem=context, need_to_know=goals))
#     return personify(response)
import requests
def get_answer(context, goals):
    url = 'http://localhost:5000/api/v1/ask'
    myobj = {'context': context, 'goals': goals}
    x = requests.post(url, json = myobj)
    x_json = x.json()
    try:
        return (x_json['response'], x_json['intermediate_steps'])
    except:
        return (x_json['response'], ["Please check the console"])


# @app.route('/api/v1/search', methods=['post'])
# def api_search():
#     query = request.json['query']
#     response = search_notes(query)
#     return jsonify({'response': response})
def search_notes(query):
    url = 'http://localhost:5000/api/v1/search'
    myobj = {'query': query}
    x = requests.post(url, json = myobj)
    x_json = x.json()
    return x_json['response']

import streamlit as st

st.title("StraightAGPT")


st.header("Search for notes")
# get search query from user
query = st.text_input("Search query")
# button to submit the search query
submit_button = st.button("Search")
# if the button is pressed
if submit_button:
    # show a loading indicator
    with st.spinner("Searching..."):
        response = search_notes(query)
    # display the answer streamed from the model
    st.write(response)


st.header("Solve Problems")
# get context input from user (large text box)
context = st.text_area("Context", height=200)

# get question input from user (single line text box)
question = st.text_input("Question")

# button to submit the question
submit_button = st.button("Answer")

# if the button is pressed
if submit_button:
    # show a loading indicator
    with st.spinner("Thinking..."):
        response = get_answer(context, question)
        steps = response[1]
        response = response[0]
    # display the answer streamed from the model
    st.write(response)
    for step in steps:
        st.write(step)
    # show a button to add this response to the context
    add_to_context = st.button("Add to context")
    if add_to_context:
        context_new = f"{question}: {response}"
        # update the context text area with the new context
        context = st.text_area("Context", context_new, height=200)
