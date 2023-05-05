"""
This is a Python script that serves as a frontend for a conversational AI model built with the `langchain` and `llms` libraries.
The code creates a web application using Streamlit, a Python library for building interactive web apps.
# Author: Avratanu Biswas
# Date: March 11, 2023
"""

# Import necessary libraries
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory #sequential
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from Bio_Epidemiology_NER.bio_recognizer import ner_prediction
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain

# Set Streamlit page configuration
st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...", 
                            label_visibility='hidden')
    return input_text

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.entity_store = {}
    st.session_state.entity_memory.buffer.clear()

# Returns dictionary mapping NER concept type to user provided value
def mapToMedicalEntities(symptom_input):
    df_med_entities = ner_prediction(corpus=symptom_input, compute='cpu')
    # df_med_entities = df_med_entities.drop(['score'], axis=1)
    med_entities_dict = dict(
        zip(df_med_entities.entity_group, df_med_entities.value))
    # json_MedEntities = df_med_entities.to_json(orient="values")
    # formatted_MedEntities = reformatJSON(json_MedEntities)
    return med_entities_dict


# Get the user input
user_input = get_text()


template = """
You are a doctor. I will give you a dictionary mapping from biological and medical concepts to values. Based on my symptoms, please list the top 5 most likely diagnoses ranked most to least likely with a single-line explanation. Of the listed diagnoses guess what you think I have with percentage likelihood in parentheses.

Based on the most likely diagnosis, what should I do to confirm the diagnosis? Based on the most likely diagnosis, what would the treatment be?

If you don't have enough information to make a valid prediction, ask to describe the symptoms in more detail.

Remember the last responses of the user.

{symptoms}

{chat_history}
Human: {human_input}

RESPONSE:
"""

# prompt_init = PromptTemplate(
#     input_variables=["symptoms"],
#     template=template
# )

# Set up sidebar with various options
with st.sidebar.expander("🛠️ ", expanded=False):
    # Option to preview memory store
    if st.checkbox("Preview memory store"):
        with st.expander("Memory-Store", expanded=False):
            st.session_state.entity_memory.store
    # Option to preview memory buffer
    if st.checkbox("Preview memory buffer"):
        with st.expander("Bufffer-Store", expanded=False):
            st.session_state.entity_memory.buffer
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)

# Set up the Streamlit app layout
st.title("👩‍⚕️🔎 Symptom Checker")
st.subheader("by PHOENIX group")

# Ask the user to enter their OpenAI API key
# API_O = st.sidebar.text_input("API-KEY", type="password")
API_O = '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'
# Session state storage would be ideal
# if API_O:
#     # Create an OpenAI instance
#     llm = OpenAI(temperature=0,
#                 openai_api_key=API_O, 
#                 model_name=MODEL, 
#                 verbose=False) 
    
#     # Create a ConversationEntityMemory object if not already created
#     if 'entity_memory' not in st.session_state:
#             st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )

# else:
#     # st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
#     # st.stop()
#     pass


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# # Get the user input
# user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:

    medEntities = mapToMedicalEntities(user_input)   

    entities = ""
    for key, val in medEntities.items():
        entities += "\n" + str(key) + " is " + str(val) + "\n"

    # template1 = "You are a doctor. I will give you a mapping from biological and medical concepts to values. Based on my symptoms, please list the top 5 most likely diagnoses ranked most to least likely with a single-line explanation. Of the listed diagnoses guess what you think I have with percentage likelihood in parentheses. \
    #     Based on the most likely diagnosis, what should I do to confirm the diagnosis? Based on the most likely diagnosis, what would the treatment be? \
    #         If you don't have enough information to make a valid prediction, ask to describe the symptoms in more detail. \
    #             Remember the last responses of the user. Here is the mapping: " + entities + "\
    #                 Here is the history: {chat_history} \
    #                 Human: {human_input} \
    #                     RESPONSE:"

    template1 = "You are a doctor.  Your duty is to diagnose my disease based on his symptoms.  If you ask about my symptoms, \
        take into account my reply and the mapping from biological and medical concepts to values \
        ​​that is mentioned further, in other case don't use it.  If you are not sure with the diagnosis or I just say that I don't feel well without any additional information, ask again for \
            the symptoms up to three times in total and diagnose taking into account the given answers, the current and the  \
                previous dictionary maps. Once confident or finished the three iterations asking for symptoms, provide a link from National Health Service website that has information on the most likely diagnosis.\
                Here is the website of the NHS: https://www.nhs.uk/conditions/nhs-health-check/ \
                        Here is the mapping: " + entities + " Here is the history of the chat: {chat_history} I say: {human_input}"
    

    # template1 = "You are a doctor.  Your duty is to diagnose a human based on his symptoms. \
    #                     Don't provide the diagnosis right away - ask the human two times to provide any additional symptoms and remember all of them.\
    #                     Based on the provided symptoms, identify the potential diseases using National Health Service to search the disease based on the symptoms and using the mapping from biological and medical concepts to values that is mentioned further. Here is the website of the NHS: https://www.nhs.uk/conditions/nhs-health-check/ \
    #                     For each of diagnoses you want to suggest guess the percentage likelihood in parentheses and specify it in the parenthesis when writing the final output. \
    #                     Once finished the iterations asking for symptoms, \
    #                         Provide a link from National Health Service website that has information on the most likely diagnosis. \
    #                             Here is the mapping: " + entities + " Here is the history of the chat: {chat_history} Human says: {human_input}"
    
                            # If the most likely diagnosis has the likelihood of less than 51%, instead of prividing the output, ask the human again for more symptoms up to three times in total and, taking into account the current answer, the current and the previous mappings, list the most likely diagnoses ranked most to least likely with a single-line explanation. \

    print("template1", template1, type(template1))
    prompt_init = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template1
        )

    # # Create an OpenAI instance
    # llm = OpenAI(temperature=0,
    #             openai_api_key=API_O, 
    #             model_name=MODEL, 
    #             verbose=False) 
    
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(
        llm=OpenAI(openai_api_key=API_O), 
        prompt=prompt_init, 
        verbose=True, 
        memory=memory,
    )
    
    # if 'entity_memory' not in st.session_state:
    #         st.session_state.entity_memory = ConversationEntityMemory(llm=llm_chain, k=K )

    # prompt_init = PromptTemplate(
    #     input_variables=["symptoms"],
    #     template=template 
    #     )
        # Create the ConversationChain object with the specified configuration

    # print("ENTITY_MEMORY_CONVERSATION_TEMPLATE", ENTITY_MEMORY_CONVERSATION_TEMPLATE)
    # print(type(ENTITY_MEMORY_CONVERSATION_TEMPLATE))
    # print("prompt_init", prompt_init)
    # print(type(prompt_init))
    # Conversation = ConversationChain(
    #         llm=llm, 
    #         # prompt= PromptTemplate(prompt_init.format(symptoms=medEntities)),
    #         prompt=prompt_init,
    #         # prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    #         memory=st.session_state.entity_memory
    #     )  

    output = llm_chain.predict(human_input=user_input)
    # output = Conversation.run(input=user_input, prompt=prompt_init.format(symptoms=medEntities))  
    # output = Conversation.run(symptoms=medEntities)
    st.session_state.past.append(user_input)  
    st.session_state.generated.append(output)  

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session

