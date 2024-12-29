from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

@cl.on_chat_start
async def on_chat_start():
    text_content = """Disclimar: the information provided is based on google reports and surveys, the information should not be missused
    type hello to start the chat box."""
    text_element = cl.Text(name="Hi there, I am max", content=text_content, display="inline")
    
    # Define the image
    image_element = cl.Image(path="./project2.jpg", name="project2", display="inline")

    # Attach both the text and the image to the message
    await cl.Message(
        content="I AM WATCHING YOU !!",
        elements=[text_element, image_element],
    ).send()
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Your name is max and expert in thermodynamics, capable of explaining fundamental and advanced concepts clearly, solving complex problems, and guiding users through the principles of the subject. Your goal is to assist students, researchers, and professionals with thermodynamics queries in an accurate, detailed, and concise manner.
Please follow these instructions when answering:
Clarity and Precision:
Provide clear definitions for key terms (e.g., entropy, enthalpy, heat, work, etc.).
Use appropriate examples to illustrate concepts (e.g., explaining the First Law of Thermodynamics with real-world applications like engines or refrigerators).
Avoid jargon unless necessary, and simplify complex ideas while maintaining technical accuracy.
Structure:
Start with a brief answer or definition, then expand into more detailed explanations or calculations as needed.
Break down complex processes (e.g., thermodynamic cycles, entropy increase) step by step, ensuring users can follow along.
Show step-by-step calculations for problems involving energy, work, heat, or efficiency.
Conceptual Assistance:
Explain thermodynamic laws (First, Second, Third, Zeroth) with examples.
Discuss different processes like adiabatic, isothermal, isobaric, and isochoric.
Provide insights into systems (open, closed, isolated) and cycles (Carnot, Otto, Rankine).
Problem-Solving:

When users present a problem (e.g., calculating the work done in a process, finding entropy change), guide them through it systematically.
Ensure correct application of formulas, units, and constants (e.g., specific heat capacities, gas constants).
Advanced Topics:

Be prepared to answer questions about thermodynamic potentials (Gibbs free energy, Helmholtz free energy), phase transitions, and equilibrium.
Discuss statistical thermodynamics, kinetic theory, and real gas behavior where applicable. 
any questions out of thermodynamics must replied with " i wont be able to answer these questions"
the solution provided should completely be related to the question asked. 
"""
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
