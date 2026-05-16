
import os
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
# Guardrails imports
try:
    from guardrails import Guard, OnFailAction
except ModuleNotFoundError:
    # Minimal stubs so the file parses and GuardRails wrappers become no-ops
    class OnFailAction:
        EXCEPTION = "exception"
        FIX = "fix"
        REFRAIN = "refrain"
        FILTER = "filter"
        NONE = "none"

    class Guard:
        def __init__(self, *args, **kwargs):
            pass
        @classmethod
        def from_pydantic(cls, *args, **kwargs):
            return cls()
        @classmethod
        def from_rail_string(cls, *args, **kwargs):
            return cls()
        def use_many(self, *args, **kwargs):
            return self
        def use(self, *args, **kwargs):
            return self
        def __call__(self, func):
            return func
        def parse(self, text, *args, **kwargs):
            return text
        def validate(self, text, *args, **kwargs):
            return text

# For custom PII detection using Presidio
#import spacy

# Load environment variables
load_dotenv()

# Get OPENAI_API_KEY from google.colab.userdata if running in Colab
#from google.colab import userdata
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


client = OpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = 'gpt-4o-mini'
system_prompt_openai = "You are a helpful assistant." # Default system prompt for general OpenAI chat

def check_moderation(text: str):
    """
    Sends text to the OpenAI Moderation API and returns the result.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string.")

    try:
        response = client.moderations.create(
            model="omni-moderation-latest",  # Latest moderation model
            input=text
        )
        return response
    except Exception as e:
        print(f"Error calling moderation API: {e}")
        return None

# --- OpenAI Moderation API related functions ---
def check_moderation_flag(text: str) -> bool:
    """
    Sends text to the OpenAI Moderation API and returns True if flagged, False otherwise.
    Returns False and prints error if API call fails.
    """
    if not text or not isinstance(text, str):
        print("Error: Input for moderation must be a non-empty string.")
        return False # Or raise ValueError, depending on desired strictness

    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=text
        )
        return response.results[0].flagged
    except Exception as e:
        print(f"Error calling moderation API: {e}")
        return False # Fail-open: If moderation fails, let the request proceed but log the error

async def get_chat_response_openai_async(user_request: str, system_prompt: str = system_prompt_openai) -> str:
    """
    Gets a chat response from OpenAI's GPT model.
    Includes error handling for API calls.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL, messages=messages, temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting chat response from OpenAI: {e}")
        # Return a generic error message to the user, log full error for debugging
        return "I apologize, but I encountered an issue while generating a response. Please try again."


async def execute_chat_with_input_moderation(user_request: str) -> str:
    """
    Executes a chat request with input moderation.
    Handles errors from moderation checks and chat response generation.
    """
    print("Getting LLM response (with input moderation)")
    moderation_task = asyncio.create_task(check_moderation_flag(user_request))
    chat_task = asyncio.create_task(get_chat_response_openai_async(user_request))

    try:
        while True:
            done, pending = await asyncio.wait(
                [moderation_task, chat_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Check moderation task result if it's done
            if moderation_task in done:
                is_flagged = moderation_task.result() # This will be False if check_moderation_flag had an internal error
                if is_flagged:
                    # If input is flagged, cancel chat and return moderation message
                    chat_task.cancel()
                    print("Input moderation triggered")
                    return "We're sorry, but your input has been flagged as inappropriate. Please rephrase your input and try again."
                # If not flagged, or moderation check failed (returned False), then proceed to chat_task
            
            # Check chat task result if it's done
            if chat_task in done:
                # get_chat_response_openai_async already handles its own exceptions and returns an error string
                response = chat_task.result()
                print("Got LLM response (input passed moderation)")
                return response

            # If neither task is done (meaning moderation_task isn't done either), wait
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        print("A task was cancelled during execution of execute_chat_with_input_moderation.")
        return "Your request was interrupted."
    except Exception as e:
        print(f"An unexpected error occurred in execute_chat_with_input_moderation: {e}")
        return "An unexpected internal error occurred."


async def execute_all_moderations(user_request: str) -> str:
    """
    Executes a chat request with both input and output moderation.
    Includes error handling.
    """
    print("Getting LLM response (with input and output moderation)")
    input_moderation_task = asyncio.create_task(check_moderation_flag(user_request))
    chat_task = asyncio.create_task(get_chat_response_openai_async(user_request))

    try:
        while True:
            done, pending = await asyncio.wait(
                [input_moderation_task, chat_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Check input moderation task first
            if input_moderation_task in done:
                input_flagged = input_moderation_task.result()
                if input_flagged:
                    chat_task.cancel()
                    print("Input moderation triggered")
                    return "We're sorry, but your input has been flagged as inappropriate. Please rephrase your input and try again."

            # If chat task is done (and input wasn't flagged)
            if chat_task in done:
                chat_response = chat_task.result() # This can contain an error message if get_chat_response_openai_async failed
                
                # If chat_response itself is an error message, we might want to return it directly
                # Or, if get_chat_response_openai_async returns non-error strings, proceed with output moderation
                if chat_response.startswith("I apologize, but I encountered an issue"):
                    return chat_response # Return the error from chat generation

                # Perform output moderation only if chat response was successfully generated
                output_moderation_response = await check_moderation_flag(chat_response)

                # Check if output moderation is triggered
                if output_moderation_response == True:
                    print("Output moderation triggered")
                    return "Sorry, we're not permitted to give this answer. I can help you with any general queries you might have."

                print('Passed moderation')
                return chat_response

            # If neither task is completed, sleep for a bit before checking again
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        print("A task was cancelled during execution of execute_all_moderations.")
        return "Your request was interrupted."
    except Exception as e:
        print(f"An unexpected error occurred in execute_all_moderations: {e}")
        return "An unexpected internal error occurred."


def custom_moderation(content: str, parameters: str) -> str:
    """
    Uses GPT model for custom content moderation based on specified parameters.
    Includes error handling.
    """
    prompt = f"""Please assess the following content for any inappropriate material. You should base your assessment on the given parameters.
    Your answer should be in json format with the following fields:
        - flagged: a boolean indicating whether the content is flagged for any of the categories in the parameters
        - reason: a string explaining the reason for the flag, if any
        - parameters: a dictionary of the parameters used for the assessment and their values
    Parameters: {parameters}\n\nContent:\n{content}\n\nAssessment:"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a content moderation assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        assessment = response.choices[0].message.content
        return assessment
    except Exception as e:
        print(f"Error during custom moderation: {e}")
        return f'{{"flagged": true, "reason": "Error during custom moderation: {e}", "parameters": null}}'


def check_image_moderation(image_url: str) -> bool:
    """
    Checks if an image is appropriate using OpenAI Moderation API.
    Returns True if safe (not flagged), False if flagged or an error occurs.
    """
    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=[{"type": "image_url", "image_url": {"url": image_url}}]
        )
        return not response.results[0].flagged # True if not flagged, False if flagged
    except Exception as e:
        print(f"Error checking image moderation: {e}")
        return False # Fail-safe: If moderation fails, consider image not safe (or handle as per policy)

# --- Guardrails AI specific functions ---
system_prompt_guardrails = "You are a helpful assistant." # Default system prompt for Guardrails chat

async def get_chat_response_guardrails_async(user_request: str, system_prompt: str = system_prompt_guardrails) -> str:
    """
    Gets a chat response from OpenAI's GPT model for use with Guardrails.
    Includes error handling for API calls.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL, messages=messages, temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting chat response for Guardrails: {e}")
        return "I apologize, but I encountered an issue while generating a response using Guardrails. Please try again."


async def topical_guardrail(user_request: str) -> str:
    """
    Checks if the user's question is on an allowed topic using a specific system prompt.
    Includes error handling for API calls.
    """
    messages = [
        {
            "role": "system",
            "content": "Your role is to assess whether the user question is allowed or not. The allowed topics are cats and dogs. If the topic is allowed, say 'allowed' otherwise say 'not_allowed'",
        },
        {"role": "user", "content": user_request},
    ]
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL, messages=messages, temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during topical guardrail check: {e}")
        return "error_checking_topic" # Return a specific string for error in guardrail


async def execute_chat_with_guardrail(user_request: str) -> str:
    """
    Executes a chat request with a topical guardrail.
    Handles errors from topical guardrail checks and chat response generation.
    """
    print("Checking topical guardrail")
    topical_guardrail_task = asyncio.create_task(topical_guardrail(user_request))
    chat_task = asyncio.create_task(get_chat_response_guardrails_async(user_request))

    try:
        while True:
            done, pending = await asyncio.wait(
                [topical_guardrail_task, chat_task], return_when=asyncio.FIRST_COMPLETED
            )
            if topical_guardrail_task in done:
                guardrail_response = topical_guardrail_task.result()
                if guardrail_response == "error_checking_topic":
                    chat_task.cancel() # Cancel chat if guardrail check failed
                    print("Error occurred during topical guardrail check.")
                    return "An error occurred while checking the topic. Please try again."
                elif guardrail_response.strip().lower() == "not_allowed":
                    chat_task.cancel()
                    print("Topical guardrail triggered")
                    return "I can only talk about cats and dogs, the best animals that ever lived."
                # If guardrail_response is "allowed", and chat_task is also done,
                # then proceed to get chat_task.result()
                elif chat_task in done:
                    chat_response = chat_task.result()
                    print("Guardrail passed, got LLM response")
                    return chat_response
            elif chat_task in done: # If chat_task finishes before guardrail_task, and guardrail_task is not an error
                # This branch implies topical_guardrail_task is still pending or not in done, but chat is done.
                # If topical_guardrail_task is pending, we still need its result before returning chat.
                # So this else-if should ideally not return unless topical_guardrail_task is also handled.
                # However, the current logic is to check if topical_guardrail_task is in done first.
                # If topical_guardrail_task is still pending, this means the first `if` branch was false,
                # so we should not proceed with chat_task.result() until topical_guardrail_task is resolved.
                # The `await asyncio.sleep(0.1)` will handle waiting for both.
                pass # This is important to not return prematurely if guardrail_task is pending.
            
            await asyncio.sleep(0.1) # Wait if neither task is completed or fully processed
    except asyncio.CancelledError:
        print("A task was cancelled during execution of execute_chat_with_guardrail.")
        return "Your request was interrupted."
    except Exception as e:
        print(f"An unexpected error occurred in execute_chat_with_guardrail: {e}")
        return "An unexpected internal error occurred."


# --- Custom PII Detection Function ---
# This will be used as an alternative/addition to the guardrails.hub.DetectPII validator
# if it continues to have issues with recognizers.

_analyzer_engine = None

def _get_analyzer_engine():
    global _analyzer_engine
    if _analyzer_engine is None:
        try:
            # Ensure the spaCy model is loaded
            nlp = spacy.load("en_core_web_lg")
            # Create a RecognizerRegistry and add spaCy recognizer
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine={"en": nlp})
            _analyzer_engine = AnalyzerEngine(registry=registry, nlp_engine={"en": nlp})
        except Exception as e:
            print(f"Error initializing Presidio AnalyzerEngine: {e}")
            _analyzer_engine = None # Set to None on failure
    return _analyzer_engine

def detect_pii_with_presidio(text: str, entities: list = None) -> dict:
    """
    Detects PII in the given text using Presidio Analyzer.
    Args:
        text (str): The input text to analyze.
        entities (list): Optional list of entities to detect (e.g., ["EMAIL_ADDRESS", "PHONE_NUMBER"]).
                         If None, all default entities will be detected.
    Returns:
        dict: A dictionary containing detected PII information.
              Returns an error dict if analyzer initialization fails or an error occurs during detection.
    """
    analyzer = _get_analyzer_engine()
    if not analyzer:
        return {"flagged": False, "error": "Presidio Analyzer not initialized."}

    try:
        if entities:
            results = analyzer.analyze(text=text, language="en", entities=entities)
        else:
            results = analyzer.analyze(text=text, language="en")

        detected_pii = []
        for res in results:
            detected_pii.append({
                "entity_type": res.entity_type,
                "start": res.start,
                "end": res.end,
                "score": res.score,
                "text": text[res.start:res.end]
            })
        return {"flagged": bool(detected_pii), "pii_details": detected_pii}
    except Exception as e:
        print(f"Error during PII detection: {e}")
        return {"flagged": False, "error": f"PII detection failed: {e}"}