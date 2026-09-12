import json
import ollama


# ==========================================
# PIPLE CONFIGURATION
# ==========================================

OLLAMA_MODEL = "qwen3-4b-local"

ALLOWED_ACTIONS = {
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "stop",
    "none",
}

ACTION_DESCRIPTIONS = {
    "move_forward": "Move the robot forward.",
    "move_backward": "Move the robot backward.",
    "turn_left": "Turn the robot left.",
    "turn_right": "Turn the robot right.",
    "stop": "Stop the robot.",
    "none": "No physical robot action is required.",
}


# ==========================================
# SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = (
    "You are Piple, a physical mobile robot powered by an AI model. "

    "Your task is to understand the user's intention and determine "
    "whether a physical robot action is required. "

    "AVAILABLE ACTIONS: "
    "move_forward, "
    "move_backward, "
    "turn_left, "
    "turn_right, "
    "stop, "
    "none. "

    "ACTION RULES: "
    "Use move_forward when the user asks the robot to move forward. "
    "Use move_backward when the user asks the robot to move backward. "
    "Use turn_left when the user asks the robot to turn left. "
    "Use turn_right when the user asks the robot to turn right. "
    "Use stop when the user asks the robot to stop. "
    "Use none when the user is asking a question, having a conversation, "
    "or when no physical robot action is required. "

    "LANGUAGE RULE: "
    "Write the response in the same language as the user's latest message. "

    "OUTPUT RULE: "
    "Return valid JSON with exactly two fields: "
    "\"action\" and \"response\". "

    "The action field must contain exactly one allowed action name. "
    "The response field must contain a short natural response to the user. "

    "Do not use markdown. "
    "Do not add any text outside the JSON."
)


# ==========================================
# ACTION HANDLER
# ==========================================

def execute_action(action):
    """
    Execute a robot action.

    For now this function ONLY simulates the action.
    No motors or ROS 2 commands are sent yet.
    """

    if action == "move_forward":
        print("🤖 ACTION: MOVE FORWARD")

    elif action == "move_backward":
        print("🤖 ACTION: MOVE BACKWARD")

    elif action == "turn_left":
        print("🤖 ACTION: TURN LEFT")

    elif action == "turn_right":
        print("🤖 ACTION: TURN RIGHT")

    elif action == "stop":
        print("🤖 ACTION: STOP")

    elif action == "none":
        print("🤖 ACTION: NONE")

    else:
        print(f"⚠️ Unknown action: {action}")


# ==========================================
# QWEN3
# ==========================================

def think(user_text, messages):

    messages.append(
        {
            "role": "user",
            "content": user_text
        }
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        think=False,
        format="json"
    )

    raw_response = response["message"]["content"]

    return raw_response


# ==========================================
# PARSE AI RESPONSE
# ==========================================

def parse_response(raw_response):

    try:
        data = json.loads(raw_response)

    except json.JSONDecodeError:
        print("⚠️ Invalid JSON from Qwen3.")

        return {
            "action": "none",
            "response": "I could not understand that."
        }

    if not isinstance(data, dict):
        print("⚠️ Qwen3 returned JSON, but not an object.")

        return {
            "action": "none",
            "response": "I could not understand that."
        }

    action = data.get("action")
    answer = data.get("response")

    if action not in ALLOWED_ACTIONS:
        print(f"⚠️ Invalid action: {action}")
        action = "none"

    if not isinstance(answer, str):
        answer = ""

    return {
        "action": action,
        "response": answer
    }

# ==========================================
# MAIN
# ==========================================

print("=================================")
print("🤖 Piple Action Brain is running!")
print("Type 'exit' to quit.")
print("=================================")


messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]


while True:

    user_input = input("\nYou: ")
    if not user_input.strip():
        continue

    if user_input.lower() in ["exit", "quit"]:
        print("Piple: See you later!")
        break

    # --------------------------------------
    # AI
    # --------------------------------------

    raw_response = think(
        user_input,
        messages
    )

    # --------------------------------------
    # Parse JSON
    # --------------------------------------

    result = parse_response(
        raw_response
    )

    action = result["action"]
    piple_response = result["response"]

    # --------------------------------------
    # Display
    # --------------------------------------

    print("\n🤖 Piple:")
    print(piple_response)

    print("\n🧠 Parsed action:")
    print(action)

    # --------------------------------------
    # Execute action
    # --------------------------------------

    execute_action(action)

    # --------------------------------------
    # Conversation memory
    # --------------------------------------

    messages.append(
        {
            "role": "assistant",
            "content": raw_response
        }
    )