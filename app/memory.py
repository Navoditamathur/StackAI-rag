conversation_history = []


def add_turn(user, assistant):
    conversation_history.append({
        "user": user,
        "assistant": assistant
    })


def get_context(n=3):
    return conversation_history[-n:]