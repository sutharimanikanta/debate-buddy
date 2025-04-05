# C:\Users\MANIKANTA\Desktop\R\RAG\debate\services\conversations.py
class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class Conversations:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        message = Message(role, content)
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def get_message_dict_list(self):
        return [m.to_dict() for m in self.messages]

    def get_messages_by_role(self, role):
        return [message for message in self.messages if message.role == role]
