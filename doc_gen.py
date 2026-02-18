import json
import random

# Categories to make the data realistic
categories = [
    "Login & Authentication",
    "Billing & Payments",
    "Technical Bugs",
    "Feature Requests",
    "Mobile App Issues"
]

# Templates to randomize the phrasing
templates = [
    "I'm having trouble with {topic}. {detail}",
    "Why is my {topic} not working? {detail}",
    "I need help with {topic}. {detail}",
    "Urgent: {topic} issue. {detail}",
    "Can you check my {topic}? {detail}"
]

details = [
    "It keeps giving me an error 404.",
    "I've tried resetting my router but nothing changes.",
    "This started happening after the last update.",
    "I need this fixed before my meeting tomorrow.",
    "The screen just freezes."
]

documents = []

for i in range(115):
    # Pick random components
    category = random.choice(categories)
    template = random.choice(templates)
    detail = random.choice(details)
    
    # Construct a fake ticket
    # In a real scenario, you'd ask an LLM to write this, but simple string formatting 
    # is free and works for testing pipelines.
    topic = category.lower().replace("&", "and")
    text = template.format(topic=topic, detail=detail)
    
    doc = {
        "id": i,
        "content": text,
        "category": category
    }
    documents.append(doc)

# Save to file
with open("documents.json", "w") as f:
    json.dump(documents, f, indent=2)

print(f"Generated {len(documents)} documents in documents.json")