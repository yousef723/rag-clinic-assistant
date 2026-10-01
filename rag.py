import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import openai
from dotenv import load_dotenv

# ============ CONFIG ============
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "clinic-assistant"

# ============ STEP 1: SAMPLE CLINIC DOCUMENT ============
clinic_policy = """
Insurance Policy: We accept Blue Cross Blue Shield, Aetna, and UnitedHealthcare.
Patients without insurance can use our self-pay program, which offers a 20% discount.

Cancellation Policy: Appointments must be cancelled at least 24 hours in advance.
Late cancellations or no-shows will incur a $50 fee.

Appointment Prep: For blood work, patients should fast for 8 hours before the appointment.
For general check-ups, no special preparation is required.

Office Hours: The clinic is open Monday to Friday, 8am to 5pm.
We are closed on weekends and major holidays.

New Patients: New patients should arrive 15 minutes early to complete paperwork.
Please bring a valid photo ID and your insurance card.
"""

with open("clinic_policy.txt", "w") as f:
    f.write(clinic_policy)

# ============ STEP 2: LOAD + CHUNK ============
with open("clinic_policy.txt", "r") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"Created {len(chunks)} chunks")

# ============ STEP 3: EMBED ============
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
print(f"Created {len(embeddings)} embeddings")

# ============ STEP 4: STORE IN PINECONE ============
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

vectors = [
    {"id": f"chunk_{i}", "values": embeddings[i].tolist(), "metadata": {"text": chunks[i]}}
    for i in range(len(chunks))
]
index.upsert(vectors=vectors)
print(f"Stored {len(vectors)} chunks in Pinecone")

# ============ STEP 5: RETRIEVE + GENERATE ============
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def answer_question(query):
    query_embedding = model.encode([query]).tolist()
    results = index.query(vector=query_embedding, top_k=2, include_metadata=True)

    context = "\n".join([match['metadata']['text'] for match in results['matches']])

    prompt = f"""Answer using ONLY this information. If not in the context, say "I don't know."

Context:
{context}

Question: {query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ============ TEST ============
if __name__ == "__main__":
    question = "Do I need to fast before my appointment?"
    answer = answer_question(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")