from sentence_transformers import SentenceTransformer, util
import re
from typing import List, Tuple

class LoanRAGChatbot:
    def __init__(self, file_path: str):
        # Read the document
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        # Split into applicant records (each record starts with "Applicant")
        self.docs = re.split(r"Applicant\s+LP\d+", self.text)[1:]  # Remove initial empty split
        self.docs = ["Applicant " + doc for doc in self.docs]  # Reattach "Applicant" prefix

        # Embedding model for semantic search
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Better for semantic similarity
        self.embeddings = self.embedder.encode(self.docs, convert_to_tensor=True)

    def get_answer(self, query: str, top_k: int = 3) -> Tuple[str, List[str]]:
        # Embed the query
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)

        # Semantic search to get top_k relevant chunks
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)[0]
        retrieved_chunks = [self.docs[hit['corpus_id']] for hit in hits]

        # Simple rule-based answer generation
        answer = self.generate_rule_based_answer(query, retrieved_chunks)
        if not answer:
            # Fallback to keyword matching if no rule applies
            answer = self.extract_keyword_answer(query, retrieved_chunks)

        return answer, retrieved_chunks

    def generate_rule_based_answer(self, query: str, chunks: List[str]) -> str:
        query = query.lower()
        approved_applicants = [chunk for chunk in chunks if "loan was approved" in chunk.lower()]
        rejected_applicants = [chunk for chunk in chunks if "loan was rejected" in chunk.lower()]

        if "approved" in query or "get loan" in query:
            if approved_applicants:
                common_traits = self.find_common_traits(approved_applicants)
                return f"Loans are often approved for applicants with traits like: {common_traits}."
            return "No clear pattern for approved loans found in the data."
        elif "rejected" in query or "denied" in query:
            if rejected_applicants:
                common_traits = self.find_common_traits(rejected_applicants)
                return f"Loans are often rejected for applicants with traits like: {common_traits}."
            return "No clear pattern for rejected loans found in the data."
        return ""

    def find_common_traits(self, chunks: List[str]) -> str:
        traits = {"male": 0, "female": 0, "married": 0, "unmarried": 0, "graduate": 0, "not graduate": 0,
                  "self-employed": 0, "not self-employed": 0, "credit history 1": 0, "credit history 0": 0,
                  "urban": 0, "rural": 0, "semiurban": 0}
        total = len(chunks)

        for chunk in chunks:
            if "male" in chunk.lower(): traits["male"] += 1
            if "female" in chunk.lower(): traits["female"] += 1
            if "married" in chunk.lower(): traits["married"] += 1
            if "unmarried" in chunk.lower(): traits["unmarried"] += 1
            if "graduate" in chunk.lower(): traits["graduate"] += 1
            if "not graduate" in chunk.lower(): traits["not graduate"] += 1
            if "self-employed" in chunk.lower(): traits["self-employed"] += 1
            if "not self-employed" in chunk.lower(): traits["not self-employed"] += 1
            if "credit history 1" in chunk.lower(): traits["credit history 1"] += 1
            if "credit history 0" in chunk.lower(): traits["credit history 0"] += 1
            if "urban" in chunk.lower(): traits["urban"] += 1
            if "rural" in chunk.lower(): traits["rural"] += 1
            if "semiurban" in chunk.lower(): traits["semiurban"] += 1

        common = [k for k, v in traits.items() if v / total > 0.5]  # Majority rule
        return ", ".join(common) if common else "no dominant traits"

    def extract_keyword_answer(self, query: str, chunks: List[str]) -> str:
        query_terms = set(query.lower().split())
        for chunk in chunks:
            if any(term in chunk.lower() for term in query_terms):
                return f"Relevant info: {chunk.split('The loan was')[0].strip()}."
        return "No relevant information found."

# Example usage (for testing)
if __name__ == "__main__":
    bot = LoanRAGChatbot("loan_knowledge_base.txt")
    answer, contexts = bot.get_answer("What kind of applicants get loans approved?")
    print("Answer:", answer)
    print("Contexts:", contexts)