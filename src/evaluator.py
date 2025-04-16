from typing import List, Dict, Any
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
from datetime import datetime

class ResponseEvaluator:
    def __init__(self, api_key: str):
        self.evaluator_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            api_key=api_key
        )
        
        self.evaluation_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator of customer support responses. Analyze the following customer support interaction and provide a detailed evaluation.

User Question: {question}

AI Response: {response}

Context Used: {context}

AI's Reasoning: {reasoning}

Evaluate the response based on the following criteria and provide a score from 0-10 for each:

1. Relevance: How well does the response address the user's specific question?
2. Correctness: Is the information provided accurate and consistent with the context?
3. Completeness: Does the response fully address all aspects of the question?
4. Clarity: Is the response clear, well-structured, and easy to understand?
5. Empathy: Does the response show appropriate empathy and professional tone?
6. Context Usage: How effectively were the similar support conversations utilized?

Provide your evaluation in the following JSON format:
{
    "scores": {
        "relevance": score,
        "correctness": score,
        "completeness": score,
        "clarity": score,
        "empathy": score,
        "context_usage": score
    },
    "average_score": average,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "improvement_suggestions": ["suggestion1", "suggestion2"]
}
""")

    async def evaluate_response(
        self,
        question: str,
        response: str,
        context: List[str],
        reasoning: str
    ) -> Dict[str, Any]:
        """Evaluate a single response using the evaluation LLM"""
        
        # Format context for evaluation
        context_text = "\n".join([f"- {ctx}" for ctx in context])
        
        # Get evaluation from LLM
        chain = self.evaluation_prompt | self.evaluator_llm
        eval_response = chain.invoke({
            "question": question,
            "response": response,
            "context": context_text,
            "reasoning": reasoning
        })
        
        # Parse the evaluation JSON
        try:
            evaluation = json.loads(eval_response.content)
            evaluation["timestamp"] = datetime.now().isoformat()
            return evaluation
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse evaluation response",
                "raw_response": eval_response.content
            }

    def calculate_context_relevance(
        self,
        question_embedding: np.ndarray,
        context_embeddings: List[np.ndarray]
    ) -> List[float]:
        """Calculate semantic similarity between question and contexts"""
        scores = []
        for ctx_embedding in context_embeddings:
            similarity = np.dot(question_embedding, ctx_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(ctx_embedding)
            )
            scores.append(float(similarity))
        return scores

    def analyze_context_distribution(self, context_scores: List[float]) -> Dict[str, Any]:
        """Analyze the distribution of context relevance scores"""
        scores_array = np.array(context_scores)
        return {
            "mean_relevance": float(np.mean(scores_array)),
            "std_relevance": float(np.std(scores_array)),
            "max_relevance": float(np.max(scores_array)),
            "min_relevance": float(np.min(scores_array)),
            "score_distribution": {
                "high": len(scores_array[scores_array >= 0.7]),
                "medium": len(scores_array[(scores_array >= 0.4) & (scores_array < 0.7)]),
                "low": len(scores_array[scores_array < 0.4])
            }
        }

class EvaluationMetrics:
    def __init__(self):
        self.evaluations = []
        
    def add_evaluation(self, evaluation: Dict[str, Any]):
        """Add a new evaluation to the metrics"""
        self.evaluations.append(evaluation)
        
    def get_aggregate_metrics(self, n_recent: int = None) -> Dict[str, Any]:
        """Calculate aggregate metrics from stored evaluations"""
        evals = self.evaluations[-n_recent:] if n_recent else self.evaluations
        
        if not evals:
            return {"error": "No evaluations available"}
            
        # Calculate average scores for each criterion
        scores = {
            criterion: [] for criterion in 
            ["relevance", "correctness", "completeness", "clarity", "empathy", "context_usage"]
        }
        
        for eval in evals:
            if "scores" in eval:
                for criterion, score in eval["scores"].items():
                    scores[criterion].append(score)
        
        # Calculate metrics
        metrics = {
            "average_scores": {
                criterion: float(np.mean(criterion_scores))
                for criterion, criterion_scores in scores.items()
                if criterion_scores
            },
            "score_distributions": {
                criterion: {
                    "mean": float(np.mean(criterion_scores)),
                    "std": float(np.std(criterion_scores)),
                    "min": float(np.min(criterion_scores)),
                    "max": float(np.max(criterion_scores))
                }
                for criterion, criterion_scores in scores.items()
                if criterion_scores
            },
            "total_evaluations": len(evals),
            "time_period": {
                "start": evals[0]["timestamp"],
                "end": evals[-1]["timestamp"]
            } if evals else None
        }
        
        # Analyze common strengths and weaknesses
        all_strengths = [s for e in evals if "strengths" in e for s in e["strengths"]]
        all_weaknesses = [w for e in evals if "weaknesses" in e for w in e["weaknesses"]]
        
        from collections import Counter
        
        metrics["common_strengths"] = dict(Counter(all_strengths).most_common(5))
        metrics["common_weaknesses"] = dict(Counter(all_weaknesses).most_common(5))
        
        return metrics 