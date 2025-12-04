# topic_diversity.py

from collections import defaultdict

class TopicDiversity:
    def __init__(self, topk=10):
        self.topk = topk

    def score(self, model_output):
        topics = model_output["topics"]
        unique_words = set()
        total_words = 0

        for topic in topics:
            unique_words.update(topic[:self.topk])
            total_words += len(topic[:self.topk])

        return len(unique_words) / total_words if total_words else 0.0
