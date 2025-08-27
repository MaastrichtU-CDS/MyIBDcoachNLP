#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
from bertopic import BERTopic
import pickle

def automatic_merge_topics(hierarchical_topics, topic_model, distance_threshold = 0.7):
    
    topics_to_merge = hierarchical_topics["Topics"][hierarchical_topics["distance"] <= distance_threshold]

    if topics_to_merge:
        topic_model.merge_topics()