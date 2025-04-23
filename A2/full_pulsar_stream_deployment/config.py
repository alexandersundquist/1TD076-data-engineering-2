# config.py
import os

# Get Pulsar URL from environment variable or use default
PULSAR_SERVICE_URL = os.getenv('PULSAR_URL', 'pulsar://localhost:6650')

# Original input data
INPUT_STRING = "I want to be capitalized right now please"
ITERATION = 5 # Process the first 5 words

# Topic Names
TOPIC_WORDS = 'persistent://public/default/words-to-process'
TOPIC_PROCESSED = 'persistent://public/default/processed-words'

# Subscription names (consumers in the same group share messages)
WORKER_SUBSCRIPTION = 'my-worker-subscription'
MERGER_SUBSCRIPTION = 'my-merger-subscription'
