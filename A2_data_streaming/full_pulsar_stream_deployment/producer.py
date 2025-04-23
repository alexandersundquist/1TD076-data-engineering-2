# producer.py
import pulsar
import json
import time
from config import PULSAR_SERVICE_URL, INPUT_STRING, ITERATION, TOPIC_WORDS

print(f"--- Producer ---")
print(f"Connecting to Pulsar at {PULSAR_SERVICE_URL}")
print(f"Input String: '{INPUT_STRING}'")
print(f"Processing first {ITERATION} words.")
print(f"Target Topic: {TOPIC_WORDS}")

client = None
producer = None
try:
    # Create a Pulsar client
    client = pulsar.Client(PULSAR_SERVICE_URL)

    # Create a producer on the topic
    producer = client.create_producer(TOPIC_WORDS)

    words = INPUT_STRING.split()
    words_to_send = 0

    # Send the first ITERATION words
    for i, word in enumerate(words):
        if i >= ITERATION:
            print(f"Reached iteration limit ({ITERATION}). Stopping producer.")
            break

        message_data = {
            "word": word,
            "index": i
        }
        # Serialize data to JSON bytes
        message_body = json.dumps(message_data).encode('utf-8')

        # Send the message asynchronously
        producer.send_async(message_body, callback=lambda res, msg_id: print(f'Message sent successfully: ID={msg_id}, Data={message_data}') if res == pulsar.Result.Ok else print(f'Failed to send message: {res}'))
        words_to_send += 1

    # Ensure all messages are sent before exiting
    producer.flush()
    print(f"\nProducer finished sending {words_to_send} messages.")

except Exception as e:
    print(f"An error occurred in the producer: {e}")

finally:
    # Clean up
    if producer:
        try:
            producer.close()
            print("Producer closed.")
        except Exception as e:
            print(f"Error closing producer: {e}")
    if client:
        try:
            client.close()
            print("Pulsar client closed.")
        except Exception as e:
            print(f"Error closing client: {e}")
