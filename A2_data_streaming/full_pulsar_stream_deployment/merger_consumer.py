# merger_consumer.py
import pulsar
import json
import time
from config import PULSAR_SERVICE_URL, TOPIC_PROCESSED, MERGER_SUBSCRIPTION, ITERATION

print(f"--- Merger Consumer ---")
print(f"Connecting to Pulsar at {PULSAR_SERVICE_URL}")
print(f"Listening on Topic: {TOPIC_PROCESSED}")
print(f"Subscription: {MERGER_SUBSCRIPTION}")
print(f"Expecting {ITERATION} results.")

client = None
consumer = None
results = {} # Dictionary to store index: processed_word

try:
    client = pulsar.Client(PULSAR_SERVICE_URL)

    # Create consumer - Exclusive or Failover is fine here as only one merger runs
    consumer = client.subscribe(
        TOPIC_PROCESSED,
        MERGER_SUBSCRIPTION,
        consumer_type=pulsar.ConsumerType.Exclusive # Only one merger instance
    )

    print("Merger started. Waiting for processed messages...")

    # Loop until we have the expected number of results
    while len(results) < ITERATION:
        msg = None
        try:
            # Wait for a message, add a timeout (e.g., 60 seconds)
            # to avoid waiting forever if a message is lost upstream.
            msg = consumer.receive(timeout_millis=60000) # 60 seconds timeout

            message_data = json.loads(msg.data().decode('utf-8'))
            processed_word = message_data.get('processed_word')
            index = message_data.get('index')

            if processed_word is None or index is None:
                 print(f"Received malformed message: {msg.data().decode('utf-8')}")
                 consumer.acknowledge(msg)
                 continue

            if index not in results:
                results[index] = processed_word
                print(f"Received result: Index={index}, Word='{processed_word}' (Total: {len(results)}/{ITERATION})")
            else:
                 print(f"Received duplicate result for Index={index}. Ignoring.")

            # Acknowledge the message
            consumer.acknowledge(msg)

        except pulsar.exceptions.Timeout:
             print(f"Timeout waiting for message. Have {len(results)}/{ITERATION} results. Still waiting...")
             # Decide if you want to break or continue waiting after a timeout
             # For this example, we'll continue waiting.
             continue
        except Exception as e:
            print(f"Error processing message: {e}")
            if msg:
                # Don't acknowledge on error, let Pulsar redeliver
                consumer.negative_acknowledge(msg)
                print(f"Negatively acknowledged message: ID={msg.message_id()}")
            time.sleep(1) # Avoid fast loop on error

    # --- Merging Step ---
    print("\nReceived all expected results. Merging...")
    if len(results) == ITERATION:
        try:
            # Ensure all indices from 0 to ITERATION-1 are present
            sorted_words = [results[i] for i in range(ITERATION)]
            final_string = " ".join(sorted_words)
            print("-" * 30)
            print(f"Final Resultant String: {final_string}")
            print("-" * 30)
        except KeyError as e:
             print(f"Error: Missing result for index {e}. Cannot form final string.")
             print(f"Received results: {results}")
    else:
        # This part might be reached if timeout logic changes
        print(f"Error: Did not receive all {ITERATION} results. Only got {len(results)}.")
        print(f"Received results: {results}")

except KeyboardInterrupt:
    print("\nMerger consumer shutting down...")
except Exception as e:
    print(f"A critical error occurred in the merger: {e}")
finally:
    # Clean up
    if consumer:
        try:
            consumer.close()
            print("Consumer closed.")
        except Exception as e:
            print(f"Error closing consumer: {e}")
    if client:
        try:
            client.close()
            print("Pulsar client closed.")
        except Exception as e:
            print(f"Error closing client: {e}")
