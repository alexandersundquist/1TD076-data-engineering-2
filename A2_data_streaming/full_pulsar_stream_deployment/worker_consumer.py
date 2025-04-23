# worker_consumer.py
import pulsar
import json
import time
from config import PULSAR_SERVICE_URL, TOPIC_WORDS, TOPIC_PROCESSED, WORKER_SUBSCRIPTION

print(f"--- Worker Consumer ---")
print(f"Connecting to Pulsar at {PULSAR_SERVICE_URL}")
print(f"Listening on Topic: {TOPIC_WORDS}")
print(f"Subscription: {WORKER_SUBSCRIPTION}")
print(f"Publishing results to Topic: {TOPIC_PROCESSED}")

client = None
consumer = None
processed_producer = None

try:
    client = pulsar.Client(PULSAR_SERVICE_URL)

    # Create producer for processed words topic FIRST
    processed_producer = client.create_producer(TOPIC_PROCESSED)

    # Create consumer
    consumer = client.subscribe(
        TOPIC_WORDS,
        WORKER_SUBSCRIPTION,
        consumer_type=pulsar.ConsumerType.Shared # Allows multiple workers
    )

    print("Worker started. Waiting for messages...")

    while True: # Keep running indefinitely
        msg = None
        try:
            # Wait for a message
            msg = consumer.receive() # Blocks until a message is available
            message_data = json.loads(msg.data().decode('utf-8'))
            word = message_data.get('word')
            index = message_data.get('index')

            if word is None or index is None:
                 print(f"Received malformed message (missing word or index): {msg.data().decode('utf-8')}")
                 consumer.acknowledge(msg) # Acknowledge malformed message to remove it
                 continue

            print(f"\nReceived message: ID={msg.message_id()}, Index={index}, Word='{word}'")

            # --- Processing Step ---
            processed_word = word.upper() # Apply the conversion
            print(f"Processed: Index={index}, Result='{processed_word}'")
            # -----------------------

            output_data = {
                "processed_word": processed_word,
                "index": index
            }
            output_body = json.dumps(output_data).encode('utf-8')

            # Send the processed result to the next topic
            processed_producer.send_async(output_body, callback=lambda res, msg_id: print(f'Result sent successfully: ID={msg_id}') if res == pulsar.Result.Ok else print(f'Failed to send result: {res}'))
            processed_producer.flush() # Ensure it sends reasonably quickly

            # Acknowledge the message successfully processed
            consumer.acknowledge(msg)
            print(f"Acknowledged message: ID={msg.message_id()}")

        except Exception as e:
            print(f"Error processing message: {e}")
            # Negative acknowledgement will make Pulsar redeliver the message later
            if msg:
                consumer.negative_acknowledge(msg)
                print(f"Negatively acknowledged message: ID={msg.message_id()}")
            # Add a small delay to prevent rapid failure loops
            time.sleep(1)


except KeyboardInterrupt:
     print("\nWorker consumer shutting down...")
except Exception as e:
    print(f"A critical error occurred in the worker: {e}")
finally:
    # Clean up resources
    if consumer:
        try:
            consumer.close()
            print("Consumer closed.")
        except Exception as e:
            print(f"Error closing consumer: {e}")
    if processed_producer:
         try:
            processed_producer.close()
            print("Result Producer closed.")
         except Exception as e:
            print(f"Error closing result producer: {e}")
    if client:
        try:
            client.close()
            print("Pulsar client closed.")
        except Exception as e:
            print(f"Error closing client: {e}")
