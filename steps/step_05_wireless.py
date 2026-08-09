"""Step 5: structured data becomes bytes and crosses an imperfect link."""

from fundamentals import Message, WirelessLink, decode_message, encode_message


message = Message(sequence=1, sender="laptop", kind="STATUS", value="observe")
air_bytes = encode_message(message)
print("MESSAGE", message)
print("BYTES  ", air_bytes)

link = WirelessLink(latency_ms=45, drop_sequences=frozenset({2}))
received = link.send(message.sequence, air_bytes)
print("LINK 1 ", received)
print("READ 1 ", decode_message(received.delivered))

lost = link.send(2, encode_message(Message(2, "laptop", "STATUS", "again")))
print("LINK 2 ", lost)
print("SAFE OUTPUT WHEN LOST: HOLD / STOP; never reuse an old movement command")
