"""Step 6: lock a message, unlock it, and reject changed ciphertext."""

from fundamentals import LockedMessageError, Message, SecretBox, encode_message


message = Message(sequence=7, sender="teacher", kind="MODE", value="observe-only")
plain = encode_message(message)
box = SecretBox.generate()
locked = box.lock(message)

print("ENCODED, READABLE", plain)
print("ENCRYPTED BYTES  ", locked[:36], b"...")
print("UNLOCKED         ", box.unlock(locked))

changed = locked[:-1] + bytes([locked[-1] ^ 1])
try:
    box.unlock(changed)
except LockedMessageError as error:
    print("CHANGED MESSAGE REJECTED:", error)

print("The secret key is intentionally never printed or committed.")
