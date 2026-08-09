from __future__ import annotations

import unittest

from fundamentals import (
    LockedMessageError,
    Message,
    SecretBox,
    WirelessLink,
    decode_message,
    encode_message,
)


class CommunicationTest(unittest.TestCase):
    def test_json_encoding_round_trips_unicode(self) -> None:
        message = Message(1, "robot", "STATE", "관찰 중")
        self.assertEqual(decode_message(encode_message(message)), message)

    def test_encoding_is_visible_and_is_not_encryption(self) -> None:
        payload = encode_message(Message(1, "robot", "STATE", "STOP"))
        self.assertIn(b"STOP", payload)

    def test_wireless_link_can_drop_a_known_sequence(self) -> None:
        link = WirelessLink(drop_sequences=frozenset({2}))
        self.assertIsNotNone(link.send(1, b"one").delivered)
        self.assertIsNone(link.send(2, b"two").delivered)

    def test_secret_box_hides_plaintext_and_round_trips(self) -> None:
        box = SecretBox.generate()
        message = Message(3, "teacher", "MODE", "observe-only")
        token = box.lock(message)
        self.assertNotIn(b"observe-only", token)
        self.assertEqual(box.unlock(token), message)

    def test_changed_ciphertext_is_rejected(self) -> None:
        box = SecretBox.generate()
        token = box.lock(Message(3, "teacher", "MODE", "observe-only"))
        changed = token[:-1] + bytes([token[-1] ^ 1])
        with self.assertRaises(LockedMessageError):
            box.unlock(changed)


if __name__ == "__main__":
    unittest.main()
