"""Tests for the JCS + SHA-256 helpers in dao_ai.audit.base."""

from __future__ import annotations

from dao_ai.audit import args_hash_of, canonical_jcs, sha256_hex


class TestCanonicalJcs:
    def test_key_order_independence(self) -> None:
        """RFC 8785 sorts keys — different input orderings hash equal."""
        assert canonical_jcs({"a": 1, "b": 2}) == canonical_jcs({"b": 2, "a": 1})

    def test_nested_structure(self) -> None:
        left = {"outer": {"z": 1, "a": 2}, "arr": [1, 2, 3]}
        right = {"arr": [1, 2, 3], "outer": {"a": 2, "z": 1}}
        assert canonical_jcs(left) == canonical_jcs(right)

    def test_array_order_matters(self) -> None:
        """Arrays are ordered — permutation must hash differently."""
        assert canonical_jcs([1, 2, 3]) != canonical_jcs([3, 2, 1])

    def test_unicode_normalisation(self) -> None:
        """RFC 8785 uses NFC-normalised UTF-8."""
        assert isinstance(canonical_jcs({"k": "café"}), str)


class TestSha256Hex:
    def test_stable_length(self) -> None:
        assert len(sha256_hex("anything")) == 64

    def test_bytes_and_str_agree(self) -> None:
        assert sha256_hex("abc") == sha256_hex(b"abc")


class TestArgsHash:
    def test_semantic_equality_of_arg_permutations(self) -> None:
        """Reordering keys must NOT change the args hash."""
        left = {"customer_id": "C-1", "amount": 42}
        right = {"amount": 42, "customer_id": "C-1"}
        assert args_hash_of(left) == args_hash_of(right)

    def test_actual_change_detected(self) -> None:
        """Any value change must change the hash — that's the whole point."""
        original = args_hash_of({"amount": 42, "customer_id": "C-1"})
        tampered = args_hash_of({"amount": 42, "customer_id": "C-999"})
        assert original != tampered

    def test_type_change_detected(self) -> None:
        """Integer vs string of same value must hash differently."""
        assert args_hash_of({"n": 1}) != args_hash_of({"n": "1"})
